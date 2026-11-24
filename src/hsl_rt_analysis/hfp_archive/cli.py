"""hfp-archive: convert the HSL HFP archive on the storage box to the archive format v1, day by day.

Commands:
  run               advance days through fetch -> convert -> verify -> upload -> swap -> done
  status            progress per day
  report            days that need a decision (interim only, C7, failures, reappeared originals)
  retry             put FAILED / NEEDS_REVIEW days back where they stopped
  audit             re-check converted days on the storage box alone (needed before purging)
  purge-superseded  delete _superseded/ data of audited days before a date (dry run unless --execute)
  restore           rebuild a day's raw files from its Parquet files

The storage box is given with --remote or HFP_ARCHIVE_REMOTE (user@host, SSH port 23).
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import duckdb

from hsl_rt_analysis.hfp_archive import ops
from hsl_rt_analysis.hfp_archive.config import RawArchivePolicy, Settings, hfp_day, validate_day
from hsl_rt_analysis.hfp_archive.discover import cutoff_day, inventory, select_days
from hsl_rt_analysis.hfp_archive.guards import AlreadyRunningError, RunLock, StopFlag
from hsl_rt_analysis.hfp_archive.pipeline import Pipeline
from hsl_rt_analysis.hfp_archive.remote import Remote, RemoteUnavailableError, SentinelMissingError, SshRemote
from hsl_rt_analysis.hfp_archive.restore import write_restored_day
from hsl_rt_analysis.hfp_archive.state import DayState, StateDB


logger = logging.getLogger(__name__)

EXIT_SENTINEL, EXIT_UNREACHABLE, EXIT_RUNNING = 3, 4, 7


def setup_logging(log_level: str, log_file: Path | None = None) -> None:
    """Configure logging with the specified level, to stderr and optionally to a file.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file that receives the same lines.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
    )


def _day(value: str) -> str:
    """Parse a day argument.

    Args:
        value: Command-line value.

    Returns:
        The day.

    Raises:
        argparse.ArgumentTypeError: If it is not a valid day.
    """
    try:
        return validate_day(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _common(parser: argparse.ArgumentParser) -> None:
    """Add the options every command shares.

    Args:
        parser: Sub-command parser.
    """
    home = Path.home()
    parser.add_argument("--remote", default=os.environ.get("HFP_ARCHIVE_REMOTE"), help="SSH target user@host.")
    parser.add_argument("--remote-port", type=int, default=23)
    parser.add_argument("--remote-root", default="2025_HSL_RT", help="Archive root on the storage box.")
    parser.add_argument("--state-dir", type=Path, default=home / ".local/state/hfp-archive")
    parser.add_argument("--work-dir", type=Path, default=home / "hfp-archive-work", help="Local scratch disk.")
    parser.add_argument("--memory-limit", default="8GB", help="DuckDB memory limit; it spills to disk above.")
    parser.add_argument(
        "--threads", type=int, default=0, help="DuckDB threads (0 = auto: one per GB of memory limit)."
    )
    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=Path, help="Also write the log to this file.")


def _range(parser: argparse.ArgumentParser) -> None:
    """Add day selection options.

    Args:
        parser: Sub-command parser.
    """
    parser.add_argument("--from", dest="first", type=_day, help="First day (inclusive).")
    parser.add_argument("--to", dest="last", type=_day, help="Last day (inclusive).")
    parser.add_argument("--days", nargs="+", type=_day, help="Explicit days.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments (default: sys.argv).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Advance days through the pipeline.")
    _common(run)
    _range(run)
    run.add_argument("--limit", type=int, help="At most this many days.")
    run.add_argument("--dry-run", action="store_true", help="Stop after VERIFIED; change nothing remotely.")
    run.add_argument("--cutoff-days", type=int, default=14, help="Never touch days newer than today minus this.")
    run.add_argument("--min-remote-free-gb", type=float, default=200.0, help="Stop when the box is this full.")
    run.add_argument(
        "--raw-archive", type=RawArchivePolicy, default=RawArchivePolicy.ALWAYS, choices=list(RawArchivePolicy)
    )

    for name, text in (("status", "Progress per day."), ("report", "Days that need a decision.")):
        p = sub.add_parser(name, help=text)
        _common(p)
        if name == "status":
            p.add_argument("--all", action="store_true", help="Also list finished days.")

    retry = sub.add_parser("retry", help="Resume FAILED / NEEDS_REVIEW days where they stopped.")
    _common(retry)
    retry.add_argument("days", nargs="+", type=_day)

    audit = sub.add_parser("audit", help="Re-check converted days on the storage box.")
    _common(audit)
    _range(audit)
    audit.add_argument("--deep", action="store_true", help="Also rebuild every raw file from the Parquet files.")
    audit.add_argument("--full", action="store_true", help="Also recompute the SHA-256 of the superseded files.")
    audit.add_argument("--workers", type=int, default=4, help="Days audited in parallel (SSH sessions).")

    purge = sub.add_parser("purge-superseded", help="Delete _superseded/ data of audited days.")
    _common(purge)
    purge.add_argument("--before", type=_day, required=True, help="Only days strictly before this day.")
    purge.add_argument("--audit-max-age-days", type=int, default=7)
    purge.add_argument("--execute", action="store_true", help="Really delete (default: dry run).")

    restore = sub.add_parser("restore", help="Rebuild a day's raw files from its Parquet files.")
    _common(restore)
    restore.add_argument("day", type=_day)
    restore.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> Settings:
    """Turn the parsed options into Settings.

    Args:
        args: Parsed arguments.

    Returns:
        Settings.

    Raises:
        SystemExit: If no remote is given.
    """
    if not args.remote:
        sys.exit("No storage box: pass --remote user@host or set HFP_ARCHIVE_REMOTE.")
    extra = {}
    for opt in ("cutoff_days", "min_remote_free_gb", "raw_archive"):
        if hasattr(args, opt):
            extra[opt] = getattr(args, opt)
    return Settings(
        remote=args.remote,
        remote_port=args.remote_port,
        remote_root=args.remote_root,
        state_dir=args.state_dir,
        work_dir=args.work_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
        **extra,
    )


def cmd_run(args: argparse.Namespace, settings: Settings, remote: Remote, db: StateDB) -> int:
    """Run the pipeline over the selected days.

    Args:
        args: Parsed arguments.
        settings: Settings.
        remote: The storage box.
        db: State database.

    Returns:
        Exit code.
    """
    with RunLock(settings.lock_file), StopFlag() as stop:
        remote.check_sentinel()
        cutoff = cutoff_day(settings)
        days = select_days(inventory(remote), db, cutoff, args.first, args.last, args.days)
        if args.limit:
            days = days[: args.limit]
        logger.info("%d days to process (cutoff %s)%s", len(days), cutoff, " [dry-run]" if args.dry_run else "")
        summary = Pipeline(settings, remote, db, stop).run(days, args.dry_run)
        logger.info("Run ended: %s, %s", summary.stopped_reason or "all selected days handled", dict(summary.finished))
        return summary.exit_code


def cmd_audit(args: argparse.Namespace, settings: Settings, remote: Remote, db: StateDB) -> int:
    """Audit the converted days in the range.

    Args:
        args: Parsed arguments.
        settings: Settings.
        remote: The storage box.
        db: State database.

    Returns:
        Exit code (1 if any day failed).
    """
    remote.check_sentinel()
    inv = inventory(remote)
    days = sorted(d for d in inv.hfp if (args.days is None or d in args.days))
    days = [d for d in days if not (args.first and d < args.first) and not (args.last and d > args.last)]
    failed = 0
    # Server-side checksums are the slow part and run per SSH session, so days are audited in parallel;
    # the results are recorded here in the main thread (the SQLite connection is not shared).
    with ThreadPoolExecutor(max_workers=1 if args.deep else args.workers) as pool:
        futures = {pool.submit(ops.audit_day, remote, settings, day, args.deep, args.full): day for day in days}
        for future in as_completed(futures):
            day, result = futures[future], future.result()
            db.record_audit(day, result["ok"], args.deep, result)
            failed += not result["ok"]
            logger.log(logging.INFO if result["ok"] else logging.ERROR, "audit %s: %s", day, result)
    logger.info("Audited %d days, %d failed", len(days), failed)
    return 1 if failed else 0


def cmd_purge(args: argparse.Namespace, settings: Settings, remote: Remote, db: StateDB) -> int:
    """List or delete the purgeable _superseded/ data.

    Args:
        args: Parsed arguments.
        settings: Settings.
        remote: The storage box.
        db: State database.

    Returns:
        Exit code.
    """
    with RunLock(settings.lock_file):
        remote.check_sentinel()
        inv = inventory(remote)
        ok, blocked = ops.purge_candidates(db, inv, args.before, timedelta(days=args.audit_max_age_days))
        for line in blocked:
            logger.warning("not purged: %s", line)
        freed = ops.purge(remote, inv, ok, args.execute)
        verb = "Freed" if args.execute else "[dry-run] would free"
        logger.info("%s %.1f GB from %d days (%d blocked)", verb, freed / 1e9, len(ok), len(blocked))
        for day in ok if args.execute else []:
            db.set_state(day, DayState.DONE, "superseded data purged", purged=True)
    return 0


def cmd_restore(args: argparse.Namespace, settings: Settings, remote: Remote) -> int:
    """Download a day's Parquet files and write its rebuilt raw files.

    Args:
        args: Parsed arguments.
        settings: Settings.
        remote: The storage box.

    Returns:
        Exit code.
    """
    local = args.out / f"_dt={args.day}"
    remote.download_dir(hfp_day(args.day), local)
    names = [f["name"] for f in json.loads((local / "_manifest.json").read_text())["source"]["files"]]
    repeats, rejects = local / "_repeats.parquet", local / "_rejects.json.gz"
    con = duckdb.connect()
    n = write_restored_day(
        con,
        sorted(local.glob("part-*.parquet")),
        repeats if repeats.exists() else None,
        rejects if rejects.exists() else None,
        names,
        args.out / args.day,
    )
    logger.info("Wrote %d files to %s", n, args.out / args.day)
    return 0


def dispatch(args: argparse.Namespace, settings: Settings, remote: Remote, db: StateDB) -> int:
    """Run one command.

    Args:
        args: Parsed arguments.
        settings: Settings.
        remote: The storage box.
        db: State database.

    Returns:
        Exit code.
    """
    match args.command:
        case "run":
            return cmd_run(args, settings, remote, db)
        case "status":
            pending = [d for d in select_days(inventory(remote), db, cutoff_day(settings)) if db.get(d) is None]
            print("\n".join(ops.status_lines(db, pending, args.all)))
            return 0
        case "report":
            print("\n".join(ops.report_lines(inventory(remote), db)))
            return 0
        case "retry":
            pipeline = Pipeline(settings, remote, db)
            for day in args.days:
                pipeline.retry(day)
            return 0
        case "audit":
            return cmd_audit(args, settings, remote, db)
        case "purge-superseded":
            return cmd_purge(args, settings, remote, db)
        case "restore":
            return cmd_restore(args, settings, remote)
    return 2


def main(argv: list[str] | None = None, remote: Remote | None = None) -> None:
    """Run the main logic.

    Args:
        argv: Arguments (default: sys.argv).
        remote: The storage box (default: SSH to --remote); tests pass a local stand-in.
    """
    args = parse_args(argv)
    setup_logging(args.log, args.log_file)
    settings = build_settings(args)
    if remote is None:
        remote = SshRemote(settings.remote, settings.remote_root, settings.remote_port, settings.max_wait_s)
    db = StateDB(settings.state_db)
    try:
        code = dispatch(args, settings, remote, db)
    except SentinelMissingError as exc:
        logger.error("Stopping: %s", exc)
        code = EXIT_SENTINEL
    except RemoteUnavailableError as exc:
        logger.error("Stopping: %s", exc)
        code = EXIT_UNREACHABLE
    except AlreadyRunningError as exc:
        logger.error("%s", exc)
        code = EXIT_RUNNING
    finally:
        db.close()
    sys.exit(code)


if __name__ == "__main__":
    main()

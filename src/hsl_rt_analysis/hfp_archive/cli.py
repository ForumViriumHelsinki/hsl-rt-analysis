"""hfp-archive: convert the HSL HFP archive on the storage box to the archive format v1, day by day.

Commands:
  run               advance days through fetch -> convert -> verify -> upload -> swap -> done
  status            progress per day
  report            days that need a decision (interim only, C7, failures, reappeared originals)
  retry             put FAILED / NEEDS_REVIEW days back where they stopped
  audit             re-check converted days on the storage box alone (needed before purging)
  purge-superseded  delete _superseded/ data of audited days before a date (dry run unless --execute)
  restore           rebuild a day's raw files from its Parquet files
  replicate         copy finished days to the Azure Blob container, verified file by file
  replica-status    which converted days are (not yet) in the Azure Blob container
  azure-check       write, read back and list a small probe blob to test the SAS token

The storage box is given with --remote or HFP_ARCHIVE_REMOTE (user@host, SSH port 23).
The Azure container is given with AZURE_BLOB_SAS_URL, its layout with AZURE_HFP_PREFIX
(default /hsl/hfp/) and AZURE_HFP_RAW_PREFIX (raw tars are copied only when it is set).
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
from dotenv import load_dotenv

from hsl_rt_analysis.hfp_archive import ops
from hsl_rt_analysis.hfp_archive.blobstore import (
    AzureBlobStore,
    BlobAuthError,
    BlobStore,
    BlobUnavailableError,
)
from hsl_rt_analysis.hfp_archive.config import RawArchivePolicy, Settings, hfp_day, validate_day
from hsl_rt_analysis.hfp_archive.discover import cutoff_day, inventory, select_days
from hsl_rt_analysis.hfp_archive.guards import AlreadyRunningError, RunLock, StopFlag
from hsl_rt_analysis.hfp_archive.pipeline import Pipeline
from hsl_rt_analysis.hfp_archive.remote import Remote, RemoteUnavailableError, SentinelMissingError, SshRemote
from hsl_rt_analysis.hfp_archive.replicate import BlobLayout, Replicator, digests, select_replica_days
from hsl_rt_analysis.hfp_archive.restore import write_restored_day
from hsl_rt_analysis.hfp_archive.state import DayState, StateDB, now_iso


logger = logging.getLogger(__name__)

EXIT_SENTINEL, EXIT_UNREACHABLE, EXIT_AZURE_AUTH, EXIT_RUNNING = 3, 4, 5, 7
AZURE_COMMANDS = {"replicate", "replica-status", "azure-check"}


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

    replicate = sub.add_parser("replicate", help="Copy finished days to the Azure Blob container.")
    _common(replicate)
    _range(replicate)
    replicate.add_argument("--limit", type=int, help="At most this many days.")
    replicate.add_argument("--verify", action="store_true", help="Download every blob back and compare SHA-256.")

    for name, text in (("replica-status", "Days not yet in Azure."), ("azure-check", "Test the SAS token.")):
        _common(sub.add_parser(name, help=text))
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


def cmd_replicate(args: argparse.Namespace, settings: Settings, remote: Remote, store: BlobStore, db: StateDB) -> int:
    """Copy the selected finished days to the blob container.

    It has its own lock, so it can run next to a conversion run (both only read the same days).

    Args:
        args: Parsed arguments.
        settings: Settings.
        remote: The storage box.
        store: The blob container.
        db: State database.

    Returns:
        Exit code.
    """
    layout = BlobLayout.from_env(os.environ)
    with RunLock(settings.state_dir / "replicate.lock"), StopFlag() as stop:
        remote.check_sentinel()
        replicator = Replicator(settings, remote, store, layout, db, stop)
        done = set(db.replicas(replicator.target))
        days = select_replica_days(inventory(remote).hfp, done, args.first, args.last, args.days)
        if args.limit:
            days = days[: args.limit]
        tar = f", tars to {layout.raw_prefix}" if layout.raw_prefix else ", no raw tars"
        logger.info("%d days to copy to %s%s", len(days), replicator.target, tar)
        summary = replicator.run(days, args.verify)
        logger.info(
            "Replication ended: %s; %d copied, %d failed, %d skipped, %.1f GB uploaded",
            summary.stopped_reason or "all selected days handled",
            len(summary.done),
            len(summary.failed),
            len(summary.skipped),
            summary.uploaded_bytes / 1e9,
        )
        return summary.exit_code


def cmd_replica_status(remote: Remote, store: BlobStore, db: StateDB) -> int:
    """Print which converted days are copied, failed or still missing in the container.

    Args:
        remote: The storage box.
        store: The blob container.
        db: State database.

    Returns:
        Exit code.
    """
    layout = BlobLayout.from_env(os.environ)
    records = db.replicas(f"{store.describe()}/{layout.data_prefix}")
    hfp = inventory(remote).hfp
    missing = sorted(hfp - set(records))
    failed = [r for r in records.values() if r.state == "failed"]
    for r in failed:
        print(f"{r.day}  failed     {r.last_error}")
    print(
        f"{store.describe()}/{layout.data_prefix}: {len(records) - len(failed)} copied, {len(failed)} failed, "
        f"{len(missing)} not yet ({missing[0] if missing else '-'} ... {missing[-1] if missing else '-'})"
    )
    return 0


def cmd_azure_check(store: BlobStore, settings: Settings) -> int:
    """Write a probe blob, read it back and list it: proves the create, write, read and list permissions.

    Args:
        store: The blob container.
        settings: Settings (for the local scratch directory).

    Returns:
        Exit code (1 if the probe did not read back).
    """
    layout = BlobLayout.from_env(os.environ)
    name = f"{layout.data_prefix}_hfp-archive-probe.txt"
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    local, back = settings.work_dir / "azure-probe.txt", settings.work_dir / "azure-probe-back.txt"
    local.write_text(f"hfp-archive probe {now_iso()}\n")
    sha, md5 = digests(local)
    store.upload(local, name, sha, md5, overwrite=True)
    store.download(name, back)
    ok = back.read_bytes() == local.read_bytes() and name in store.list_names(layout.data_prefix)
    local.unlink()
    back.unlink()
    logger.log(logging.INFO if ok else logging.ERROR, "azure-check %s: %s", "ok" if ok else "FAILED", store.describe())
    return 0 if ok else 1


def dispatch(
    args: argparse.Namespace, settings: Settings, remote: Remote, db: StateDB, store: BlobStore | None = None
) -> int:
    """Run one command.

    Args:
        args: Parsed arguments.
        settings: Settings.
        remote: The storage box.
        db: State database.
        store: The blob container (only for the Azure commands).

    Returns:
        Exit code.
    """
    if args.command in AZURE_COMMANDS:
        assert store is not None
        match args.command:
            case "replicate":
                return cmd_replicate(args, settings, remote, store, db)
            case "replica-status":
                return cmd_replica_status(remote, store, db)
            case _:
                return cmd_azure_check(store, settings)
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


def build_store(settings: Settings) -> BlobStore:
    """Connect to the container named by AZURE_BLOB_SAS_URL.

    Args:
        settings: Settings (for the connection wait time).

    Returns:
        The container.

    Raises:
        SystemExit: If AZURE_BLOB_SAS_URL is not set.
    """
    sas_url = os.environ.get("AZURE_BLOB_SAS_URL", "").strip()
    if not sas_url:
        sys.exit("No Azure container: set AZURE_BLOB_SAS_URL (container URL with a SAS token) in .env.")
    return AzureBlobStore(sas_url, settings.max_wait_s)


def main(argv: list[str] | None = None, remote: Remote | None = None, store: BlobStore | None = None) -> None:
    """Run the main logic.

    Args:
        argv: Arguments (default: sys.argv).
        remote: The storage box (default: SSH to --remote); tests pass a local stand-in.
        store: The blob container (default: AZURE_BLOB_SAS_URL); tests pass a local stand-in.
    """
    # .env in the working directory holds HFP_ARCHIVE_REMOTE and the Azure SAS URL; reading it here keeps
    # the token off command lines (tmux, systemd), where other users could see it in the process list.
    load_dotenv(Path.cwd() / ".env", override=False)
    args = parse_args(argv)
    setup_logging(args.log, args.log_file)
    settings = build_settings(args)
    if remote is None:
        remote = SshRemote(settings.remote, settings.remote_root, settings.remote_port, settings.max_wait_s)
    if store is None and args.command in AZURE_COMMANDS:
        store = build_store(settings)
    db = StateDB(settings.state_db)
    try:
        code = dispatch(args, settings, remote, db, store)
    except SentinelMissingError as exc:
        logger.error("Stopping: %s", exc)
        code = EXIT_SENTINEL
    except RemoteUnavailableError as exc:
        logger.error("Stopping: %s", exc)
        code = EXIT_UNREACHABLE
    except BlobUnavailableError as exc:
        logger.error("Stopping: %s", exc)
        code = EXIT_UNREACHABLE
    except BlobAuthError as exc:
        logger.error("Stopping: Azure refused the SAS token (expired or missing a permission?): %s", exc)
        code = EXIT_AZURE_AUTH
    except AlreadyRunningError as exc:
        logger.error("%s", exc)
        code = EXIT_RUNNING
    finally:
        db.close()
    sys.exit(code)


if __name__ == "__main__":
    main()

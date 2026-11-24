"""Prototype: convert one day of raw HFP files to the archive format v1 and run the checks.

This is a reference for the real converter, not the converter itself: it reads from and writes
to local directories only, never deletes source data and does not build the raw tar archive.
It parses one file hour at a time (stage_hour.sql), merges the hours (finalize_day.sql) and
checks the written file (verify_day.sql), so memory stays bounded on a busy day.

Usage:
    uv run --no-project --with duckdb==1.5.3 python docs/prototype/run_prototype.py \
        data/raw/all/2025-03-11 --out /tmp/hfp-archive
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
SCHEMA_VERSION = 1
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FILE_HOUR_RE = re.compile(r"-(\d{4}-\d{2}-\d{2}T\d{2})\.txt(\.gz)?$")
SORT_ORDER = "event_type, date_trunc('hour', tst), transport_mode, route, dir, oday, start, veh, tst"
PARQUET_OPTS = "FORMAT parquet, COMPRESSION zstd"
# The day is written as parts that together hold every row exactly once (C1 checks this). Sorting one part
# at a time keeps DuckDB's sort spill to about a quarter of the day instead of the whole day.
_BLOCK_HOUR = "hour(coalesce(tst, received_hour))"
PARTS = {
    "part-000.parquet": "event_type <> 'VP'",
    "part-001.parquet": f"event_type = 'VP' AND {_BLOCK_HOUR} < 6",
    "part-002.parquet": f"event_type = 'VP' AND {_BLOCK_HOUR} >= 6 AND {_BLOCK_HOUR} < 12",
    "part-003.parquet": f"event_type = 'VP' AND {_BLOCK_HOUR} >= 12 AND {_BLOCK_HOUR} < 18",
    "part-004.parquet": f"event_type = 'VP' AND {_BLOCK_HOUR} >= 18",
}


def setup_logging(log_level: str) -> None:
    """Configure logging with the specified level.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("day_dir", type=Path, help="Raw day directory, e.g. data/raw/all/2025-03-11.")
    parser.add_argument("--out", type=Path, required=True, help="Archive root to write hsl/hfp/dt=.../ under.")
    parser.add_argument("--memory-limit", default="8GB", help="DuckDB memory limit; it spills to disk above.")
    parser.add_argument("--max-temp", default="20GB", help="Largest DuckDB spill allowed in the work directory.")
    parser.add_argument("--keep-stage", action="store_true", help="Keep the staged hour files after success.")
    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def files_by_hour(day_dir: Path) -> dict[str, list[Path]]:
    """Group the raw files of one day directory by their file hour.

    Args:
        day_dir: Directory named YYYY-MM-DD holding <route>-YYYY-MM-DDTHH.txt[.gz] files.

    Returns:
        File hour (YYYY-MM-DDTHH) mapped to its sorted files.

    Raises:
        ValueError: If the directory name is not a date, a file name does not match, or there are no files.
    """
    if not DAY_RE.match(day_dir.name):
        msg = f"{day_dir} is not a YYYY-MM-DD directory"
        raise ValueError(msg)
    hours: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(day_dir.iterdir()):
        match = FILE_HOUR_RE.search(path.name)
        if match is None or not match.group(1).startswith(day_dir.name):
            msg = f"unexpected file in {day_dir}: {path.name}"
            raise ValueError(msg)
        hours[match.group(1)].append(path)
    if not hours:
        msg = f"no raw files in {day_dir}"
        raise ValueError(msg)
    return dict(sorted(hours.items()))


def stage_hour(con: duckdb.DuckDBPyConnection, hour: str, files: list[Path], stage_dir: Path) -> None:
    """Parse one file hour and write its staged rows, file stats and rejects to the stage directory.

    Args:
        con: DuckDB connection.
        hour: File hour, YYYY-MM-DDTHH.
        files: Raw files of that hour.
        stage_dir: Directory for the staged files.
    """
    con.execute("SET VARIABLE src = ?::VARCHAR[]", [[str(f) for f in files]])
    con.execute((HERE / "stage_hour.sql").read_text())
    con.execute(f"COPY staged TO '{stage_dir / f'rows-{hour}.parquet'}' ({PARQUET_OPTS})")
    con.execute(f"COPY file_stats TO '{stage_dir / f'stats-{hour}.parquet'}' ({PARQUET_OPTS})")
    if con.sql("SELECT count(*) FROM rejects").fetchone()[0]:
        con.execute(f"COPY rejects TO '{stage_dir / f'rejects-{hour}.parquet'}' ({PARQUET_OPTS})")
    # Free the hour's text before the next hour is read.
    con.execute(
        "DROP VIEW staged; DROP TABLE keep_rows; DROP TABLE rejects; DROP TABLE file_stats; DROP TABLE classified"
    )


def write_day(con: duckdb.DuckDBPyConnection, stage_dir: Path, part_dir: Path, day: str) -> list[Path]:
    """Merge the staged hours and write the day's Parquet parts (and rejects, if any) atomically.

    Args:
        con: DuckDB connection.
        stage_dir: Directory holding the staged hour files.
        part_dir: The dt=YYYY-MM-DD directory.
        day: The day, YYYY-MM-DD.

    Returns:
        Paths of the written Parquet parts.
    """
    con.execute(f"SET VARIABLE staged_glob = '{stage_dir / 'rows-*.parquet'}'")
    con.execute((HERE / "finalize_day.sql").read_text())
    part_dir.mkdir(parents=True, exist_ok=True)
    metadata = f"{{format: 'hsl-hfp-archive', schema_version: '{SCHEMA_VERSION}', dt: '{day}'}}"
    written = []
    for name, condition in PARTS.items():
        if not con.sql(f"SELECT count(*) FROM final WHERE {condition}").fetchone()[0]:
            continue  # An empty part is not written, e.g. the night hours of a partial day.
        final, tmp = part_dir / name, part_dir / f".{name}.tmp"
        con.execute(
            f"COPY (SELECT * FROM final WHERE {condition} ORDER BY {SORT_ORDER}) TO '{tmp}' "
            f"({PARQUET_OPTS}, ROW_GROUP_SIZE 1000000, KV_METADATA {metadata})"
        )
        tmp.replace(final)
        written.append(final)
        logger.info("Wrote %s (%.0f MB)", final, final.stat().st_size / 1e6)
    rejects = sorted(stage_dir.glob("rejects-*.parquet"))
    if rejects:
        tmp_rejects = part_dir / "._rejects.json.gz.tmp"
        con.execute(
            f"COPY (SELECT * FROM read_parquet({[str(p) for p in rejects]})) TO '{tmp_rejects}' "
            "(FORMAT json, COMPRESSION gzip)"
        )
        tmp_rejects.replace(part_dir / "_rejects.json.gz")
    return written


def run_checks(con: duckdb.DuckDBPyConnection, parquet: list[Path], stage_dir: Path) -> dict[str, dict[str, Any]]:
    """Run the statements of verify_day.sql and collect every row that carries an `ok` column.

    Args:
        con: DuckDB connection holding the finalize_day.sql views.
        parquet: The Parquet parts to read back.
        stage_dir: Directory holding the staged file stats.

    Returns:
        Check name mapped to the check's result columns.
    """
    con.execute("SET VARIABLE parquet = ?::VARCHAR[]", [[str(p) for p in parquet]])
    con.execute(f"SET VARIABLE stats_glob = '{stage_dir / 'stats-*.parquet'}'")
    results: dict[str, dict[str, Any]] = {}
    for statement in (HERE / "verify_day.sql").read_text().split(";\n"):
        code = "\n".join(line for line in statement.splitlines() if not line.lstrip().startswith("--")).strip()
        if not code:
            continue
        relation = con.sql(code)
        if relation is None or "ok" not in relation.columns:
            continue
        row = dict(zip(relation.columns, relation.fetchone(), strict=True))
        name = row.pop("check_name")
        results[name] = {k: str(v) if isinstance(v, int) and abs(v) > 2**53 else v for k, v in row.items()}
    return results


def counts(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Collect the line and row counts recorded in the manifest.

    Args:
        con: DuckDB connection holding the verify_day.sql views.

    Returns:
        Count fields for the manifest.
    """
    lines, rejected, kept = con.sql("SELECT sum(lines), sum(rejected), sum(kept) FROM stats").fetchone()
    across = con.sql("SELECT count(*) FROM later_copy").fetchone()[0]
    by_event = dict(con.sql("SELECT event_type, count(*) FROM pq GROUP BY ALL ORDER BY ALL").fetchall())
    return {
        "raw_lines": int(lines),
        "rejected": int(rejected),
        "repeats_removed": int(lines - rejected - kept + across),
        "rows": sum(by_event.values()),
        "rows_by_event_type": by_event,
    }


def write_manifest(part_dir: Path, day: str, files: list[Path], parquet: list[Path], info: dict[str, Any]) -> Path:
    """Write the day's _manifest.json atomically.

    Args:
        part_dir: The dt=YYYY-MM-DD directory.
        day: The day, YYYY-MM-DD.
        files: Raw source files of the day.
        parquet: The written Parquet parts.
        info: Counts and check results.

    Returns:
        Path of the manifest.
    """
    manifest = {
        "format": "hsl-hfp-archive",
        "schema_version": SCHEMA_VERSION,
        "dt": day,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "converter": {"name": "docs/prototype/run_prototype.py", "duckdb": duckdb.__version__},
        "source": {"files": [{"name": f.name, "size": f.stat().st_size, "sha256": sha256_of(f)} for f in files]},
        "parquet": [{"path": p.name, "size": p.stat().st_size, "sha256": sha256_of(p)} for p in parquet],
        **info,
    }
    path = part_dir / "_manifest.json"
    tmp = part_dir / "._manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, default=str))
    tmp.replace(path)
    return path


def main() -> None:
    """Run the main logic."""
    args = parse_args()
    setup_logging(args.log)
    hours = files_by_hour(args.day_dir)
    day = args.day_dir.name
    part_dir = args.out / "hsl" / "hfp" / f"dt={day}"
    stage_dir = args.out / "_work" / day
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory_limit}'; SET enable_progress_bar = false")
    con.execute(f"SET temp_directory = '{stage_dir / 'duckdb_tmp'}'; SET max_temp_directory_size = '{args.max_temp}'")
    started = time.monotonic()
    for hour, files in hours.items():
        stage_hour(con, hour, files, stage_dir)
        logger.info("Staged %s: %d files", hour, len(files))
    parquet = write_day(con, stage_dir, part_dir, day)
    total_mb = sum(p.stat().st_size for p in parquet) / 1e6
    logger.info("Wrote %d parts (%.0f MB) after %.0f s", len(parquet), total_mb, time.monotonic() - started)
    checks = run_checks(con, parquet, stage_dir)
    for name, result in checks.items():
        logger.log(logging.INFO if result["ok"] else logging.ERROR, "%s: %s", name, result)
    info = {"counts": counts(con), "checks": checks, "all_checks_ok": all(r["ok"] for r in checks.values())}
    all_files = [f for files in hours.values() for f in files]
    manifest = write_manifest(part_dir, day, all_files, parquet, info)
    con.close()
    if info["all_checks_ok"] and not args.keep_stage:
        shutil.rmtree(stage_dir)
    logger.info(
        "Done in %.0f s, all checks ok: %s, manifest %s", time.monotonic() - started, info["all_checks_ok"], manifest
    )


if __name__ == "__main__":
    main()

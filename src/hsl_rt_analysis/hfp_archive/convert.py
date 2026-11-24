"""Convert one local raw day directory to the Parquet parts and run checks C1-C4 (from the prototype).

The day is parsed one file hour at a time (stage_hour.sql), merged (finalize_day.sql) and written as
parts; the checks then read the written files back (verify_day.sql). Everything happens in local
directories; the pipeline moves the results to the storage box.
"""

import json
import logging
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import duckdb

from hsl_rt_analysis.hfp_archive.config import SCHEMA_VERSION
from hsl_rt_analysis.hfp_archive.rebuild import line_sql, payload_sql, shapes_query


logger = logging.getLogger(__name__)

SQL_DIR = files("hsl_rt_analysis.hfp_archive") / "sql"
FILE_HOUR_RE = re.compile(r"^(.+)-(\d{4}-\d{2}-\d{2}T\d{2})\.txt(\.gz)?$")
SORT_ORDER = "event_type, date_trunc('hour', tst), transport_mode, route, dir, oday, start, veh, tst"
PARQUET_OPTS = "FORMAT parquet, COMPRESSION zstd"
# The day is written as parts that together hold every row exactly once (C1 checks this). Sorting one part
# at a time keeps DuckDB's sort spill to about a quarter of the day instead of the whole day.
_BLOCK_HOUR = "hour(coalesce(tst, received_hour))"
VP_BLOCK_HOURS = 3  # 8 VP parts: the sort of one part stays small enough to keep the local spill low
PARTS = {"part-000.parquet": "event_type <> 'VP'"} | {
    f"part-{i + 1:03d}.parquet": (
        f"event_type = 'VP' AND {_BLOCK_HOUR} >= {i * VP_BLOCK_HOURS} AND {_BLOCK_HOUR} < {(i + 1) * VP_BLOCK_HOURS}"
    )
    for i in range(24 // VP_BLOCK_HOURS)
}
REJECTS_NAME = "_rejects.json.gz"
REPEATS_NAME = "_repeats.parquet"
FILE_STATS_NAME = "file_stats.parquet"
CONVERSION_NAME = "conversion.json"


class UnexpectedFileError(ValueError):
    """A day directory holds a file that is not one of the day's raw files."""


@dataclass(frozen=True)
class Conversion:
    """What convert_day produced."""

    parts: list[Path]
    rejects: Path | None
    repeats: Path | None
    counts: dict[str, Any]
    checks: dict[str, dict[str, Any]]
    seconds: float


def sql(name: str) -> str:
    """Return the text of one of the package's SQL files.

    Args:
        name: File name in the sql directory.

    Returns:
        SQL text.
    """
    return (SQL_DIR / name).read_text()


def files_by_hour(day_dir: Path, day: str) -> dict[str, list[Path]]:
    """Group the raw files of one day directory by their file hour.

    Args:
        day_dir: Directory holding <route>-YYYY-MM-DDTHH.txt[.gz] files.
        day: The day the directory stands for, YYYY-MM-DD.

    Returns:
        File hour (YYYY-MM-DDTHH) mapped to its sorted files.

    Raises:
        UnexpectedFileError: If a file name does not match the day, or there are no files.
    """
    hours: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(day_dir.iterdir()):
        match = FILE_HOUR_RE.match(path.name)
        if match is None or not match.group(2).startswith(day) or not path.is_file():
            msg = f"unexpected file in {day_dir}: {path.name}"
            raise UnexpectedFileError(msg)
        hours[match.group(2)].append(path)
    if not hours:
        msg = f"no raw files in {day_dir}"
        raise UnexpectedFileError(msg)
    return dict(sorted(hours.items()))


def connect(memory_limit: str, threads: int, temp_dir: Path, max_temp: str = "40GB") -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection that spills to a local directory.

    Args:
        memory_limit: DuckDB memory limit, e.g. 8GB.
        threads: Worker threads; 0 keeps DuckDB's default (all cores).
        temp_dir: Spill directory.
        max_temp: Largest spill allowed.

    Returns:
        The connection.
    """
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'; SET enable_progress_bar = false")
    con.execute(f"SET temp_directory = '{temp_dir}'; SET max_temp_directory_size = '{max_temp}'")
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    return con


def stage_hour(con: duckdb.DuckDBPyConnection, hour: str, hour_files: list[Path], stage_dir: Path) -> None:
    """Parse one file hour and write its staged rows, file stats, rejects and repeats to the stage directory.

    Args:
        con: DuckDB connection.
        hour: File hour, YYYY-MM-DDTHH.
        hour_files: Raw files of that hour.
        stage_dir: Directory for the staged files.
    """
    con.execute("SET VARIABLE src = ?::VARCHAR[]", [[str(f) for f in hour_files]])
    con.execute(sql("stage_hour.sql"))
    # Parse once into a scratch file, then add payload_raw with SQL generated for the shapes that occur:
    # the shapes are only known after parsing, and the scratch file keeps memory bounded.
    parsed = stage_dir / f".parsed-{hour}.parquet"
    con.execute(f"COPY parsed TO '{parsed}' ({PARQUET_OPTS})")
    src = f"read_parquet('{parsed}')"
    rebuilt = payload_sql(con.sql(shapes_query(src)).fetchall())
    con.execute(
        f"COPY (SELECT * EXCLUDE (payload_text, line_md5, topic_md5_hi, topic_md5_lo, source_file), "
        f"CASE WHEN {rebuilt} IS DISTINCT FROM payload_text THEN payload_text END AS payload_raw, "
        f"line_md5, topic_md5_hi, topic_md5_lo, source_file FROM {src}) "
        f"TO '{stage_dir / f'rows-{hour}.parquet'}' ({PARQUET_OPTS})"
    )
    parsed.unlink()
    con.execute(f"COPY file_stats TO '{stage_dir / f'stats-{hour}.parquet'}' ({PARQUET_OPTS})")
    # Always written, even when empty, so finalize_day.sql can read the list without special cases.
    con.execute(f"COPY hour_repeats TO '{stage_dir / f'repeats-{hour}.parquet'}' ({PARQUET_OPTS})")
    if con.sql("SELECT count(*) FROM rejects").fetchone()[0]:
        con.execute(f"COPY rejects TO '{stage_dir / f'rejects-{hour}.parquet'}' ({PARQUET_OPTS})")
    # Free the hour's text before the next hour is read.
    con.execute(
        "DROP VIEW parsed; DROP TABLE hour_repeats; DROP TABLE keep_rows; DROP TABLE rejects; "
        "DROP TABLE file_stats; DROP TABLE classified"
    )


def _copy_atomic(con: duckdb.DuckDBPyConnection, query: str, target: Path, options: str) -> Path:
    """COPY a query to a temporary name next to the target and rename it into place.

    Args:
        con: DuckDB connection.
        query: SELECT to write.
        target: Final path.
        options: COPY options.

    Returns:
        The target path.
    """
    tmp = target.with_name(f".{target.name}.tmp")
    con.execute(f"COPY ({query}) TO '{tmp}' ({options})")
    tmp.replace(target)
    return target


def write_day(con: duckdb.DuckDBPyConnection, stage_dir: Path, out_dir: Path, day: str) -> Conversion:
    """Merge the staged hours and write the day's parts, rejects and repeats.

    Args:
        con: DuckDB connection.
        stage_dir: Directory holding the staged hour files.
        out_dir: Output directory (the future dt=YYYY-MM-DD directory).
        day: The day, YYYY-MM-DD.

    Returns:
        A Conversion without counts and checks.
    """
    con.execute(f"SET VARIABLE staged_glob = '{stage_dir / 'rows-*.parquet'}'")
    con.execute(sql("finalize_day.sql"))
    # Cross-hour repeats carry no line text in the stage, so their text is rebuilt; C8 proves it.
    rebuilt = line_sql(con.sql(shapes_query("day_staged")).fetchall())
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE repeats AS "
        f"SELECT source_file, line_no, back, line FROM read_parquet('{stage_dir / 'repeats-*.parquet'}') "
        f"UNION ALL SELECT source_file, line_no, NULL::INTEGER AS back, {rebuilt} AS line "
        f"FROM day_staged SEMI JOIN later_copy USING (line_md5, received_hour)"
    )
    metadata = f"{{format: 'hsl-hfp-archive', schema_version: '{SCHEMA_VERSION}', dt: '{day}'}}"
    parts = []
    for name, condition in PARTS.items():
        if not con.sql(f"SELECT count(*) FROM final WHERE {condition}").fetchone()[0]:
            continue  # An empty part is not written, e.g. the night hours of a partial day.
        query = f"SELECT * FROM final WHERE {condition} ORDER BY {SORT_ORDER}"
        part = _copy_atomic(
            con, query, out_dir / name, f"{PARQUET_OPTS}, ROW_GROUP_SIZE 1000000, KV_METADATA {metadata}"
        )
        parts.append(part)
        logger.info("Wrote %s (%.0f MB)", part.name, part.stat().st_size / 1e6)
    rejects = repeats = None
    reject_files = sorted(stage_dir.glob("rejects-*.parquet"))
    if reject_files:
        query = f"SELECT * FROM read_parquet({[str(p) for p in reject_files]}) ORDER BY source_file, line_no"
        rejects = _copy_atomic(con, query, out_dir / REJECTS_NAME, "FORMAT json, COMPRESSION gzip")
    if con.sql("SELECT count(*) FROM repeats").fetchone()[0]:
        query = "SELECT * FROM repeats ORDER BY source_file, line_no"
        repeats = _copy_atomic(con, query, out_dir / REPEATS_NAME, PARQUET_OPTS)
    return Conversion(parts, rejects, repeats, {}, {}, 0.0)


def run_checks(con: duckdb.DuckDBPyConnection, parts: list[Path], stage_dir: Path) -> dict[str, dict[str, Any]]:
    """Run the statements of verify_day.sql (C1-C4) and collect every row that carries an `ok` column.

    Args:
        con: DuckDB connection holding the finalize_day.sql views.
        parts: The Parquet parts to read back.
        stage_dir: Directory holding the staged file stats.

    Returns:
        Check name mapped to the check's result columns.
    """
    con.execute("SET VARIABLE parquet = ?::VARCHAR[]", [[str(p) for p in parts]])
    con.execute(f"SET VARIABLE stats_glob = '{stage_dir / 'stats-*.parquet'}'")
    results: dict[str, dict[str, Any]] = {}
    for statement in sql("verify_day.sql").split(";\n"):
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


def day_counts(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Collect the line and row counts recorded in the manifest.

    Args:
        con: DuckDB connection holding the verify_day.sql views.

    Returns:
        Count fields for the manifest.
    """
    lines, rejected, kept = con.sql("SELECT sum(lines), sum(rejected), sum(kept) FROM stats").fetchone()
    across = con.sql("SELECT count(*) FROM later_copy").fetchone()[0]
    by_event = dict(con.sql("SELECT event_type, count(*) FROM pq GROUP BY ALL ORDER BY ALL").fetchall())
    raw_payloads = con.sql("SELECT count(*) FROM pq WHERE payload_raw IS NOT NULL").fetchone()[0]
    return {
        "raw_lines": int(lines),
        "rejected": int(rejected),
        "repeats_removed": int(lines - rejected - kept + across),
        "rows": sum(by_event.values()),
        "rows_by_event_type": by_event,
        "rows_with_payload_raw": int(raw_payloads),
    }


def convert_day(
    day: str, raw_dir: Path, out_dir: Path, stage_dir: Path, memory_limit: str = "8GB", threads: int = 0
) -> Conversion:
    """Convert one raw day directory and check the result (C1-C4). Starts from scratch every time.

    Args:
        day: The day, YYYY-MM-DD.
        raw_dir: Local copy of the raw day directory.
        out_dir: Output directory; emptied first.
        stage_dir: Scratch directory; emptied first and removed on success.
        memory_limit: DuckDB memory limit.
        threads: DuckDB threads (0 = all cores).

    Returns:
        The conversion with counts and check results.
    """
    hours = files_by_hour(raw_dir, day)
    for d in (out_dir, stage_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    started = time.monotonic()
    con = connect(memory_limit, threads, stage_dir / "duckdb_tmp")
    try:
        for hour, hour_files in hours.items():
            stage_hour(con, hour, hour_files, stage_dir)
            logger.debug("Staged %s: %d files", hour, len(hour_files))
        logger.info("Staged %d hours after %.0f s", len(hours), time.monotonic() - started)
        result = write_day(con, stage_dir, out_dir, day)
        checks = run_checks(con, result.parts, stage_dir)
        counts = day_counts(con)
        con.execute(
            f"COPY (SELECT * FROM stats ORDER BY source_file) TO '{out_dir.parent / FILE_STATS_NAME}' ({PARQUET_OPTS})"
        )
    finally:
        con.close()
    seconds = time.monotonic() - started
    conversion = Conversion(result.parts, result.rejects, result.repeats, counts, checks, seconds)
    (out_dir.parent / CONVERSION_NAME).write_text(
        json.dumps({"counts": counts, "checks": checks, "seconds": round(seconds, 1)}, indent=2, default=str)
    )
    shutil.rmtree(stage_dir)
    return conversion

"""Rebuild the raw files of a day from its Parquet parts, repeats and rejects, and check C8.

C8 compares, for every source file, the SHA-256 of the original file's decompressed content with the
SHA-256 of the content rebuilt from the archive. Both sides are computed independently: the original side
reads the gzip files with Python, the rebuilt side never sees them.
"""

import gzip
import hashlib
import logging
import os
import re
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from hsl_rt_analysis.hfp_archive.rebuild import FILE_STEM_SQL, line_sql, shapes_query


logger = logging.getLogger(__name__)

STEM_RE = re.compile(r"^(.+)\.txt(\.gz)?$")
BATCH_ROWS = 1_000_000


def file_stem(name: str) -> str:
    """Return a raw file name without its .txt / .txt.gz extension.

    Args:
        name: File name.

    Returns:
        The stem, e.g. 1001-2025-03-11T08.
    """
    match = STEM_RE.match(name)
    return match.group(1) if match else name


def content_digest(path: Path) -> tuple[str, str, int]:
    """Return the SHA-256 and line count of a raw file's decompressed content.

    Args:
        path: A .txt or .txt.gz file.

    Returns:
        (file name, hex digest, number of newline characters).
    """
    digest = hashlib.sha256()
    newlines = 0
    opener: Callable[..., Any] = gzip.open if path.name.endswith(".gz") else Path.open
    with opener(path, "rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
            newlines += chunk.count(b"\n")
    return path.name, digest.hexdigest(), newlines


def original_contents(paths: list[Path], workers: int | None = None) -> dict[str, dict[str, Any]]:
    """Hash the decompressed content of many raw files in parallel.

    Args:
        paths: Raw files.
        workers: Worker processes (default: all cores).

    Returns:
        File name mapped to {"content_sha256", "lines"}.
    """
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        results = pool.map(content_digest, paths, chunksize=32)
        return {name: {"content_sha256": sha, "lines": n} for name, sha, n in results}


class _FileHasher:
    """Feeds rows sorted by (stem, line_no) into one running SHA-256 per file, filling blank-line gaps."""

    def __init__(self) -> None:
        """Start with no open file."""
        self.done: dict[str, dict[str, Any]] = {}
        self.stem: str | None = None
        self.digest = hashlib.sha256()
        self.next_line = 0

    def _switch(self, stem: str) -> None:
        """Finish the current file and start another.

        Args:
            stem: The next file's stem.
        """
        self.finish()
        self.stem, self.digest, self.next_line = stem, hashlib.sha256(), 0

    def finish(self) -> None:
        """Record the current file's digest."""
        if self.stem is not None:
            self.done[self.stem] = {"content_sha256": self.digest.hexdigest(), "lines": self.next_line}
            self.stem = None

    def feed(self, stems: pa.Array, line_no: np.ndarray, lines: pa.Array) -> None:
        """Hash one batch. Each line already ends with a newline.

        Args:
            stems: File stem per row.
            line_no: 0-based line number per row.
            lines: Line text per row (string or large_string, no nulls).
        """
        if lines.type != pa.large_string():
            lines = lines.cast(pa.large_string())
        offsets = np.frombuffer(lines.buffers()[1], dtype=np.int64)[lines.offset : lines.offset + len(lines) + 1]
        data = memoryview(lines.buffers()[2])
        n = len(stems)
        # A segment ends where the file changes or a line number is skipped (a blank line in the original).
        cut = np.zeros(n, dtype=bool)
        if n > 1:
            cut[1:] = pc.not_equal(stems[1:], stems[:-1]).to_numpy(zero_copy_only=False)
            cut[1:] |= line_no[1:] != line_no[:-1] + 1
        cut[0] = True
        starts = np.flatnonzero(cut)
        ends = np.append(starts[1:], n)
        for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
            stem = stems[a].as_py()
            if stem != self.stem:
                self._switch(stem)
            gap = int(line_no[a]) - self.next_line
            if gap < 0:
                msg = f"line {line_no[a]} of {stem} appears twice"
                raise ValueError(msg)
            self.digest.update(b"\n" * gap)
            self.digest.update(data[offsets[a] : offsets[b]])
            self.next_line = int(line_no[b - 1]) + 1


def _batches(con: duckdb.DuckDBPyConnection, query: str) -> Iterator[pa.RecordBatch]:
    """Stream a query as Arrow record batches.

    Args:
        con: DuckDB connection.
        query: SELECT returning stem, line_no, line.

    Yields:
        Record batches.
    """
    reader = con.execute(query).to_arrow_reader(BATCH_ROWS)
    yield from reader


def _rows_query(hour_sql: str, rebuilt: str) -> str:
    """Return the query of one receive hour's rebuilt lines in file order.

    A repeat stored as a distance back resolves against the hour's rows and verbatim lines; the target is
    always the kept copy, which is one of those two.

    Args:
        hour_sql: SQL literal of the hour, e.g. TIMESTAMPTZ '2025-03-11 08:00:00+00'.
        rebuilt: Line rebuild expression for the archive rows.

    Returns:
        SQL query.
    """
    return f"""
        WITH base AS MATERIALIZED (
            SELECT {FILE_STEM_SQL} AS stem, line_no, {rebuilt} AS line FROM pq WHERE received_hour = {hour_sql}
            UNION ALL
            SELECT stem, line_no, line FROM extra WHERE hour = {hour_sql} AND back IS NULL
        ), refs AS (
            SELECT e.stem, e.line_no, b.line FROM extra e
            JOIN base b ON b.stem = e.stem AND b.line_no = e.line_no - e.back
            WHERE e.hour = {hour_sql} AND e.back IS NOT NULL
        )
        SELECT stem, line_no, line || chr(10) AS line FROM (
            SELECT * FROM base UNION ALL SELECT * FROM refs
        ) ORDER BY stem, line_no
    """


def restored_contents(
    con: duckdb.DuckDBPyConnection, parts: list[Path], repeats: Path | None, rejects: Path | None
) -> dict[str, dict[str, Any]]:
    """Rebuild every raw file of a day in memory-bounded hour slices and hash its content.

    Args:
        con: DuckDB connection (TimeZone is set to UTC here).
        parts: The day's Parquet parts.
        repeats: The day's _repeats.parquet, if any.
        rejects: The day's _rejects.json.gz, if any.

    Returns:
        File stem mapped to {"content_sha256", "lines"}.
    """
    con.execute("SET TimeZone = 'UTC'; SET preserve_insertion_order = false")
    con.execute(f"CREATE OR REPLACE TEMP VIEW pq AS SELECT * FROM read_parquet({[str(p) for p in parts]})")
    extra = [
        "SELECT NULL::VARCHAR AS source_file, NULL::INTEGER AS line_no, NULL::INTEGER AS back, "
        "NULL::VARCHAR AS line WHERE false"
    ]
    if repeats is not None:
        extra.append(f"SELECT source_file, line_no, back, line FROM read_parquet('{repeats}')")
    if rejects is not None:
        extra.append(
            f"SELECT source_file, line_no::INTEGER, NULL::INTEGER, line FROM read_json('{rejects}', "
            "format = 'newline_delimited', columns = {source_file: 'VARCHAR', line_no: 'INTEGER', line: 'VARCHAR'})"
        )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE extra AS SELECT regexp_extract(source_file, '^(.+)\\.txt(\\.gz)?$', 1) AS stem, "
        "strptime(regexp_extract(source_file, '-(\\d{4}-\\d{2}-\\d{2}T\\d{2})\\.txt(\\.gz)?$', 1), '%Y-%m-%dT%H')"
        f"::TIMESTAMPTZ AS hour, line_no, back, line FROM ({' UNION ALL '.join(extra)})"
    )
    rebuilt = line_sql(con.sql(shapes_query("pq")).fetchall())
    hours = con.sql(
        "SELECT DISTINCT received_hour::VARCHAR FROM pq UNION SELECT DISTINCT hour::VARCHAR FROM extra ORDER BY 1"
    ).fetchall()
    hasher = _FileHasher()
    for (hour,) in hours:
        hour_sql = f"TIMESTAMPTZ '{hour}'"
        for batch in _batches(con, _rows_query(hour_sql, rebuilt)):
            hasher.feed(batch.column(0), batch.column(1).to_numpy(zero_copy_only=False), batch.column(2))
        hasher.finish()
    return hasher.done


def check_c8(original: dict[str, dict[str, Any]], restored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare original and rebuilt file contents.

    Args:
        original: File name mapped to its content digest.
        restored: File stem mapped to its rebuilt content digest.

    Returns:
        The check result for the manifest.
    """
    by_stem = {file_stem(name): name for name in original}
    mismatched = sorted(
        name
        for stem, name in by_stem.items()
        if restored.get(stem, {}).get("content_sha256") != original[name]["content_sha256"]
    )
    unexpected = sorted(set(restored) - set(by_stem))
    return {
        "files": len(original),
        "mismatched_files": len(mismatched),
        "unexpected_files": len(unexpected),
        "examples": (mismatched + unexpected)[:10],
        "ok": not mismatched and not unexpected,
    }


def write_restored_day(
    con: duckdb.DuckDBPyConnection,
    parts: list[Path],
    repeats: Path | None,
    rejects: Path | None,
    names: list[str],
    out_dir: Path,
) -> int:
    """Write the rebuilt raw files of a day as gzip files (content-identical, not byte-identical gzip).

    Args:
        con: DuckDB connection.
        parts: The day's Parquet parts.
        repeats: The day's _repeats.parquet, if any.
        rejects: The day's _rejects.json.gz, if any.
        names: Original file names (from the manifest), which give the extensions.
        out_dir: Destination directory.

    Returns:
        Number of files written.
    """
    restored_contents(con, parts, repeats, rejects)  # sets up the views; the digests double as a self-check
    rebuilt = line_sql(con.sql(shapes_query("pq")).fetchall())
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = {file_stem(n): n for n in names}
    hours = con.sql(
        "SELECT DISTINCT received_hour::VARCHAR FROM pq UNION SELECT DISTINCT hour::VARCHAR FROM extra ORDER BY 1"
    )
    written = 0
    for (hour,) in hours.fetchall():
        current: str | None = None
        fh: Any = None
        next_line = 0
        for batch in _batches(con, _rows_query(f"TIMESTAMPTZ '{hour}'", rebuilt)):
            for stem, line_no, line in zip(*(c.to_pylist() for c in batch.columns), strict=True):
                if stem != current:
                    if fh is not None:
                        fh.close()
                    name = ext.get(stem, f"{stem}.txt.gz")
                    fh = gzip.open(out_dir / name, "wb") if name.endswith(".gz") else (out_dir / name).open("wb")  # noqa: SIM115
                    current, next_line, written = stem, 0, written + 1
                fh.write(b"\n" * (line_no - next_line) + line.encode())
                next_line = line_no + 1
        if fh is not None:
            fh.close()
    return written

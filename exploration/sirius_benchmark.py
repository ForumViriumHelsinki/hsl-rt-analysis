# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb>=2.0.1"]
# ///
"""Benchmark typical HFP queries on DuckDB's CPU engine and on the Sirius GPU extension.

Runs a fixed set of queries over v1 archive Parquet files (`dt=YYYY-MM-DD/part-*.parquet`), once read
straight from Parquet and once from a native DuckDB table, first on the CPU and then with Sirius loaded.
It reports the cold (first) and warm (median of the rest) times, whether EXPLAIN mentions the GPU, and
whether both engines return the same rows.

This is a standalone script with its own dependencies (DuckDB 2.0, which Sirius needs); the archive
converter stays pinned to its own DuckDB version. Run it with `just sirius-bench --data <dir>`.
"""

import argparse
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb


logger = logging.getLogger(__name__)

TABLE = "hfp"

# `{src}` is either read_parquet(...) or the native table. Every query orders its result fully, so the
# two engines' rows can be compared one to one. Results hold no TIMESTAMPTZ (fetching one needs pytz).
QUERIES: dict[str, str] = {
    "event_counts": """
        SELECT event_type, transport_mode, count(*) AS n FROM {src} GROUP BY ALL ORDER BY ALL""",
    "vp_filter_scan": """
        SELECT count(*) AS n, sum(spd) AS spd_sum, max(odo) AS odo_max
        FROM {src} WHERE event_type = 'VP' AND spd > 20""",
    "route_hour_vp": """
        SELECT topic_route_id, date_trunc('hour', tst)::TIMESTAMP AS hour, count(*) AS n, avg(spd) AS avg_spd,
               avg(dl) AS avg_dl
        FROM {src} WHERE event_type = 'VP' GROUP BY ALL ORDER BY topic_route_id, hour""",
    "vehicles_per_route": """
        SELECT oday, topic_route_id, count(DISTINCT (oper, veh)) AS vehicles
        FROM {src} WHERE event_type = 'VP' GROUP BY ALL ORDER BY ALL""",
    "stop_delay_top": """
        SELECT stop, count(*) AS n, avg(dl) AS avg_dl, min(dl) AS min_dl, max(dl) AS max_dl
        FROM {src} WHERE event_type IN ('ARR', 'DEP') AND stop IS NOT NULL
        GROUP BY stop ORDER BY n DESC, stop LIMIT 100""",
    "dwell_join": """
        WITH arr AS (
            SELECT oday, oper, veh, jrn, stop, min(tst) AS t FROM {src}
            WHERE event_type = 'ARR' AND stop IS NOT NULL GROUP BY ALL),
        dep AS (
            SELECT oday, oper, veh, jrn, stop, max(tst) AS t FROM {src}
            WHERE event_type = 'DEP' AND stop IS NOT NULL GROUP BY ALL)
        SELECT a.stop, count(*) AS n, avg(epoch(d.t) - epoch(a.t)) AS avg_dwell_s
        FROM arr a JOIN dep d USING (oday, oper, veh, jrn, stop) WHERE d.t >= a.t
        GROUP BY a.stop ORDER BY n DESC, a.stop LIMIT 100""",
    "vp_gap_window": """
        SELECT topic_route_id, max(gap_s) AS max_gap_s FROM (
            SELECT topic_route_id,
                   epoch(tst) - epoch(lag(tst) OVER (PARTITION BY oper, veh ORDER BY tst, tsi)) AS gap_s
            FROM {src} WHERE event_type = 'VP')
        GROUP BY ALL ORDER BY ALL""",
}


@dataclass
class Result:
    """Timings and checks of one query on one engine and one source."""

    query: str
    engine: str
    source: str
    runs_s: list[float] = field(default_factory=list)
    rows: int = 0
    plan_mentions_gpu: bool | None = None
    same_as_cpu: bool | None = None
    error: str | None = None

    @property
    def cold_s(self) -> float | None:
        """Return the first run's time, which includes reading the data.

        Returns:
            Seconds, or None if the query did not run.
        """
        return self.runs_s[0] if self.runs_s else None

    @property
    def warm_s(self) -> float | None:
        """Return the median of the runs after the first.

        Returns:
            Seconds, or None with fewer than two runs.
        """
        return statistics.median(self.runs_s[1:]) if len(self.runs_s) > 1 else None


@dataclass(frozen=True)
class Options:
    """What the benchmark runs on and with."""

    data_glob: str
    db_path: Path
    engines: list[str]
    sources: list[str]
    queries: list[str]
    repeat: int
    memory_limit: str | None
    threads: int | None
    sirius_settings: dict[str, str]


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


def connect(opts: Options, sirius: bool) -> duckdb.DuckDBPyConnection:
    """Open the benchmark database, optionally with Sirius loaded.

    Args:
        opts: Benchmark options.
        sirius: Whether to install and load the Sirius extension.

    Returns:
        The connection.
    """
    con = duckdb.connect(opts.db_path)
    con.execute("SET TimeZone = 'UTC'")
    if opts.memory_limit:
        con.execute(f"SET memory_limit = '{opts.memory_limit}'")
    if opts.threads:
        con.execute(f"SET threads = {opts.threads}")
    if sirius:
        con.execute("INSTALL sirius")
        con.execute("LOAD sirius")
        for key, value in opts.sirius_settings.items():
            con.execute(f"SET {key} = '{value}'")
    return con


def ensure_table(opts: Options, rebuild: bool) -> None:
    """Load the Parquet files into a native table and checkpoint it, unless it is already there.

    Sirius reads native tables from the database file, so the data is checkpointed before any query.

    Args:
        opts: Benchmark options.
        rebuild: Recreate the table even if it exists.
    """
    con = connect(opts, sirius=False)
    try:
        exists = con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [TABLE]).fetchone() == (1,)
        if exists and not rebuild:
            logger.info("Using the existing table %s in %s", TABLE, opts.db_path)
            return
        start = time.perf_counter()
        con.execute(f"CREATE OR REPLACE TABLE {TABLE} AS SELECT * FROM {parquet_source(opts.data_glob)}")
        con.execute("CHECKPOINT")
        n = con.execute(f"SELECT count(*) FROM {TABLE}").fetchone()
        logger.info("Loaded %s rows into %s in %.1f s", n[0] if n else "?", TABLE, time.perf_counter() - start)
    finally:
        con.close()


def parquet_source(data_glob: str) -> str:
    """Return the SQL that reads the archive Parquet files.

    Args:
        data_glob: Glob of the Parquet files.

    Returns:
        A read_parquet(...) expression.
    """
    quoted = data_glob.replace("'", "''")
    return f"read_parquet('{quoted}', hive_partitioning = false)"


def plan_mentions_gpu(con: duckdb.DuckDBPyConnection, sql: str) -> bool:
    """Tell whether the physical plan names the GPU, as a hint that Sirius took the query.

    Args:
        con: Connection.
        sql: Query.

    Returns:
        True if the EXPLAIN text mentions Sirius or the GPU.
    """
    plan = "\n".join(str(row[-1]) for row in con.execute(f"EXPLAIN {sql}").fetchall())
    logger.debug("Plan:\n%s", plan)
    return any(word in plan.lower() for word in ("sirius", "gpu", "cuda"))


def run_query(con: duckdb.DuckDBPyConnection, result: Result, sql: str, repeat: int) -> list[tuple[Any, ...]]:
    """Run a query a number of times, recording each run's time including the fetch.

    Args:
        con: Connection.
        result: Where the times go.
        sql: Query.
        repeat: Number of runs.

    Returns:
        The rows of the last run.
    """
    rows: list[tuple[Any, ...]] = []
    for _ in range(repeat):
        start = time.perf_counter()
        rows = con.execute(sql).fetchall()
        result.runs_s.append(time.perf_counter() - start)
    result.rows = len(rows)
    return rows


def same_rows(a: list[tuple[Any, ...]], b: list[tuple[Any, ...]], rel_tol: float = 1e-9) -> bool:
    """Compare two ordered results, allowing floating-point sums to differ in the last digits.

    Args:
        a: First result.
        b: Second result.
        rel_tol: Relative tolerance for floats.

    Returns:
        True if the results match.
    """
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b, strict=True):
        if len(row_a) != len(row_b):
            return False
        for x, y in zip(row_a, row_b, strict=True):
            if isinstance(x, float) and isinstance(y, float):
                if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=1e-9):
                    return False
            elif x != y:
                return False
    return True


def run_engine(opts: Options, engine: str, reference: dict[tuple[str, str], list[tuple[Any, ...]]]) -> list[Result]:
    """Run every query on every source with one engine.

    The CPU engine runs first and fills `reference`; the GPU engine's rows are compared with it.

    Args:
        opts: Benchmark options.
        engine: "cpu" or "gpu".
        reference: CPU rows per (query, source), filled by the CPU run.

    Returns:
        The results.
    """
    con = connect(opts, sirius=engine == "gpu")
    results = []
    try:
        for source in opts.sources:
            src = parquet_source(opts.data_glob) if source == "parquet" else TABLE
            for name in opts.queries:
                results.append(run_one(con, engine, source, name, QUERIES[name].format(src=src), opts, reference))
    finally:
        con.close()
    return results


def run_one(
    con: duckdb.DuckDBPyConnection,
    engine: str,
    source: str,
    name: str,
    sql: str,
    opts: Options,
    reference: dict[tuple[str, str], list[tuple[Any, ...]]],
) -> Result:
    """Run one query, catching its error so that the other queries still run.

    Args:
        con: Connection.
        engine: "cpu" or "gpu".
        source: "parquet" or "table".
        name: Query name.
        sql: Query text.
        opts: Benchmark options.
        reference: CPU rows per (query, source).

    Returns:
        The result.
    """
    result = Result(name, engine, source)
    try:
        result.plan_mentions_gpu = plan_mentions_gpu(con, sql)
        rows = run_query(con, result, sql, opts.repeat)
    except duckdb.Error as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.warning("%s / %s / %s failed: %s", engine, source, name, result.error)
        return result
    if engine == "cpu":
        reference[name, source] = rows
    elif (name, source) in reference:
        result.same_as_cpu = same_rows(reference[name, source], rows)
    logger.info("%s / %s / %s: cold %.3f s, %d rows", engine, source, name, result.cold_s, result.rows)
    return result


def fmt_s(value: float | None) -> str:
    """Format seconds for the report.

    Args:
        value: Seconds or None.

    Returns:
        The text.
    """
    return "-" if value is None else f"{value:8.3f}"


def report(results: list[Result]) -> str:
    """Return a plain-text table with CPU and GPU side by side.

    Args:
        results: All results.

    Returns:
        The table.
    """
    by_key = {(r.query, r.source, r.engine): r for r in results}
    lines = [f"{'source':8} {'query':20} {'cpu cold':>9} {'cpu warm':>9} {'gpu cold':>9} {'gpu warm':>9} "
             f"{'speedup':>8} {'gpu plan':>8} {'same':>5}"]  # fmt: skip
    for source, query in dict.fromkeys((r.source, r.query) for r in results):
        cpu, gpu = by_key.get((query, source, "cpu")), by_key.get((query, source, "gpu"))
        speedup = "-"
        if cpu and gpu and cpu.warm_s and gpu.warm_s:
            speedup = f"{cpu.warm_s / gpu.warm_s:7.2f}x"
        plan = "-" if not gpu or gpu.plan_mentions_gpu is None else ("yes" if gpu.plan_mentions_gpu else "no")
        same = "-" if not gpu or gpu.same_as_cpu is None else ("yes" if gpu.same_as_cpu else "NO")
        if gpu and gpu.error:
            plan, same = "error", "-"
        times = " ".join(f"{fmt_s(t):>9}" for r in (cpu, gpu) for t in ((r.cold_s, r.warm_s) if r else (None, None)))
        lines.append(f"{source:8} {query:20} {times} {speedup:>8} {plan:>8} {same:>5}")
    return "\n".join(lines)


def benchmark(opts: Options, rebuild: bool) -> list[Result]:
    """Run the whole benchmark.

    Args:
        opts: Benchmark options.
        rebuild: Recreate the native table.

    Returns:
        All results.
    """
    if "table" in opts.sources:
        ensure_table(opts, rebuild)
    reference: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
    results: list[Result] = []
    # CPU first: once Sirius is loaded into this process it may stay loaded, so it must not touch the CPU run.
    if "cpu" in opts.engines:
        results += run_engine(opts, "cpu", reference)
    if "gpu" in opts.engines:
        try:
            results += run_engine(opts, "gpu", reference)
        except duckdb.Error as exc:
            # Typically INSTALL/LOAD sirius failing (no GPU, driver, platform); the CPU results still count.
            logger.error("GPU run stopped: %s", exc)
    return results


def parse_settings(pairs: list[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE options.

    Args:
        pairs: The option values.

    Returns:
        Key mapped to value.

    Raises:
        argparse.ArgumentTypeError: If a value has no `=`.
    """
    settings = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.isidentifier():
            raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {pair!r}")
        settings[key] = value
    return settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="directory holding dt=*/part-*.parquet")
    parser.add_argument("--db", type=Path, default=Path("sirius-bench.duckdb"), help="benchmark database file")
    parser.add_argument("--engines", default="cpu,gpu", help="comma-separated: cpu, gpu")
    parser.add_argument("--sources", default="parquet,table", help="comma-separated: parquet, table")
    parser.add_argument("--queries", default=",".join(QUERIES), help="comma-separated query names")
    parser.add_argument("--repeat", type=int, default=4, help="runs per query; the first one counts as cold")
    parser.add_argument("--memory-limit", help="DuckDB memory_limit, e.g. 24GB")
    parser.add_argument("--threads", type=int, help="DuckDB threads")
    parser.add_argument("--sirius-setting", action="append", default=[], metavar="KEY=VALUE",
                        help="extra SET for the GPU connection, repeatable")  # fmt: skip
    parser.add_argument("--rebuild", action="store_true", help="reload the native table from the Parquet files")
    parser.add_argument("--output", type=Path, help="write the results as JSON here")
    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def build_options(args: argparse.Namespace) -> Options:
    """Turn the arguments into benchmark options, checking the names.

    Args:
        args: Parsed arguments.

    Returns:
        The options.

    Raises:
        SystemExit: If a name is unknown or no Parquet file matches.
    """
    names = ("engines", "sources", "queries")
    split = {k: [v.strip() for v in getattr(args, k).split(",") if v.strip()] for k in names}
    allowed = {"engines": {"cpu", "gpu"}, "sources": {"parquet", "table"}, "queries": set(QUERIES)}
    for key, values in split.items():
        if bad := set(values) - allowed[key]:
            raise SystemExit(f"unknown {key}: {', '.join(sorted(bad))}")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if not any(args.data.glob("**/part-*.parquet")):
        raise SystemExit(f"no part-*.parquet files under {args.data}")
    return Options(
        data_glob=str(args.data / "**" / "part-*.parquet"),
        db_path=args.db,
        engines=split["engines"],
        sources=split["sources"],
        queries=split["queries"],
        repeat=args.repeat,
        memory_limit=args.memory_limit,
        threads=args.threads,
        sirius_settings=parse_settings(args.sirius_setting),
    )


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark and report the results.

    Args:
        argv: Arguments, or None for sys.argv.
    """
    args = parse_args(argv)
    setup_logging(args.log)
    opts = build_options(args)
    logger.info("DuckDB %s, data %s", duckdb.__version__, opts.data_glob)
    try:
        results = benchmark(opts, args.rebuild)
    except duckdb.Error as exc:
        logger.error("Benchmark stopped: %s", exc)
        sys.exit(2)
    logger.info("Results (seconds; warm = median of runs 2..n; speedup = cpu warm / gpu warm):\n%s", report(results))
    if args.output:
        payload = {"duckdb": duckdb.__version__, "data": opts.data_glob, "results": [asdict(r) for r in results]}
        args.output.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()

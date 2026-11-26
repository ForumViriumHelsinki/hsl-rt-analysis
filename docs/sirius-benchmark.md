# Sirius GPU benchmark

[Sirius](https://github.com/sirius-db/sirius) is a DuckDB core extension that runs supported analytical
queries on an NVIDIA GPU and falls back to the CPU engine for the rest. `exploration/sirius_benchmark.py`
measures whether it helps with typical HFP queries over the v1 archive (`hsl/hfp/dt=*/part-*.parquet`).

It is for analysis queries only. The archive converter spends its time on rsync, gzip and JSON parsing,
which the GPU does not touch, and it stays pinned to its own DuckDB version (recorded in every manifest).

## What it does

- Seven queries: event counts, a filtered scan, hourly VP aggregates per route, distinct vehicles per
  route, stop delays, an ARR/DEP join for dwell times, and a window function (`lag`) for VP gaps.
- Each query runs from Parquet (`read_parquet`) and from a native table in a DuckDB file. Sirius reads
  native tables, so the script loads and checkpoints the table once and reuses it after that.
- The CPU engine runs first, then a fresh connection with `INSTALL sirius; LOAD sirius`.
- Each query runs `--repeat` times (default 4): the first run is reported as cold, the median of the rest
  as warm, and the speedup is CPU warm / GPU warm.
- `gpu plan` says whether `EXPLAIN` mentions Sirius, the GPU or CUDA. It is only a hint that the query did
  not fall back to the CPU. Read the plans with `--log DEBUG` if the hint and the timings disagree.
- `same` compares the GPU rows with the CPU rows, with a small tolerance for floating-point sums.
  `NO` means the GPU result is wrong, or the query is not as deterministic as intended.

The script has its own dependencies (`duckdb>=2.0.1`, inline in the file), so `uv run --script` builds a
separate environment and the project's pinned DuckDB is not affected.

## Requirements

- Linux on x86-64 or arm64. DuckDB publishes no Sirius build for macOS: on a Mac the CPU part runs and the
  GPU part stops with HTTP 404.
- An NVIDIA GPU with a recent driver (`nvidia-smi` must work). The extension picks its CUDA backend from
  the driver and is about 700 MB to download.
- The GPU has to hold the working set: an RTX 4090 has 24 GB. One full day is about 0.5 GB of Parquet and
  about 40 million rows, so start with one day and then try a week.

## Running it

Copy a few days from the storage box to a local directory (the archive's own day directories):

```bash
mkdir -p ~/sirius-bench/data
```

```bash
rsync -a -e 'ssh -p 23' "$HFP_ARCHIVE_REMOTE:2025_HSL_RT/hsl/hfp/dt=2026-10-15" ~/sirius-bench/data/
```

Run the benchmark (the first run loads the table, later runs reuse it; `--rebuild` reloads it after you
add days):

```bash
just sirius-bench --data ~/sirius-bench/data --db ~/sirius-bench/bench.duckdb --output ~/sirius-bench/1day.json
```

Useful options:

| Option | Use |
| --- | --- |
| `--engines cpu` / `--engines gpu` | run only one engine |
| `--sources table` | only the native table (the case Sirius is documented for) |
| `--queries dwell_join,vp_gap_window` | a subset of the queries |
| `--memory-limit 24GB --threads 16` | DuckDB CPU settings, the same for both engines |
| `--sirius-setting KEY=VALUE` | an extra `SET` for the GPU connection, repeatable |
| `--log DEBUG` | log every query plan |

The results table goes to the log, and `--output` writes every run's time as JSON.

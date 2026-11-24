# HFP archive converter (`hfp-archive`)

`hsl_rt_analysis.hfp_archive` converts the HSL HFP data on the Hetzner storage box from the current
layout (`raw/all/`, `interim/`) to the archive format v1 defined in
[`hfp-archive-format.md`](hfp-archive-format.md), one day at a time, without losing data in any
situation. A run can last weeks; it survives interruptions, sleep and lost connections.

## Setup

```bash
just setup
echo 'HFP_ARCHIVE_REMOTE=<user>@<user>.your-storagebox.de' >> .env   # not committed
```

| Environment variable | Meaning |
| --- | --- |
| `HFP_ARCHIVE_REMOTE` | SSH target of the storage box (key-based login on port 23 must work) |

Everything goes over SSH and rsync, so the WebDAV mount is not needed. The archive root
`2025_HSL_RT/` must hold the sentinel file `.hfp-archive-sentinel`, created once with:

```bash
ssh -p 23 <user>@<user>.your-storagebox.de touch 2025_HSL_RT/.hfp-archive-sentinel
```

Local state lives in `~/.local/state/hfp-archive/` (SQLite database `state.sqlite`, lock file
`run.lock`), scratch data in `~/hfp-archive-work/` (one directory per day, removed when the day is done).

## The state machine

| State | Reached when |
| --- | --- |
| `pending` | the day is known |
| `fetched` | the raw day is copied here and every file's SHA-256 equals the one computed on the storage box |
| `converted` | the Parquet parts, repeats and rejects are written and C1-C4 pass |
| `verified` | C8 (raw rebuild), C5 (tar, if kept) and C7 (interim) pass |
| `uploaded` | the tar and the `dt=` directory are on the storage box: uploaded to temporary names, SHA-256 read back on the server (C6), renamed; the manifest written last |
| `swapped` | `raw/all/<day>` and `interim/<day>` moved to `_superseded/`, after checking on the server that they still hold exactly what was converted |
| `done` | the manifest says `done` and the local scratch is removed |
| `needs_review` | a decision is needed: C7 failed, a source changed after the fetch, a conflict on the storage box |
| `failed` | a check or a step failed; `retry` resumes the day where it stopped |

The state is in SQLite and, from `uploaded` on, also in the day's `_manifest.json`
(`archive_state`). Every step first looks at what already exists, so a run that stops anywhere picks
up from the last recorded state. A `dt=` directory appears in one atomic rename, so readers never see a
half-uploaded day.

Before every step the run checks the stop flag, the sentinel on the storage box and the free local
disk; before uploading also the free space on the storage box (`--min-remote-free-gb`, default 200).
The first SIGINT / SIGTERM lets the current step finish and then stops; a second one aborts at once
(still safe: the step is redone). A lost connection is waited out for up to 6 hours; a missing
sentinel stops the run at once. Rsync and SSH run in their own process group, so a Ctrl-C does not
kill a transfer in the middle.

Days newer than today minus 14 days are never touched: dckr3's cron syncs only yesterday's
directories and keeps its own copies for 8 days. Days that have interim files but no raw files are
only reported.

## Commands

```bash
just archive run --days 2026-10-15 --dry-run   # fetch, convert, verify locally; nothing changes remotely
just archive run --days 2026-10-15             # one day for real (continues a dry-run day from verified)
just archive run --from 2025-03-11 --limit 3   # the next three days
just archive-bg                                # bulk run in tmux + caffeinate, log in ~/hfp-archive-work/logs/run.log
just archive-log                               # follow the log
just archive-status                            # progress per day (--all also lists finished days)
just archive report                            # interim-only days, needs_review, failed, reappeared originals
just archive retry 2026-01-05                  # resume a failed / needs_review day
just archive-stop                              # stop the bulk run after its current step
```

`archive-bg` keeps running when the terminal closes. `caffeinate -is` keeps the Mac from idle sleep
and, on AC power, from system sleep; a closed lid still sleeps the Mac unless it runs clamshell with
power and a display. After waking, the run waits for the network and carries on.

## Before purging `_superseded/`

The storage box has 1.2 TB free and is shared with other projects. The new files take roughly
1.4 TB with the tar (0.6 TB without), while the originals stay in `_superseded/`, so the box fills
up about halfway through the bulk run. The run then stops by itself (`--min-remote-free-gb`), and
the superseded data of finished days has to be purged.

1. `just archive audit --from <first> --to <last>` checks every converted day on the storage box
   alone, without trusting the local database:
   - the manifest says every required check passed and the schema version is 1;
   - the SHA-256 of every Parquet file, the repeats, the rejects and the tar, computed on the server
     now, equals the manifest (no damage since the upload);
   - `_superseded/` holds exactly the files the manifest lists as converted, by name and SHA-256;
   - the original paths `raw/all/<day>` and `interim/<day>` are gone (nothing reappeared).
2. `--deep` additionally downloads the Parquet files and rebuilds every raw file, comparing it with
   the manifest's `content_sha256` (C8 again, independent of the conversion run). Run it on a sample,
   e.g. one day per week, or on all days if time allows (about the download time of the Parquet files).
3. `just archive status` and `just archive report` must show no `failed` / `needs_review` day in the
   range that you have not decided about.
4. `just archive purge-superseded --before <day>` lists what would be deleted, per day and in GB.
   Only `done` days with a passed audit from the last 7 days are included; others are listed as
   blocked with the reason.
5. `just archive purge-superseded --before <day> --execute` deletes them. Nothing outside
   `_superseded/` can be deleted by the tool.

## Restoring raw files

```bash
just archive restore 2026-10-15 --out /tmp/restored
```

downloads the day's `dt=` directory and writes every raw file with its original content (the gzip
bytes differ from the original gzip, the decompressed content is identical). For a day with a tar, the
tar holds the byte-exact original gzip files.

## Measurements

2026-10-15, a full October 2026 day, on a MacBook with 14 cores and 48 GB RAM (DuckDB memory limit
8 GB, all cores):

| Measure | Value |
| --- | --- |
| Raw | 5,087 files, 1.59 GB gzip, 85.0 M lines (24 % exact repeats, 0 rejects) |
| Parquet | 64.5 M rows, 9 parts, 1.05 GB, plus `_repeats.parquet` 58 MB; no row needed `payload_raw` |
| Raw tar | 1.60 GB |
| Fetch | about 3 min download (~10 MB/s) + 2 min server-side SHA-256 of raw and interim |
| Convert (stage, parts, C1-C4) | 407 s |
| Verify (C8, tar + C5, C7) | 321 s |
| Upload + swap + done | 1.5 min |
| Peak local working disk | 7.4 GB (raw copy, stage, DuckDB spill, parts) |
| Peak process memory | 20 GB (DuckDB exceeds its limit during the hourly staging and the C8 rebuild) |

A first measurement with 6-hour VP parts and verbatim repeats peaked at 17.5 GB of working disk and
took 30 minutes; 3-hour parts and repeats stored as distances brought it to the figures above.
Expect roughly 15-20 minutes per full day, so the 600 raw days take about a week of running time.

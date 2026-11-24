# HSL HFP archive format, version 1

This document fixes the long-term archive format for HSL High-Frequency Positioning (HFP) data: what
is stored, where, in which format, and which checks must pass before a source file may be replaced. The
same layout is used on the Hetzner storage box and later in Azure (ADLS Gen2), so data can be copied
between them unchanged.

The converter that applies this format to the storage box is `hsl_rt_analysis.hfp_archive`, see
[`hfp-archive-converter.md`](hfp-archive-converter.md). The single-day prototype it grew from is in
[`prototype/`](prototype/), see [Prototype](#prototype).

## Principles

1. **Nothing is lost.** Every raw line ends up either as a Parquet row, verbatim in the day's repeats
   file, or verbatim in the day's rejects file, each with its source file and line number. Together they
   rebuild the content of every original file byte for byte (check C8).
2. **The raw tar is a policy, not a necessity.** The per-day tar keeps the original gzip files byte for
   byte. It is written for every day (`--raw-archive always`) until the decision to drop it is made;
   with `on-c8-failure` it is written only for a day whose files do not all rebuild exactly. If the
   Parquet conversion is ever found to be wrong, a day is rebuilt from its raw files, or from the tar.
3. **Days are independent.** A day of Parquet is built only from the raw files of the same day
   directory, so each day can be converted, checked and replaced on its own.
4. **Fail loudly.** A value that does not fit its column type, an unexpected file name or a failed
   check stops the day. Nothing is skipped silently.

## Source data (current layout)

| Data | Path | Content |
| --- | --- | --- |
| Raw | `raw/all/YYYY-MM-DD/<route>-YYYY-MM-DDTHH.txt[.gz]` | one line per MQTT message: `<topic> <json>` |
| Interim | `interim/YYYY-MM-DD/<route>_YYYYMMDD.geoparquet` | VP messages only, a subset of the fields |
| Interim | `interim/YYYY-MM-DD/<route>_YYYYMMDD_other_messages.txt.gz` | other events, raw lines |

- The collector (`docker/mqtt_dumper/hfp_mqtt_data_dumper.py`) subscribes to
  `/hfp/v2/journey/ongoing/#`, so every topic starts with `/hfp/v2/journey/ongoing/`.
- A line goes into the file of its topic's route (`topic` level 9) and the UTC hour **when it was
  received**. The file name hour is therefore the receive hour, not the vehicle timestamp `tst`.
- Raw files are compressed to `.txt.gz` about two hours after they were last written.
- Interim files are derived from the raw files. They lack the topic fields (transport mode, headsign,
  next stop, geohash), `tsi`, `label` and `seq`, and store the position only as a geometry. The archive
  is therefore built from the raw files, never from interim files.

Observed in a full-network sample (2025-03-11, 37.9 M lines): 16 event types, 92 % of them `VP`. All
event types share the same core of about 22 payload fields. Only a few types add fields:

| Event types | Extra payload fields |
| --- | --- |
| DUE, ARR, DEP, ARS, PDE, PAS, WAIT, DOO, DOC | `ttarr`, `ttdep` |
| TLR | `sid`, `signal-groupid` and ten `tlp-*` fields |
| TLA | `tlp-requestid`, `tlp-decision` |
| VJA, VJOUT | `block`, `dr-type` |
| DA | `dr-type` |
| VP | `label`, `seq` (some messages) |

DOUT (60 messages that day) carried no keys outside the mapped set either.

## Archive layout

All paths are relative to an archive root: a directory on the storage box, or a container in Azure.
The INRIX archive uses the same root (`inrix/segment-speed/YYYY/MM/DD/...`).

```
<root>/
├── hsl/hfp/dt=YYYY-MM-DD/
│   ├── part-000.parquet        # non-VP events
│   ├── part-001.parquet        # VP, hours 00-02 UTC
│   ├── ...                     # VP, one part per 3 hours
│   ├── part-008.parquet        # VP, hours 21-23 UTC
│   ├── _manifest.json          # counts, checksums, check results, archive state
│   ├── _repeats.parquet        # only when some lines were exact repeats
│   └── _rejects.json.gz        # only when some lines did not parse
├── hsl/hfp-raw/YYYY/MM/DD/
│   └── hfp-raw-YYYY-MM-DD.tar  # the day's original raw files, unchanged (see the raw tar policy)
└── _superseded/                # replaced source files, until purged (see "Replacing the old layout")
```

- `dt` is the **raw day directory**, i.e. the UTC date of the receive hour. It is not the date of
  `tst`. This keeps the mapping from raw files to Parquet one to one, so each day can be checked on its
  own. A message whose `tst` is just before midnight can therefore sit in the next `dt`. Queries that
  filter on `tst` near a day boundary should include the neighbouring `dt` (see [Reading](#reading)).
  In the sample, 99.9 % of the messages were received within 60 minutes of their `tst` hour.
- The 3-hour blocks of `part-001` to `part-008` use `hour(coalesce(tst, received_hour))`, so every row
  belongs to exactly one part. A part with no rows is not written (for example on a partial day). The
  blocks were 6 hours in the first draft; 3 hours halve the local sort spill of a busy day (about
  1.5 GB of raw data and 65 M rows in October 2026), and a part is still 100-200 MB.
- Files whose names start with `_` are ignored by Spark, Hive and Fabric. Parquet readers must glob
  `part-*.parquet`, not `*`.

## Parquet dataset

- **Compression:** zstd (DuckDB default level). **Row group:** 1,000,000 rows (about 13 MB).
- **Sort order within each part:**
  `event_type, date_trunc('hour', tst), transport_mode, route, dir, oday, start, veh, tst`.
  With the hour first, time-window queries skip most row groups. Within an hour, rows follow the
  journey, which compresses well. Sorting by `tst` alone made the files 2.4 times larger.
- **File metadata (key-value):** `format=hsl-hfp-archive`, `schema_version=1`, `dt=YYYY-MM-DD`.
- **No geometry column.** Position is stored as `lat` and `lon`. A WKB geometry column added 51 % to
  the size. DuckDB (`ST_Point(lon, lat)`) and GeoPandas (`points_from_xy`) create points on read.

### Columns (60)

Topic levels (split on `/`, 1-based as in DuckDB, level 1 is the empty string before the first `/`):

| Column | Type | Source |
| --- | --- | --- |
| `received_hour` | TIMESTAMPTZ | hour in the raw file name (UTC) |
| `journey_type` | VARCHAR | topic level 4 (`journey`) |
| `temporal_type` | VARCHAR | topic level 5 (`ongoing`, `upcoming`) |
| `event_type` | VARCHAR | the payload's only key (`VP`, `DUE`, ...); the topic has it in lower case |
| `transport_mode` | VARCHAR | topic level 7 (`bus`, `tram`, `train`, `metro`, `ferry`, ...) |
| `topic_operator_id` | VARCHAR | topic level 8, zero-padded as in the topic (`0022`) |
| `topic_vehicle_number` | VARCHAR | topic level 9, zero-padded (`01400`) |
| `topic_route_id` | VARCHAR | topic level 10 |
| `topic_direction_id` | VARCHAR | topic level 11 |
| `headsign` | VARCHAR | topic level 12, NULL when empty |
| `topic_start_time` | VARCHAR | topic level 13, NULL when empty |
| `next_stop` | VARCHAR | topic level 14, NULL when empty |
| `geohash_level` | TINYINT | topic level 15, NULL when empty |
| `geohash` | VARCHAR | topic levels 16-19 joined with `/` (`60;24/19/75/71`), NULL when all empty |
| `topic_sid` | VARCHAR | topic level 20, only on TLR/TLA topics |

Payload fields (JSON key in parentheses when it differs from the column name):

| Columns | Type |
| --- | --- |
| `desi`, `dir`, `line`, `start`, `loc`, `stop`, `route`, `label` | VARCHAR |
| `oper`, `hdg`, `occu` | SMALLINT |
| `veh`, `dl`, `odo`, `jrn`, `seq` | INTEGER |
| `tsi` | BIGINT |
| `tst`, `ttarr`, `ttdep` | TIMESTAMPTZ |
| `oday` | DATE |
| `spd`, `acc`, `lat`, `lon` (`long`) | DOUBLE |
| `drst`, `dr_type` (`dr-type`) | TINYINT |
| `block`, `sid`, `signal_groupid` (`signal-groupid`) | INTEGER |
| `tlp_signalgroupnbr`, `tlp_requestid`, `tlp_line_configid`, `tlp_point_configid`, `tlp_frequency`, `tlp_att_seq` | INTEGER |
| `tlp_requesttype`, `tlp_prioritylevel`, `tlp_reason`, `tlp_decision`, `tlp_protocol` | VARCHAR |
| `payload_unmapped` | VARCHAR: the whole payload object as JSON when it holds a key not listed above, else NULL |

Rebuild columns, which make the raw lines restorable byte for byte (see [Rebuilding the raw lines](#rebuilding-the-raw-lines)):

| Column | Type | Content |
| --- | --- | --- |
| `line_no` | INTEGER | 0-based line number in the source file, blank lines counted |
| `payload_shape` | VARCHAR | the payload's keys in order with their JSON type, e.g. `desi:s,dir:s,oper:i,...,stop:n` |
| `payload_raw` | VARCHAR | the original payload text, only where the rebuild from the typed columns would differ, else NULL |

- `line` and `stop` are VARCHAR because the feed sends them both as numbers and as strings. `spd` is
  sent both as an integer and a float.
- Casts are strict: a value that does not fit its type stops the conversion of the day. The fix is a new
  schema version, never a silent NULL.
- `spd`, `acc`, `lat` and `lon` store the float `-0.0` as `0.0`. DuckDB's Parquet dictionary treats the
  two as one value and keeps whichever it saw first, so the sign could otherwise flip silently (and
  `hash()`, which check C4 uses, does not tell them apart). The sign lives on in `payload_shape` (`z`).
- The topic rebuilds byte for byte from the topic columns:
  `'/hfp/v2/' || journey_type || '/' || temporal_type || '/' || lower(event_type) || '/' || transport_mode || '/' || topic_operator_id || '/' || topic_vehicle_number || '/' || topic_route_id || '/' || topic_direction_id || '/' || coalesce(headsign, '') || '/' || coalesce(topic_start_time, '') || '/' || coalesce(next_stop, '') || '/' || coalesce(geohash_level::VARCHAR, '') || '/' || coalesce(geohash, '///') || coalesce('/' || topic_sid, '')`.
- What the Parquet files do **not** keep, and the raw tar does: the gzip bytes of the original files
  (the content is kept, the compressed form is not), and anything C8 reports as not rebuildable, e.g. a
  file whose last line has no newline.

### Rebuilding the raw lines

The collector writes every line as `f"{topic} {json.dumps(payload)}"` with Python's defaults (", " and
": " separators, floats as `repr`, `ensure_ascii`). The topic follows from the topic columns. The
payload follows from the typed columns plus `payload_shape`, which records per key its JSON type:
`s` string, `S` timestamp string without milliseconds, `i` integer, `d` float, `z` the float `-0.0`,
`n` null, `x` other. Where the typed rebuild would not give the original bytes (an escaped non-ASCII
string, a key outside the schema, an unusual number format), the original payload text is in
`payload_raw`. The converter decides this per row by comparing the rebuild with the original, so the
rebuild is never silently wrong. In the sample hour 2025-03-11T14 (4.7 M lines) no row needed
`payload_raw`.

A file is rebuilt from its rows (`topic_route_id` and `received_hour` name the file), its repeats
(a `back` repeat copies the line `back` lines earlier) and its rejects, ordered by `line_no`; a gap in
`line_no` is a blank line. `hfp-archive restore` writes
the rebuilt files.

## Conversion rules

1. **Valid line:** `<topic> <payload>`, split at the first ` {`. The topic has 19 or 20 levels and
   starts `/hfp/v2/journey/`. The payload is a JSON object with exactly one key.
2. **Rejects:** every other non-empty line is written verbatim with its source file name and line
   number to `_rejects.json.gz` (JSON lines: `{"source_file": ..., "line_no": ..., "line": ...}`).
   Rejects never stop the day, but their count is in the manifest and must be reviewed.
3. **Repeats:** lines with identical bytes (compared by their 128-bit MD5) are kept once as a row: the
   first copy in file order of the earliest receive hour. Every other copy goes to `_repeats.parquet`
   with its source file and line number (`source_file`, `line_no`, `back`, `line`): a copy in the same
   file as its kept copy as `back`, the distance in lines back to it (`line` NULL); any other copy
   verbatim in `line` (`back` NULL). A `back` target is always a row or a verbatim repeat of the same
   file. In the 2025-03-11 sample 0.13 % of the lines were repeats (QoS 1 redeliveries); in October 2026
   24 %, because the feed delivers the messages of some tram routes (2015, 1001, 1003, 1007, 1009, ...)
   four times in a row. Their count is in the manifest.
4. **Empty lines** are ignored and not counted.
5. **Unexpected files:** a file in a day directory that does not match
   `<route>-YYYY-MM-DDTHH.txt[.gz]` with the directory's date stops the day.

## Raw archive (tar)

- One uncompressed POSIX (pax) tar per day: `hsl/hfp-raw/YYYY/MM/DD/hfp-raw-YYYY-MM-DD.tar`, written
  for every day under the policy `always` and only for days that fail C8 under `on-c8-failure`.
- Members are the day's original files, unchanged and in sorted order, named
  `YYYY-MM-DD/<original name>`, with their original modification times and empty owner fields.
- The members are already gzip files, so the tar is not compressed again.
- One file per day instead of about 3,500 makes the Archive tier in Azure cheaper: rehydration and read
  operations are charged per blob.

## Manifest (`_manifest.json`)

Written last, after all checks have passed, via a temporary name and an atomic rename.

```json
{
  "format": "hsl-hfp-archive",
  "schema_version": 1,
  "dt": "2025-03-11",
  "created_at": "2026-11-10T13:11:17+00:00",
  "converter": {"name": "...", "version": "...", "duckdb": "1.5.3"},
  "source": {"path": "raw/all/2025-03-11", "files": [{"name": "1001-2025-03-11T08.txt.gz", "size": 123,
             "mtime": 1741683600, "sha256": "...", "content_sha256": "..."}]},
  "interim": {"path": "interim/2025-03-11", "files": [{"name": "1001_20250311.geoparquet", "size": 1, "sha256": "..."}]},
  "parquet": [{"path": "part-000.parquet", "size": 80123456, "sha256": "..."}],
  "rejects": null,
  "repeats": {"path": "_repeats.parquet", "size": 1234, "sha256": "..."},
  "raw_archive": {"path": "hsl/hfp-raw/2025/03/11/hfp-raw-2025-03-11.tar", "size": 0, "sha256": "...", "members": 3468},
  "counts": {"raw_lines": 37888592, "rejected": 0, "repeats_removed": 50865, "rows": 37837727,
             "rows_by_event_type": {"VP": 34873500}, "rows_with_payload_raw": 0},
  "checks": {"C1 line balance": {"ok": true}, "...": {}, "C8 raw rebuild": {"ok": true}},
  "all_checks_ok": true,
  "archive_state": {"state": "done", "superseded": {"raw": "_superseded/raw/all/2025-03-11"},
                    "history": [{"state": "uploaded", "at": "..."}, {"state": "swapped", "at": "..."}]}
}
```

`sha256` is the checksum of a file as stored, read back from the storage box; `content_sha256` is the
checksum of a raw file's decompressed content, which C8 and the deep audit compare against.
`raw_archive` is `null` for a day without tar. The manifest is replaced (temporary name, checked, renamed)
when `archive_state` advances; its data fields never change. The prototype writes neither `raw_archive`
nor the rebuild fields.

## Checks before a source file may be replaced

All of these must pass; C8 must pass as well for a day that keeps no raw tar. C1-C4 run in SQL
(`verify_day.sql`) against the Parquet files **as read back**; C5-C8 are in the converter.

| Check | Rule |
| --- | --- |
| C1 line balance | raw lines = Parquet rows + repeats within an hour + repeats across hours + rejects |
| C2 per-file counts | for every source file (route × hour): lines kept = Parquet rows with that `received_hour` and `topic_route_id` |
| C3 topic rebuild | the multiset of topics rebuilt from the Parquet columns equals the multiset of original topics (compared as sums of 128-bit MD5 halves) |
| C4 content fingerprint | the order-independent sum of row hashes of the rows meant to be written equals that of the rows read back |
| C5 raw tar | every tar member's SHA-256 equals the SHA-256 of the original file, and the member list equals the day's file list |
| C6 storage read-back | the SHA-256 of every file uploaded to the storage box (or Azure) is read back from there and equals the local one |
| C7 interim sanity | per interim route group, the interim GeoParquet holds no more VP rows than the raw files hold VP lines (repeats included, as the interim builder saw them), and VP rows plus other-message lines are no more than the raw lines; otherwise interim has data that raw lacks, and the day stops for review. Interim row counts come from the Parquet footers, read on the server |
| C8 raw rebuild | for every source file, the SHA-256 of its decompressed content equals that of the content rebuilt from the Parquet rows, repeats and rejects as read back |

A negative test confirmed that C1-C4 catch a removed row, and that C4 catches a single changed value,
with one blind spot: `hash()` treats `-0.0` and `0.0` as equal, which is why floats store `-0.0` as
`0.0` and C8 compares bytes.

## Replacing the old layout

- A source file is never overwritten or deleted directly. After C1-C7 have passed and the new files are
  in place, the old files are moved to `<root>/_superseded/<their original relative path>`.
- Purging `_superseded/` is a separate, explicit step with its own date limit.
- Days that the server-side daily rsync can still write to are not converted. dckr3 syncs only
  yesterday's directories (cron 16:11 and 16:15 EET) and deletes its own raw and interim files after
  8 days (`cleanup_hsl_dirs.sh`, except interim `2015_*`), so the converter keeps a 14-day margin.
- A day with interim files but no raw files, or where C7 fails, is not converted automatically. It is
  reported for a manual decision.

## Reading

```sql
-- DuckDB: one route's positions on one day.
SELECT tst, lat, lon, spd, dl
FROM read_parquet('<root>/hsl/hfp/dt=*/part-*.parquet', hive_partitioning = true)
WHERE dt = '2025-03-11' AND event_type = 'VP' AND route = '1071';

-- A tst window that touches midnight: include the next dt as well.
SELECT count(*) FROM read_parquet('<root>/hsl/hfp/dt=*/part-*.parquet', hive_partitioning = true)
WHERE dt IN ('2025-03-11', '2025-03-12') AND tst >= '2025-03-11 23:30:00+00' AND tst < '2025-03-12 00:30:00+00';
```

`dt` is read as a string. Bus positions for the INRIX transit analysis are
`event_type = 'VP' AND transport_mode = 'bus'`, which replaces the operator list in
`inrix_speed_api.anomaly.hfp.NON_BUS_OPERATORS`.

## Reference measurements

Full-network sample 2025-03-11 (raw from 08:28 to 23:59 UTC, 3,468 files), MacBook with 14 cores and
the prototype at an 8 GB DuckDB memory limit:

| Measure | Value |
| --- | --- |
| Raw lines | 37,888,592 (827 MB gzip) |
| Parquet | 503 MB in 5 parts (0.61 × the gzip size) |
| Conversion and checks | 124 s |
| Peak process memory | 11.8 GB |
| Peak disk (stage, DuckDB spill, output) | 4.8 GB, plus the local copy of the raw day |

A full October 2026 day (2026-10-15: 1.6 GB of raw data, 85 M lines, 24 % of them repeats) gave
1.05 GB of Parquet in 9 parts and 58 MB of repeats; the converter needed 12 minutes locally and peaked
at 7.4 GB of working disk with 3-hour VP parts, see
[`hfp-archive-converter.md`](hfp-archive-converter.md#measurements).

## Prototype

[`prototype/`](prototype/) holds a single-day reference implementation. It reads local files and writes
to a local directory only. It never deletes source data and does not build the tar.

- `stage_hour.sql` parses one file hour: classification, repeats within the hour, typed columns.
- `finalize_day.sql` merges the staged hours and drops repeats across hours.
- `verify_day.sql` runs checks C1-C4.
- `run_prototype.py` drives the three, writes the parts, the rejects and the manifest.

```bash
uv run --no-project --with duckdb==1.5.3 python docs/prototype/run_prototype.py \
    data/raw/all/2025-03-11 --out /tmp/hfp-archive --memory-limit 8GB
```

## Changing the format

Any change to columns, types, sort order or partitioning raises `schema_version`. Converted days keep
their version in the manifest and the file metadata. A newer version is produced from the raw tar or
from the raw files rebuilt out of the Parquet files (`hfp-archive restore`), so old days can be
converted again without the original layout.

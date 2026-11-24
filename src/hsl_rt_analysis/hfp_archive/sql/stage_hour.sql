-- Stage 1: parse the raw files of ONE file hour into staged rows of the archive schema v1.
-- See docs/hfp-archive-format.md for the rules this implements. One hour at a time keeps memory
-- bounded (a busy hour is about 2 GB of text); finalize_day.sql then merges the staged hours.
--
-- Variables the runner sets before executing this file:
--   src      list of the hour's raw files, <route>-YYYY-MM-DDTHH.txt[.gz]
--
-- Creates in the connection:
--   file_stats    per source file: lines read, rejected and kept (the C1/C2 reference)
--   rejects       lines that do not parse, verbatim with their source file and line number
--   hour_repeats  exact repeats inside the hour with their source file and line number: as the distance
--                 back to the kept copy in the same file, or verbatim
--   parsed        one row per distinct valid line of the hour: the archive columns (except payload_raw,
--                 which the runner adds with the generated rebuild SQL, see rebuild.py) plus payload_text
--                 and the helper columns line_md5, topic_md5_hi/_lo and source_file
SET preserve_insertion_order = true;
SET TimeZone = 'UTC';

-- Floats are stored with -0.0 turned into 0.0: DuckDB's Parquet dictionary treats the two as one value and
-- keeps whichever came first, so the sign could flip silently. The sign lives on in payload_shape ('z').
CREATE OR REPLACE TEMP MACRO nz(x) AS CASE WHEN x = 0 THEN 0.0::DOUBLE ELSE x END;

-- 1. Read every line verbatim, blank lines included, in file order. The unit separator never occurs in
--    HFP lines, so each line is one column. Insertion order is preserved so that rowid follows the files
--    line by line; line_no (0-based, blank lines counted) is what lets the raw files be rebuilt (C8).
CREATE OR REPLACE TEMP TABLE all_lines AS
SELECT
    regexp_extract(filename, '([^/]+)$', 1) AS source_file,
    line
FROM read_csv(getvariable('src'), columns = {'line': 'VARCHAR'}, delim = '\x1f', quote = '', escape = '',
              header = false, filename = true, auto_detect = false);

CREATE OR REPLACE TEMP VIEW raw_lines AS
SELECT
    a.rowid AS rid,
    a.source_file,
    regexp_extract(a.source_file, '^(.+)-(\d{4}-\d{2}-\d{2}T\d{2})\.txt(\.gz)?$', 1) AS file_route,
    strptime(regexp_extract(a.source_file, '-(\d{4}-\d{2}-\d{2}T\d{2})\.txt(\.gz)?$', 1), '%Y-%m-%dT%H')::TIMESTAMPTZ
        AS file_hour,
    (a.rowid - f.first_rid)::INTEGER AS line_no,
    a.line
FROM all_lines a
JOIN (SELECT source_file, min(rowid) AS first_rid FROM all_lines GROUP BY ALL) f USING (source_file)
WHERE a.line IS NOT NULL AND a.line <> '';

-- 2. Classify: a valid line is "<topic> {<one-key JSON object>}" with a /hfp/v2/journey/ topic of 19 or 20
--    levels. Repeats of the same bytes inside the hour are dropped here (a line always lands in the file
--    of its own route, so repeats meet in the same file); repeats across hours are dropped by finalize_day.
CREATE OR REPLACE TEMP TABLE classified AS
WITH s AS (
    SELECT *, instr(line, ' {') AS sep FROM raw_lines
), t AS (
    SELECT *,
        CASE WHEN sep > 0 THEN string_split(substr(line, 1, sep - 1), '/') END AS lv,
        CASE WHEN sep > 0 THEN substr(line, sep + 1) END AS payload
    FROM s
)
SELECT rid, source_file, file_route, file_hour, line_no, line, sep, lv[6] AS topic_event,
    -- SQL does not short-circuit AND, so the JSON functions only see text that json_valid accepted.
    coalesce(
        len(lv) IN (19, 20) AND lv[2] = 'hfp' AND lv[3] = 'v2' AND lv[4] = 'journey'
        AND CASE WHEN json_valid(payload)
                 THEN json_type(payload) = 'OBJECT' AND len(json_keys(payload)) = 1 ELSE false END,
        false) AS valid,
    md5_number(line) AS line_md5
FROM t;
DROP VIEW raw_lines;
DROP TABLE all_lines;

CREATE OR REPLACE TEMP TABLE rejects AS
SELECT source_file, line_no, line FROM classified WHERE NOT valid;

-- Per source file: lines read, lines rejected and distinct valid lines kept (the C1/C2 reference).
CREATE OR REPLACE TEMP TABLE file_stats AS
SELECT source_file, file_route, file_hour,
    count(*) AS lines,
    count(*) FILTER (NOT valid) AS rejected,
    count(DISTINCT line_md5) FILTER (valid) AS kept,
    -- Valid VP lines, repeats included: what the interim files were built from (check C7).
    count(*) FILTER (valid AND topic_event = 'vp') AS vp_lines
FROM classified GROUP BY ALL;

-- 3. One copy per distinct line: the first one in file order. The others are kept verbatim as repeats.
CREATE OR REPLACE TEMP TABLE keep_rows AS
SELECT min(rid) AS id FROM classified WHERE valid GROUP BY line_md5;

-- A repeat in the same file as its kept copy (nearly always: the feed sends some tram messages four times
-- in a row) is stored as the distance back to that copy; only other repeats keep their text.
CREATE OR REPLACE TEMP TABLE hour_repeats AS
WITH k AS (
    SELECT c.line_md5, c.source_file, c.line_no FROM classified c SEMI JOIN keep_rows ON c.rid = keep_rows.id
)
SELECT c.source_file, c.line_no,
    CASE WHEN k.source_file = c.source_file THEN c.line_no - k.line_no END AS back,
    CASE WHEN k.source_file = c.source_file THEN NULL ELSE c.line END AS line
FROM classified c JOIN k USING (line_md5)
WHERE c.valid AND c.rid NOT IN (SELECT id FROM keep_rows);

-- 4. Parse. Casts are strict on purpose: a value that does not fit its type aborts the run instead of
--    being silently dropped. Keys not listed below are kept verbatim in payload_unmapped.
--    payload_shape records the key order and the JSON type of every key (z = the float -0.0, S = a timestamp
--    without milliseconds), so the payload text can be rebuilt
--    from the typed columns; payload_raw keeps the original payload text of the rows where it cannot be.
CREATE OR REPLACE TEMP VIEW parsed AS
WITH d AS (
    SELECT file_hour, source_file, line_no, line_md5, line, sep
    FROM classified SEMI JOIN keep_rows ON classified.rid = keep_rows.id
), v AS (
    SELECT file_hour, source_file, line_no, line_md5,
        string_split(substr(line, 1, sep - 1), '/') AS lv,
        md5_number_upper(substr(line, 1, sep - 1)) AS topic_md5_hi,
        md5_number_lower(substr(line, 1, sep - 1)) AS topic_md5_lo,
        substr(line, sep + 1) AS payload_text,
        substr(line, sep + 1)::JSON AS j, json_keys(substr(line, sep + 1))[1] AS ev
    FROM d
), e AS (
    SELECT *, json_extract(j, '$."' || ev || '"') AS f FROM v
), f AS (
    -- The values of '$.*' come in key order, so they pair with json_keys by position.
    SELECT *, json_keys(f) AS fk, json_extract(f, '$.*') AS fv FROM e
)
SELECT
    file_hour AS received_hour,
    lv[4] AS journey_type,
    lv[5] AS temporal_type,
    ev AS event_type,
    lv[7] AS transport_mode,
    lv[8] AS topic_operator_id,
    lv[9] AS topic_vehicle_number,
    lv[10] AS topic_route_id,
    lv[11] AS topic_direction_id,
    nullif(lv[12], '') AS headsign,
    nullif(lv[13], '') AS topic_start_time,
    nullif(lv[14], '') AS next_stop,
    nullif(lv[15], '')::TINYINT AS geohash_level,
    nullif(array_to_string(lv[16:19], '/'), '///') AS geohash,
    lv[20] AS topic_sid,
    f->>'desi' AS desi,
    f->>'dir' AS dir,
    (f->>'oper')::SMALLINT AS oper,
    (f->>'veh')::INTEGER AS veh,
    (f->>'tst')::TIMESTAMPTZ AS tst,
    (f->>'tsi')::BIGINT AS tsi,
    nz((f->>'spd')::DOUBLE) AS spd,
    (f->>'hdg')::SMALLINT AS hdg,
    nz((f->>'lat')::DOUBLE) AS lat,
    nz((f->>'long')::DOUBLE) AS lon,
    nz((f->>'acc')::DOUBLE) AS acc,
    (f->>'dl')::INTEGER AS dl,
    (f->>'odo')::INTEGER AS odo,
    (f->>'drst')::TINYINT AS drst,
    (f->>'oday')::DATE AS oday,
    (f->>'jrn')::INTEGER AS jrn,
    f->>'line' AS line,
    f->>'start' AS start,
    f->>'loc' AS loc,
    f->>'stop' AS stop,
    f->>'route' AS route,
    (f->>'occu')::SMALLINT AS occu,
    f->>'label' AS label,
    (f->>'seq')::INTEGER AS seq,
    (f->>'ttarr')::TIMESTAMPTZ AS ttarr,
    (f->>'ttdep')::TIMESTAMPTZ AS ttdep,
    (f->>'dr-type')::TINYINT AS dr_type,
    (f->>'block')::INTEGER AS block,
    (f->>'sid')::INTEGER AS sid,
    (f->>'signal-groupid')::INTEGER AS signal_groupid,
    (f->>'tlp-signalgroupnbr')::INTEGER AS tlp_signalgroupnbr,
    (f->>'tlp-requestid')::INTEGER AS tlp_requestid,
    f->>'tlp-requesttype' AS tlp_requesttype,
    f->>'tlp-prioritylevel' AS tlp_prioritylevel,
    f->>'tlp-reason' AS tlp_reason,
    f->>'tlp-decision' AS tlp_decision,
    (f->>'tlp-line-configid')::INTEGER AS tlp_line_configid,
    (f->>'tlp-point-configid')::INTEGER AS tlp_point_configid,
    (f->>'tlp-frequency')::INTEGER AS tlp_frequency,
    f->>'tlp-protocol' AS tlp_protocol,
    (f->>'tlp-att-seq')::INTEGER AS tlp_att_seq,
    CASE WHEN len(list_filter(fk, k -> NOT list_contains([
            'desi', 'dir', 'oper', 'veh', 'tst', 'tsi', 'spd', 'hdg', 'lat', 'long', 'acc', 'dl', 'odo',
            'drst', 'oday', 'jrn', 'line', 'start', 'loc', 'stop', 'route', 'occu', 'label', 'seq', 'ttarr',
            'ttdep', 'dr-type', 'block', 'sid', 'signal-groupid', 'tlp-signalgroupnbr', 'tlp-requestid',
            'tlp-requesttype', 'tlp-prioritylevel', 'tlp-reason', 'tlp-decision', 'tlp-line-configid',
            'tlp-point-configid', 'tlp-frequency', 'tlp-protocol', 'tlp-att-seq'], k))) > 0
         THEN f::VARCHAR END AS payload_unmapped,
    line_no,
    CASE WHEN json_type(f) = 'OBJECT' THEN array_to_string(list_transform(fk, (k, i) -> k || ':' ||
        CASE json_type(fv[i])
            -- Timestamps come with milliseconds (s) or, from some feeds, whole seconds only (S).
            WHEN 'VARCHAR' THEN CASE WHEN k IN ('tst', 'ttarr', 'ttdep')
                AND regexp_matches(fv[i]::VARCHAR, '^"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"$') THEN 'S' ELSE 's' END
            WHEN 'BIGINT' THEN 'i' WHEN 'UBIGINT' THEN 'i'
            WHEN 'DOUBLE' THEN CASE WHEN fv[i]::DOUBLE = 0 AND signbit(fv[i]::DOUBLE) THEN 'z' ELSE 'd' END
            WHEN 'NULL' THEN 'n' ELSE 'x' END), ',') END AS payload_shape,
    payload_text,
    line_md5,
    topic_md5_hi,
    topic_md5_lo,
    source_file
FROM f;

-- Prototype stage 1: parse the raw files of ONE file hour into staged rows of the archive schema v1.
-- See docs/hfp-archive-format.md for the rules this implements. One hour at a time keeps memory
-- bounded (a busy hour is about 1 GB of text); finalize_day.sql then merges the staged hours.
--
-- Variables the runner sets before executing this file:
--   src      list of the hour's raw files, <route>-YYYY-MM-DDTHH.txt[.gz]
--
-- Creates in the connection:
--   file_stats  per source file: lines read, rejected and kept (the C1/C2 reference)
--   rejects     lines that do not parse, verbatim with their source file
--   staged      one row per distinct valid line of the hour: the archive columns plus the helper
--               columns line_md5, topic_md5_hi/_lo and source_file (dropped by finalize_day.sql)
SET preserve_insertion_order = false;
SET TimeZone = 'UTC';

-- 1. Read every line verbatim. The unit separator never occurs in HFP lines, so each line is one column.
CREATE OR REPLACE TEMP VIEW raw_lines AS
SELECT
    regexp_extract(filename, '([^/]+)$', 1) AS source_file,
    regexp_extract(filename, '([^/]+)-(\d{4}-\d{2}-\d{2}T\d{2})\.txt(\.gz)?$', 1) AS file_route,
    strptime(regexp_extract(filename, '-(\d{4}-\d{2}-\d{2}T\d{2})\.txt(\.gz)?$', 1), '%Y-%m-%dT%H')::TIMESTAMPTZ
        AS file_hour,
    line
FROM read_csv(getvariable('src'), columns = {'line': 'VARCHAR'}, delim = '\x1f', quote = '', escape = '',
              header = false, filename = true, auto_detect = false)
WHERE line IS NOT NULL AND line <> '';

-- 2. Classify: a valid line is "<topic> {<one-key JSON object>}" with a /hfp/v2/journey/ topic of 19 or 20
--    levels. Repeats of the same bytes inside the hour are dropped here (a line always lands in the file
--    of its own route, so repeats meet in the same file); repeats across hours are dropped by finalize_day.
-- Only the line itself and a few small columns are materialised, so an hour costs about its text size.
CREATE OR REPLACE TEMP TABLE classified AS
WITH s AS (
    SELECT *, instr(line, ' {') AS sep FROM raw_lines
), t AS (
    SELECT *,
        CASE WHEN sep > 0 THEN string_split(substr(line, 1, sep - 1), '/') END AS lv,
        CASE WHEN sep > 0 THEN substr(line, sep + 1) END AS payload
    FROM s
)
SELECT source_file, file_route, file_hour, line, sep,
    -- SQL does not short-circuit AND, so the JSON functions only see text that json_valid accepted.
    coalesce(
        len(lv) IN (19, 20) AND lv[2] = 'hfp' AND lv[3] = 'v2' AND lv[4] = 'journey'
        AND CASE WHEN json_valid(payload)
                 THEN json_type(payload) = 'OBJECT' AND len(json_keys(payload)) = 1 ELSE false END,
        false) AS valid,
    md5_number(line) AS line_md5
FROM t;

CREATE OR REPLACE TEMP TABLE rejects AS
SELECT source_file, line FROM classified WHERE NOT valid;

-- Per source file: lines read, lines rejected and distinct valid lines kept (the C1/C2 reference).
CREATE OR REPLACE TEMP TABLE file_stats AS
SELECT source_file, file_route, file_hour,
    count(*) AS lines,
    count(*) FILTER (NOT valid) AS rejected,
    count(DISTINCT line_md5) FILTER (valid) AS kept
FROM classified GROUP BY ALL;

-- 3. Parse. Casts are strict on purpose: a value that does not fit its type aborts the run instead of
--    being silently dropped. Keys not listed below are kept verbatim in payload_unmapped.
-- One copy per distinct line, chosen by row id so the text itself is never sorted.
CREATE OR REPLACE TEMP TABLE keep_rows AS
SELECT min(rowid) AS id FROM classified WHERE valid GROUP BY line_md5;

CREATE OR REPLACE TEMP VIEW staged AS
WITH d AS (
    SELECT file_hour, source_file, line_md5, line, sep
    FROM classified SEMI JOIN keep_rows ON classified.rowid = keep_rows.id
), v AS (
    SELECT file_hour, source_file, line_md5,
        string_split(substr(line, 1, sep - 1), '/') AS lv,
        md5_number_upper(substr(line, 1, sep - 1)) AS topic_md5_hi,
        md5_number_lower(substr(line, 1, sep - 1)) AS topic_md5_lo,
        substr(line, sep + 1)::JSON AS j, json_keys(substr(line, sep + 1))[1] AS ev
    FROM d
), f AS (
    SELECT *, json_extract(j, '$."' || ev || '"') AS f FROM v
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
    (f->>'spd')::DOUBLE AS spd,
    (f->>'hdg')::SMALLINT AS hdg,
    (f->>'lat')::DOUBLE AS lat,
    (f->>'long')::DOUBLE AS lon,
    (f->>'acc')::DOUBLE AS acc,
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
    CASE WHEN len(list_filter(json_keys(f), k -> NOT list_contains([
            'desi', 'dir', 'oper', 'veh', 'tst', 'tsi', 'spd', 'hdg', 'lat', 'long', 'acc', 'dl', 'odo',
            'drst', 'oday', 'jrn', 'line', 'start', 'loc', 'stop', 'route', 'occu', 'label', 'seq', 'ttarr',
            'ttdep', 'dr-type', 'block', 'sid', 'signal-groupid', 'tlp-signalgroupnbr', 'tlp-requestid',
            'tlp-requesttype', 'tlp-prioritylevel', 'tlp-reason', 'tlp-decision', 'tlp-line-configid',
            'tlp-point-configid', 'tlp-frequency', 'tlp-protocol', 'tlp-att-seq'], k))) > 0
         THEN f::VARCHAR END AS payload_unmapped,
    line_md5,
    topic_md5_hi,
    topic_md5_lo,
    source_file
FROM f;

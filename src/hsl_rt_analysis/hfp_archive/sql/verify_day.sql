-- Prototype checks for one converted day. Run after finalize_day.sql in the same connection, with:
--   parquet      path of the written part-000.parquet, read back from where it was stored
--   stats_glob   glob of the day's staged file_stats files (one per hour)
-- Every statement that returns a row returns an `ok` column; anything but true fails the day.

CREATE OR REPLACE TEMP VIEW pq AS
SELECT * FROM read_parquet(getvariable('parquet'), hive_partitioning = false);
CREATE OR REPLACE TEMP VIEW stats AS
SELECT * FROM read_parquet(getvariable('stats_glob'), hive_partitioning = false);

-- C1: every raw line is accounted for exactly once: kept, repeated within its hour, repeated in a
--     later hour, or rejected.
SELECT 'C1 line balance' AS check_name, *,
       raw_lines = rows + repeats_in_hour + repeats_across_hours + rejected AS ok
FROM (SELECT
    (SELECT sum(lines) FROM stats) AS raw_lines,
    (SELECT count(*) FROM pq) AS rows,
    (SELECT sum(lines - rejected - kept) FROM stats) AS repeats_in_hour,
    (SELECT count(*) FROM later_copy) AS repeats_across_hours,
    (SELECT sum(rejected) FROM stats) AS rejected);

-- C2: per source file (route x hour), the lines expected to survive match the rows read back.
SELECT 'C2 per-file counts' AS check_name, count(*) AS mismatched_files, count(*) = 0 AS ok
FROM (
    SELECT s.file_hour AS h, s.file_route AS r, sum(s.kept - coalesce(l.n, 0)) AS n
    FROM stats s
    LEFT JOIN (SELECT source_file, count(*) AS n FROM later_copy GROUP BY ALL) l USING (source_file)
    GROUP BY ALL
) expected
FULL JOIN (
    SELECT received_hour AS h, topic_route_id AS r, count(*) AS n FROM pq GROUP BY ALL
) got USING (h, r)
WHERE coalesce(expected.n, 0) IS DISTINCT FROM coalesce(got.n, 0);

-- C3: the topic of every row rebuilds byte for byte from the columns. Compared as order-independent sums
--     of the two 64-bit halves of the MD5 of the original topics versus the topics rebuilt from the file.
CREATE OR REPLACE TEMP VIEW rebuilt AS
SELECT '/hfp/v2/' || journey_type || '/' || temporal_type || '/' || lower(event_type) || '/' || transport_mode
    || '/' || topic_operator_id || '/' || topic_vehicle_number || '/' || topic_route_id || '/'
    || topic_direction_id || '/' || coalesce(headsign, '') || '/' || coalesce(topic_start_time, '') || '/'
    || coalesce(next_stop, '') || '/' || coalesce(geohash_level::VARCHAR, '') || '/'
    || coalesce(geohash, '///') || coalesce('/' || topic_sid, '') AS topic
FROM pq;
SELECT 'C3 topic rebuild' AS check_name, *, original = rebuilt AS ok
FROM (SELECT
    (SELECT sum(topic_md5_hi::HUGEINT)::VARCHAR || '/' || sum(topic_md5_lo::HUGEINT)::VARCHAR FROM kept)
        AS original,
    (SELECT sum(md5_number_upper(topic)::HUGEINT)::VARCHAR || '/' || sum(md5_number_lower(topic)::HUGEINT)::VARCHAR
     FROM rebuilt) AS rebuilt);

-- C4: the file read back holds exactly the rows that were meant to be written (order-independent).
SELECT 'C4 content fingerprint' AS check_name, *, written = read_back AS ok
FROM (SELECT
    (SELECT sum(hash(t)::HUGEINT) FROM final t) AS written,
    (SELECT sum(hash(t)::HUGEINT) FROM pq t) AS read_back);

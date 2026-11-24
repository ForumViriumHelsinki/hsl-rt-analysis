-- Stage 2: merge the staged hours of one day into the final rows.
-- Variables the runner sets before executing this file:
--   staged_glob    glob of the day's staged row files (stage_hour.sql `staged`, one file per hour)
--
-- Creates in the connection (views stream from the staged files, so memory stays bounded):
--   day_staged  all staged rows of the day
--   later_copy  rows whose exact line already appeared in an earlier file hour (cross-hour repeats)
--   final       the archive rows: day_staged minus later_copy, without the helper columns
SET preserve_insertion_order = false;
SET TimeZone = 'UTC';

CREATE OR REPLACE TEMP VIEW day_staged AS
SELECT * FROM read_parquet(getvariable('staged_glob'), hive_partitioning = false);

-- A line kept once per hour can still repeat in another hour; keep the copy from the earliest hour.
CREATE OR REPLACE TEMP TABLE later_copy AS
WITH first_seen AS (
    SELECT line_md5, min(received_hour) AS first_hour
    FROM day_staged GROUP BY line_md5 HAVING count(*) > 1
)
SELECT s.line_md5, s.received_hour, s.source_file, s.line_no
FROM day_staged s JOIN first_seen USING (line_md5)
WHERE s.received_hour > first_seen.first_hour;

CREATE OR REPLACE TEMP VIEW kept AS
SELECT s.* FROM day_staged s ANTI JOIN later_copy USING (line_md5, received_hour);

CREATE OR REPLACE TEMP VIEW final AS
SELECT * EXCLUDE (line_md5, topic_md5_hi, topic_md5_lo, source_file) FROM kept;

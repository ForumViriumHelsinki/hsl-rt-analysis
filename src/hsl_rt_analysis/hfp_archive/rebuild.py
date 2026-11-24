"""SQL that rebuilds a raw HFP line from an archive row, generated per payload shape.

The collector writes every line as `f"{topic} {json.dumps(payload)}"` (Python defaults: ", " and ": "
separators, floats as repr, ensure_ascii). The payload text therefore follows from the typed values plus
payload_shape, the ordered list of `key:type` (s string, S a timestamp
without milliseconds, i integer, d float, z the float -0.0, n null, x other). Whatever
this rebuild cannot reproduce byte for byte is stored verbatim in payload_raw, so the rebuild never has to
be complete, only never silently wrong; check C8 proves the result against the original files.

One plain concatenation is generated per distinct (event type, shape). A day has a few hundred of them,
and a CASE over them is two orders of magnitude faster than a generic per-key lambda.
"""

import re


TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%gZ"
TS_FORMAT_SECONDS = "%Y-%m-%dT%H:%M:%SZ"
TS_COLUMNS = ("tst", "ttarr", "ttdep")

# JSON key mapped to the SQL text of its typed value. DuckDB prints a DOUBLE as the shortest round-trip
# text, as Python's repr does, and a timestamp with strftime in the collector's millisecond format.
KEY_TEXT: dict[str, str] = {
    **{k: k for k in ("desi", "dir", "line", "start", "loc", "stop", "route", "label")},
    **{f"tlp-{k}": f"tlp_{k}" for k in ("requesttype", "prioritylevel", "reason", "decision", "protocol")},
    **{
        k: f"{k}::VARCHAR"
        for k in ("oper", "veh", "tsi", "hdg", "dl", "odo", "drst", "jrn", "occu", "seq", "block", "sid")
    },
    "dr-type": "dr_type::VARCHAR",
    "signal-groupid": "signal_groupid::VARCHAR",
    **{
        f"tlp-{k}": f"tlp_{k.replace('-', '_')}::VARCHAR"
        for k in ("signalgroupnbr", "requestid", "line-configid", "point-configid", "frequency", "att-seq")
    },
    "spd": "spd::VARCHAR",
    "lat": "lat::VARCHAR",
    "long": "lon::VARCHAR",
    "acc": "acc::VARCHAR",
    **{k: f"strftime({k} AT TIME ZONE 'UTC', '{TS_FORMAT}')" for k in ("tst", "ttarr", "ttdep")},
    "oday": "strftime(oday, '%Y-%m-%d')",
}
DOUBLE_COLUMNS = {"spd": "spd", "lat": "lat", "long": "lon", "acc": "acc"}
# Keys and event types that json.dumps writes without escapes and that are safe inside a SQL literal.
PLAIN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_SHAPES = 5000

TOPIC_SQL = (
    "'/hfp/v2/' || journey_type || '/' || temporal_type || '/' || lower(event_type) || '/' || transport_mode"
    " || '/' || topic_operator_id || '/' || topic_vehicle_number || '/' || topic_route_id || '/'"
    " || topic_direction_id || '/' || coalesce(headsign, '') || '/' || coalesce(topic_start_time, '') || '/'"
    " || coalesce(next_stop, '') || '/' || coalesce(geohash_level::VARCHAR, '') || '/'"
    " || coalesce(geohash, '///') || coalesce('/' || topic_sid, '')"
)
# The collector files a line under the topic's route and the hour it was received.
FILE_STEM_SQL = "topic_route_id || '-' || strftime(received_hour AT TIME ZONE 'UTC', '%Y-%m-%dT%H')"


def value_sql(key: str, code: str) -> str | None:
    """Return the SQL text of one key's JSON value, or None where it cannot be rebuilt.

    Args:
        key: JSON key.
        code: Shape type code.

    Returns:
        SQL expression or None.
    """
    if code == "n":
        return "'null'"
    if code == "S":
        return (
            f"'\"' || strftime({key} AT TIME ZONE 'UTC', '{TS_FORMAT_SECONDS}') || '\"'" if key in TS_COLUMNS else None
        )
    if code == "z":
        return "'-0.0'" if key in DOUBLE_COLUMNS else None
    text = KEY_TEXT.get(key)
    if text is None:
        return None
    if code == "s":
        return f"'\"' || {text} || '\"'"
    if code == "d":
        return text
    if code == "i":
        if key in DOUBLE_COLUMNS:
            col = DOUBLE_COLUMNS[key]
            return f"(CASE WHEN {col} = trunc({col}) AND abs({col}) < 1e15 THEN {col}::BIGINT::VARCHAR END)"
        return text
    return None


def shape_sql(event_type: str, shape: str) -> str | None:
    """Return the SQL that rebuilds the payload text of one (event type, shape), or None.

    Args:
        event_type: The payload's only key, e.g. VP.
        shape: payload_shape value, e.g. 'desi:s,dir:s,...'.

    Returns:
        SQL concatenation, or None if some key cannot be rebuilt.
    """
    if not PLAIN_RE.match(event_type):
        return None
    pieces = []
    for element in shape.split(",") if shape else []:
        key, sep, code = element.rpartition(":")
        if not sep or not PLAIN_RE.match(key):
            return None
        value = value_sql(key, code)
        if value is None:
            return None
        pieces.append(f"'\"{key}\": ' || {value}")
    body = " || ', ' || ".join(pieces) if pieces else "''"
    return f"'{{\"{event_type}\": {{' || {body} || '}}}}'"


def payload_sql(shapes: list[tuple[str, str]]) -> str:
    """Return a SQL expression that rebuilds the payload text of a row of the archive columns.

    Args:
        shapes: Distinct (event_type, payload_shape) pairs that occur, most frequent first.

    Returns:
        SQL expression; NULL for rows whose shape cannot be rebuilt.
    """
    by_event: dict[str, list[str]] = {}
    for event_type, shape in shapes[:MAX_SHAPES]:
        if shape is None:
            continue
        expr = shape_sql(event_type, shape)
        if expr is not None:
            literal = shape.replace("'", "''")
            by_event.setdefault(event_type, []).append(f"WHEN '{literal}' THEN {expr}")
    if not by_event:
        return "NULL::VARCHAR"
    branches = [f"WHEN '{ev}' THEN CASE payload_shape {' '.join(whens)} END" for ev, whens in by_event.items()]
    return f"(CASE event_type {' '.join(branches)} END)"


def line_sql(shapes: list[tuple[str, str]]) -> str:
    """Return a SQL expression that rebuilds the whole raw line of an archive row.

    Args:
        shapes: Distinct (event_type, payload_shape) pairs that occur.

    Returns:
        SQL expression.
    """
    return f"({TOPIC_SQL}) || ' ' || coalesce(payload_raw, {payload_sql(shapes)})"


def shapes_query(relation: str) -> str:
    """Return the query listing the distinct shapes of a relation, most frequent first.

    Args:
        relation: Table, view or table function holding event_type and payload_shape.

    Returns:
        SQL query.
    """
    return (
        f"SELECT event_type, payload_shape FROM {relation} WHERE payload_shape IS NOT NULL "
        "GROUP BY ALL ORDER BY count(*) DESC"
    )

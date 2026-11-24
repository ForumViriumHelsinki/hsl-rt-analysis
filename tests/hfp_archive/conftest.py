"""Fixtures: a fake storage box directory holding synthetic raw and interim days."""

import gzip
import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pytest

from hsl_rt_analysis.hfp_archive.config import SENTINEL, Settings
from hsl_rt_analysis.hfp_archive.remote import LocalRemote
from hsl_rt_analysis.hfp_archive.state import StateDB


DAY = "2026-01-05"
ROOT = "2025_HSL_RT"


def topic(event: str, route: str, mode: str = "bus", sid: str | None = None, headsign: str = "Kamppi") -> str:
    """Return an HFP topic with 19 levels (20 with a signal id).

    Args:
        event: Event type in lower case.
        route: Route id (topic level 10).
        mode: Transport mode.
        sid: Optional signal id (TLR / TLA topics).
        headsign: Headsign level.

    Returns:
        The topic.
    """
    t = f"/hfp/v2/journey/ongoing/{event}/{mode}/0022/01400/{route}/1/{headsign}/08:05/1234567/5/60;24/19/75/71"
    return t + (f"/{sid}" if sid else "")


def vp(route: str, hour: int, second: int, **override: object) -> str:
    """Return one VP line written the way the collector writes it.

    Args:
        route: Route id.
        hour: UTC hour of the timestamp.
        second: Seconds into the hour, for distinct timestamps.
        **override: Payload fields to replace or add (None means JSON null).

    Returns:
        The line without newline.
    """
    payload = {
        "desi": route[:4], "dir": "1", "oper": 22, "veh": 1400,
        "tst": f"2026-01-05T{hour:02d}:00:{second % 60:02d}.123Z", "tsi": 1767600000 + 3600 * hour + second,
        "spd": 4.5, "hdg": 310, "lat": 60.195338, "long": 24.955357, "acc": 0.62,
        "dl": -186, "odo": 7328, "drst": 0, "oday": DAY, "jrn": 687, "line": 28, "start": "08:05", "loc": "GPS",
        "stop": None, "route": route, "occu": 0,
    }  # fmt: skip
    payload.update(override)
    return f"{topic('vp', route)} {json.dumps({'VP': payload})}"


def hour_lines(route: str, hour: int) -> list[str]:
    """Return the lines of one synthetic route-hour file, with every case the converter must handle.

    Args:
        route: Route id.
        hour: UTC hour.

    Returns:
        Lines without newlines.
    """
    lines = [vp(route, hour, s) for s in range(6)]
    lines += [
        vp(route, hour, 30, spd=4),  # a float sent as an integer
        vp(route, hour, 31, acc=-0.0),  # negative zero
        vp(route, hour, 32, tst=f"2026-01-05T{hour:02d}:00:32Z"),  # timestamp without milliseconds
        vp(route, hour, 33, desi="Ä1"),  # json.dumps escapes non-ASCII: kept in payload_raw
        vp(route, hour, 34, foo=1),  # a key outside the schema: payload_unmapped
        vp(route, hour, 35, odo=None, drst=None),
        lines_dep(route, hour),
        tlr_line(route, hour),
    ]
    lines.append(lines[0])  # an exact repeat inside the hour (QoS 1 redelivery)
    return lines


def lines_dep(route: str, hour: int) -> str:
    """Return a DEP line with the timetable fields.

    Args:
        route: Route id.
        hour: UTC hour.

    Returns:
        The line.
    """
    body = {
        "desi": "1",
        "dir": "1",
        "oper": 22,
        "veh": 1400,
        "tst": f"2026-01-05T{hour:02d}:10:00.015Z",
        "stop": 1220416,
        "ttarr": "2026-01-05T08:09:00.000Z",
        "ttdep": "2026-01-05T08:10:00.000Z",
        "route": route,
    }
    return f"{topic('dep', route)} {json.dumps({'DEP': body})}"


def tlr_line(route: str, hour: int) -> str:
    """Return a TLR line: a 20-level topic and the traffic light fields.

    Args:
        route: Route id.
        hour: UTC hour.

    Returns:
        The line.
    """
    body = {
        "desi": "1",
        "dir": "1",
        "oper": 22,
        "veh": 1400,
        "tst": f"2026-01-05T{hour:02d}:00:01.000Z",
        "sid": 4021,
        "signal-groupid": 3,
        "tlp-requestid": 7,
        "tlp-requesttype": "NORMAL",
    }
    return f"{topic('tlr', route, sid='4021')} {json.dumps({'TLR': body})}"


def write_gz(path: Path, text: str, mtime: int = 1767600000) -> None:
    """Write a gzip file the way the collector's compressor does and set its mtime.

    Args:
        path: Destination.
        text: Content.
        mtime: Modification time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.utime(path, (mtime, mtime))


def make_raw_day(box: Path, day: str = DAY, routes: tuple[str, ...] = ("1001", "1001 5", "2015", "9995")) -> Path:
    """Write a synthetic raw day directory: 2 hours x routes, with repeats, a reject and a blank line.

    Args:
        box: Directory standing in for the storage box SSH home.
        day: The day.
        routes: Route ids.

    Returns:
        The day directory.
    """
    day_dir = box / ROOT / "raw" / "all" / day
    for route in routes:
        for hour in (8, 9):
            lines = hour_lines(route, hour)
            if hour == 9:
                # A repeat in a later hour, itself repeated in its own hour: the second copy points back at
                # the first, which is no row but a verbatim repeat itself.
                lines += [hour_lines(route, 8)[1]] * 2
            if route == "2015" and hour == 8:
                lines += ["", "garbage line without a payload", "/hfp/v2/journey/ongoing/vp/bus {not json"]
            write_gz(day_dir / f"{route}-{day}T{hour:02d}.txt.gz", "\n".join(lines) + "\n")
    return day_dir


def make_interim_day(box: Path, day: str = DAY, vp_rows: dict[str, int] | None = None) -> Path:
    """Write a synthetic interim day: one geoparquet with VP rows and an other-messages file per group.

    Args:
        box: Fake storage box home.
        day: The day.
        vp_rows: Route group mapped to its number of VP rows.

    Returns:
        The interim day directory.
    """
    vp_rows = vp_rows if vp_rows is not None else {"1001": 20, "2015": 10, "9xxx": 10}
    d = box / ROOT / "interim" / day
    d.mkdir(parents=True, exist_ok=True)
    stamp = day.replace("-", "")
    for group, n in vp_rows.items():
        duckdb.sql(
            f"COPY (SELECT range AS veh FROM range({n})) TO '{d / f'{group}_{stamp}.geoparquet'}' (FORMAT parquet)"
        )
        write_gz(d / f"{group}_{stamp}_other_messages.txt.gz", "a\nb\n")
    return d


def snapshot(path: Path) -> dict[str, bytes]:
    """Return every file below a directory with its bytes.

    Args:
        path: Directory.

    Returns:
        Relative path mapped to content.
    """
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


@dataclass
class Box:
    """A fake storage box and everything a pipeline needs around it."""

    base: Path
    remote: LocalRemote
    settings: Settings
    db: StateDB
    raw_before: dict[str, bytes]
    interim_before: dict[str, bytes]

    @property
    def root(self) -> Path:
        """Return the archive root directory."""
        return self.base / ROOT


@pytest.fixture
def make_box(tmp_path: Path) -> Iterator[Callable[..., Box]]:
    """Return a factory for a fake storage box with one synthetic day.

    Args:
        tmp_path: pytest temporary directory.

    Yields:
        Factory taking keyword overrides for Settings.
    """
    dbs: list[StateDB] = []

    def factory(interim: bool = True, **overrides: Any) -> Box:  # noqa: ANN401
        base = tmp_path / "box"
        raw_dir = make_raw_day(base)
        interim_dir = make_interim_day(base) if interim else None
        (base / ROOT / SENTINEL).touch()
        settings = Settings(
            remote="local", state_dir=tmp_path / "state", work_dir=tmp_path / "work", memory_limit="1GB", threads=2,
            min_remote_free_gb=0, min_local_free_gb=0, **overrides,
        )  # fmt: skip
        db = StateDB(settings.state_db)
        dbs.append(db)
        return Box(
            base,
            LocalRemote(base, ROOT),
            settings,
            db,
            snapshot(raw_dir),
            snapshot(interim_dir) if interim_dir else {},
        )

    yield factory
    for db in dbs:
        db.close()

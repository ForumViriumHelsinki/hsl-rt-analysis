"""Tests of audit, purge, status and report, and of the smaller building blocks."""

import json
import os
import signal
import tarfile
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import duckdb
import pytest

from hsl_rt_analysis.hfp_archive import ops
from hsl_rt_analysis.hfp_archive.checks import build_tar, check_c5, route_group
from hsl_rt_analysis.hfp_archive.discover import inventory
from hsl_rt_analysis.hfp_archive.guards import StopFlag
from hsl_rt_analysis.hfp_archive.pipeline import Pipeline
from hsl_rt_analysis.hfp_archive.rebuild import payload_sql, shape_sql
from hsl_rt_analysis.hfp_archive.remote import check_safe, parse_ls_output, parse_sha256_output, sha256_of
from hsl_rt_analysis.hfp_archive.state import DayState
from tests.hfp_archive.conftest import DAY, Box, make_interim_day, snapshot


@pytest.fixture
def done_box(make_box: Callable[..., Box]) -> Box:
    """Return a box whose day has gone all the way through.

    Args:
        make_box: Box factory.

    Returns:
        The box.
    """
    box = make_box()
    Pipeline(box.settings, box.remote, box.db).run([DAY])
    assert box.db.require(DAY).state == DayState.DONE
    return box


@pytest.mark.parametrize(("deep", "full"), [(False, False), (False, True), (True, False)])
def test_audit_passes_on_a_converted_day(done_box: Box, deep: bool, full: bool) -> None:
    """The audit reads only the storage box and confirms a good day."""
    result = ops.audit_day(done_box.remote, done_box.settings, DAY, deep, full)
    assert result["ok"], result


def test_audit_catches_damage(done_box: Box) -> None:
    """A changed Parquet file or a changed superseded original fails the audit."""
    part = next((done_box.root / "hsl" / "hfp" / f"dt={DAY}").glob("part-*.parquet"))
    part.write_bytes(part.read_bytes()[:-1] + b"X")
    original = next((done_box.root / "_superseded" / "raw" / "all" / DAY).iterdir())
    original.write_bytes(b"changed")
    result = ops.audit_day(done_box.remote, done_box.settings, DAY, deep=False, full=True)
    assert result["parquet sha256"] is False
    assert result["superseded raw sha256"] is False
    assert result["ok"] is False
    assert ops.audit_day(done_box.remote, done_box.settings, DAY, deep=False)["superseded raw sizes"] is False


def test_purge_is_a_dry_run_and_needs_an_audit(done_box: Box) -> None:
    """Without an audit nothing qualifies; with one, the default only lists; --execute deletes."""
    inv = inventory(done_box.remote)
    ok, blocked = ops.purge_candidates(done_box.db, inv, "2026-01-06", timedelta(days=7))
    assert ok == []
    assert blocked == [f"{DAY}: no passed audit"]
    done_box.db.record_audit(DAY, True, False, {})
    ok, _ = ops.purge_candidates(done_box.db, inv, "2026-01-06", timedelta(days=7))
    assert ok == [DAY]
    assert ops.purge_candidates(done_box.db, inv, DAY, timedelta(days=7))[0] == []  # strictly before
    before = snapshot(done_box.root)
    freed = ops.purge(done_box.remote, inv, ok, execute=False)
    assert freed > 0
    assert snapshot(done_box.root) == before
    ops.purge(done_box.remote, inv, ok, execute=True)
    assert not (done_box.root / "_superseded" / "raw" / "all" / DAY).exists()
    assert not (done_box.root / "_superseded" / "interim" / DAY).exists()
    assert (done_box.root / "hsl" / "hfp" / f"dt={DAY}" / "_manifest.json").exists()


def test_purge_refuses_paths_outside_superseded(done_box: Box) -> None:
    """The delete primitive itself cannot touch anything outside _superseded/."""
    with pytest.raises(ValueError, match="_superseded"):
        done_box.remote.remove_tree(f"hsl/hfp/dt={DAY}")


def test_status_and_report(done_box: Box) -> None:
    """status lists progress per day; report lists interim-only days."""
    make_interim_day(done_box.base, day="2024-08-20")
    make_interim_day(done_box.base, day="2024-08-21")
    lines = ops.status_lines(done_box.db, pending=["2026-01-06"], show_all=True)
    assert any(line.startswith(DAY) and "done" in line for line in lines)
    assert "not started: 1" in lines[-2] or "not started: 1" in "\n".join(lines)
    report = ops.report_lines(inventory(done_box.remote), done_box.db)
    assert "  2024-08-20 .. 2024-08-21 (2 days)" in report


def test_c5_detects_a_changed_member(tmp_path: Path) -> None:
    """C5 fails when a member differs from the original or a file is missing."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("b-2026-01-05T01.txt.gz", "a-2026-01-05T01.txt.gz"):
        (raw / name).write_bytes(name.encode())
    expected = {p.name: sha256_of(p) for p in raw.iterdir()}
    tar = build_tar(raw, list(expected), tmp_path / "t.tar", DAY)
    assert check_c5(tar, expected, DAY)["ok"]
    with tarfile.open(tar) as t:
        assert [m.name for m in t] == [f"{DAY}/a-2026-01-05T01.txt.gz", f"{DAY}/b-2026-01-05T01.txt.gz"]
    assert not check_c5(tar, {**expected, "a-2026-01-05T01.txt.gz": "0" * 64}, DAY)["ok"]
    assert not check_c5(tar, {**expected, "c-2026-01-05T01.txt.gz": "0" * 64}, DAY)["ok"]


def test_stop_flag_handles_signals() -> None:
    """The first SIGINT only sets the flag; the second interrupts."""
    with StopFlag() as stop:
        os.kill(os.getpid(), signal.SIGINT)
        assert stop.requested
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)


def test_route_groups_match_the_interim_builder() -> None:
    """Route groups follow exploration/hfp_data_processor.py."""
    assert route_group("1001 5-2026-01-05T08.txt.gz") == "1001"
    assert route_group("4999-2026-01-05T08.txt.gz") == "4999"
    assert route_group("9995-2026-01-05T08.txt.gz") == "9xxx"
    assert route_group("31M1-2026-01-05T08.txt.gz") == "31M1"  # string comparison, as in the builder
    assert route_group("M1-2026-01-05T08.txt.gz") == "other"


def test_shape_rebuild_matches_json_dumps() -> None:
    """The generated SQL writes payloads exactly as Python's json.dumps does."""
    payload = {"desi": "1", "spd": 4, "acc": -0.0, "lat": 60.1, "tst": "2026-01-05T08:00:00Z", "stop": None}
    shape = "desi:s,spd:i,acc:z,lat:d,tst:S,stop:n"
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    row = (
        "SELECT 'VP' AS event_type, '" + shape + "' AS payload_shape, '1' AS desi, 4.0::DOUBLE AS spd, "
        "0.0::DOUBLE AS acc, 60.1::DOUBLE AS lat, NULL::DOUBLE AS lon, TIMESTAMPTZ '2026-01-05 08:00:00+00' AS tst, "
        "NULL AS stop"
    )
    got = con.sql(f"SELECT {payload_sql([('VP', shape)])} FROM ({row})").fetchone()[0]
    assert got == json.dumps({"VP": payload})
    assert shape_sql("VP", "unknown-key:s") is None
    assert shape_sql("VP", 'we"ird:s') is None


def test_remote_output_parsers() -> None:
    """sha256sum and ls output with spaces in names parse correctly."""
    sha = "a" * 64
    assert parse_sha256_output(f"{sha}  root/d/1001 5-x.txt.gz\n", "root/d/") == {"1001 5-x.txt.gz": sha}
    ls = "total 8\ndrwxr-xr-x 2 u g 4096 1700000000 .\n-rw-r--r-- 1 u g 12 1700000001 1001 5-x.txt.gz\n"
    [entry] = parse_ls_output(ls)
    assert (entry.name, entry.size, entry.mtime, entry.is_dir) == ("1001 5-x.txt.gz", 12, 1700000001, False)
    assert check_safe("hsl/hfp/dt=2026-01-05") == "hsl/hfp/dt=2026-01-05"
    for bad in ("raw/all/../x", "raw/all/a b", "raw/$(rm)"):
        with pytest.raises(ValueError, match="unsafe"):
            check_safe(bad)

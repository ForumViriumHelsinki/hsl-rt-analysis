"""End-to-end tests of the pipeline against a fake storage box directory."""

import gzip
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pytest

from hsl_rt_analysis.hfp_archive.config import SENTINEL, RawArchivePolicy
from hsl_rt_analysis.hfp_archive.discover import inventory, select_days
from hsl_rt_analysis.hfp_archive.guards import AlreadyRunningError, RunLock, StopFlag
from hsl_rt_analysis.hfp_archive.pipeline import Pipeline
from hsl_rt_analysis.hfp_archive.remote import LocalRemote, RemoteUnavailableError, SentinelMissingError
from hsl_rt_analysis.hfp_archive.restore import write_restored_day
from hsl_rt_analysis.hfp_archive.state import DayState
from tests.hfp_archive.conftest import DAY, Box, make_interim_day, snapshot, write_gz


class Crash(BaseException):
    """Stands for the process being killed: not an Exception, so nothing in the pipeline catches it."""


def assert_converted(box: Box) -> dict[str, Any]:
    """Assert everything a finished day must satisfy and return its manifest.

    Args:
        box: The fake storage box.

    Returns:
        The manifest.
    """
    root = box.root
    dt = root / "hsl" / "hfp" / f"dt={DAY}"
    manifest = json.loads((dt / "_manifest.json").read_text())
    assert manifest["all_checks_ok"] is True
    assert manifest["archive_state"]["state"] == "done"
    # C8 may fail only on a day that keeps its raw tar.
    failed = [name for name, r in manifest["checks"].items() if not r["ok"]]
    assert failed in ([], ["C8 raw rebuild"] if manifest["raw_archive"] else []), manifest["checks"]
    # The originals are gone from their place and kept byte for byte in _superseded/.
    assert not (root / "raw" / "all" / DAY).exists()
    assert snapshot(root / "_superseded" / "raw" / "all" / DAY) == box.raw_before
    if box.interim_before:
        assert not (root / "interim" / DAY).exists()
        assert snapshot(root / "_superseded" / "interim" / DAY) == box.interim_before
    # No temporary names are left behind, and readers see only finished files.
    leftovers = [p for p in root.rglob("*") if p.name.endswith(".tmp") or ".dt=" in p.name]
    assert leftovers == []
    # Every Parquet row, repeat and reject together rebuild every original file exactly.
    if manifest["raw_archive"]:
        with tarfile.open(root / manifest["raw_archive"]["path"]) as tar:
            members = {m.name.split("/", 1)[1]: tar.extractfile(m).read() for m in tar}  # type: ignore[union-attr]
        assert members == box.raw_before
    return manifest


def run(box: Box, **kwargs: Any) -> Pipeline:  # noqa: ANN401
    """Run the pipeline over the box's day.

    Args:
        box: The fake storage box.
        **kwargs: Pipeline keyword arguments.

    Returns:
        The pipeline.
    """
    pipeline = Pipeline(box.settings, box.remote, box.db, **kwargs)
    pipeline.run([DAY])
    return pipeline


def test_full_day(make_box: Callable[..., Box], tmp_path: Path) -> None:
    """A day goes all the way through and every original line can be rebuilt from Parquet."""
    box = make_box()
    run(box)
    assert box.db.require(DAY).state == DayState.DONE
    manifest = assert_converted(box)
    counts = manifest["counts"]
    assert counts["rejected"] == 2
    assert counts["repeats_removed"] == 16  # one per file inside the hour, two per route in the later hour
    assert counts["rows_with_payload_raw"] == 16  # escaped non-ASCII and an unmapped key, per file
    assert manifest["checks"]["C8 raw rebuild"]["files"] == 8
    dt = box.root / "hsl" / "hfp" / f"dt={DAY}"
    out = tmp_path / "restored"
    names = [f["name"] for f in manifest["source"]["files"]]
    write_restored_day(
        duckdb.connect(),
        sorted(dt.glob("part-*.parquet")),
        dt / "_repeats.parquet",
        dt / "_rejects.json.gz",
        names,
        out,
    )
    for name, data in box.raw_before.items():
        assert gzip.decompress((out / name).read_bytes()) == gzip.decompress(data)
    assert not box.settings.day_work_dir(DAY).exists()


CHECKPOINTS = [
    "fetch:downloaded",
    "state:fetched",
    "state:converted",
    "state:verified",
    f"upload:tmp:hfp-raw-{DAY}.tar",
    "upload:file:part-003.parquet",
    "upload:manifest",
    "state:uploaded",
    "swap:raw",
    "swap:interim",
    "state:swapped",
]


@pytest.mark.parametrize("point", CHECKPOINTS)
def test_crash_at_every_step_resumes(make_box: Callable[..., Box], point: str) -> None:
    """A run killed at any point finishes on the next run, with nothing lost or duplicated."""
    box = make_box()

    def crash(name: str) -> None:
        if name == point:
            raise Crash(name)

    with pytest.raises(Crash):
        run(box, checkpoint=crash)
    assert box.db.require(DAY).state != DayState.DONE
    run(box)
    assert box.db.require(DAY).state == DayState.DONE
    assert_converted(box)


def test_partial_conversion_output_is_redone(make_box: Callable[..., Box]) -> None:
    """Junk left in the work directory by a crash in the middle of the conversion does not survive."""
    box = make_box()
    pipeline = Pipeline(box.settings, box.remote, box.db)
    box.db.set_state(DAY, pipeline.fetch(DAY))
    out = box.settings.day_work_dir(DAY) / "out"
    out.mkdir(parents=True)
    (out / "part-004.parquet").write_bytes(b"half a file")
    pipeline.run([DAY])
    assert_converted(box)


def test_stop_signal_stops_at_step_boundary(make_box: Callable[..., Box]) -> None:
    """A SIGINT/SIGTERM lets the current step finish and stops before the next one."""
    box = make_box()
    stop = StopFlag()

    def request_stop(name: str) -> None:
        if name == "state:converted":
            stop.requested = True

    summary = Pipeline(box.settings, box.remote, box.db, stop, request_stop).run([DAY])
    assert summary.stopped_reason == "stop requested"
    assert box.db.require(DAY).state == DayState.CONVERTED
    run(box)
    assert_converted(box)


def test_sentinel_disappears_before_upload(make_box: Callable[..., Box]) -> None:
    """If the storage box root stops looking right, nothing is written and the run stops."""
    box = make_box()

    def unmount(name: str) -> None:
        if name == "state:verified":
            (box.root / SENTINEL).unlink()

    with pytest.raises(SentinelMissingError):
        run(box, checkpoint=unmount)
    assert box.db.require(DAY).state == DayState.VERIFIED
    assert not (box.root / "hsl").exists()
    assert snapshot(box.root / "raw" / "all" / DAY) == box.raw_before
    (box.root / SENTINEL).touch()
    run(box)
    assert_converted(box)


class FlakyRemote(LocalRemote):
    """Loses the connection once, at the first call of a chosen operation."""

    def __init__(self, base: Path, operation: str) -> None:
        """Create the remote.

        Args:
            base: Fake storage box home.
            operation: Method name that fails once.
        """
        super().__init__(base)
        self.operation = operation
        self.failed = False

    def __getattribute__(self, name: str) -> object:
        """Fail the chosen operation once.

        Args:
            name: Attribute name.

        Returns:
            The attribute.

        Raises:
            RemoteUnavailableError: On the first call of the chosen operation.
        """
        attr = super().__getattribute__(name)
        if name == object.__getattribute__(self, "operation") and not object.__getattribute__(self, "failed"):
            self.failed = True
            msg = f"connection lost during {name}"
            raise RemoteUnavailableError(msg)
        return attr


@pytest.mark.parametrize("operation", ["download_dir", "upload_file", "move", "sha256_dir"])
def test_storage_box_unreachable(make_box: Callable[..., Box], operation: str) -> None:
    """A lost connection stops the run without damage, and the next run finishes the day."""
    box = make_box()
    flaky = FlakyRemote(box.base, operation)
    with pytest.raises(RemoteUnavailableError):
        Pipeline(box.settings, flaky, box.db).run([DAY])
    assert box.db.require(DAY).state != DayState.DONE
    run(box)
    assert_converted(box)


class CorruptingRemote(LocalRemote):
    """Damages every uploaded data file, as a bad link or disk would."""

    def upload_file(self, local_path: Path, rel: str) -> None:
        """Upload, then flip the last byte of Parquet files.

        Args:
            local_path: Local source.
            rel: Remote destination.
        """
        super().upload_file(local_path, rel)
        if local_path.suffix == ".parquet":
            path = self._p(rel)
            data = bytearray(path.read_bytes())
            data[-1] ^= 0xFF
            path.write_bytes(bytes(data))


def test_c6_mismatch_fails_without_rename(make_box: Callable[..., Box]) -> None:
    """A file that reads back differently from the storage box is never renamed into place."""
    box = make_box()
    Pipeline(box.settings, CorruptingRemote(box.base), box.db).run([DAY])
    record = box.db.require(DAY)
    assert record.state == DayState.FAILED
    assert "C6" in (record.last_error or "")
    assert not (box.root / "hsl" / "hfp" / f"dt={DAY}").exists()
    assert snapshot(box.root / "raw" / "all" / DAY) == box.raw_before


def test_c7_interim_with_more_data_needs_review(make_box: Callable[..., Box]) -> None:
    """Interim holding more VP rows than raw stops the day before anything is uploaded."""
    box = make_box()
    make_interim_day(box.base, vp_rows={"1001": 10_000})
    run(box)
    record = box.db.require(DAY)
    assert record.state == DayState.NEEDS_REVIEW
    assert "C7" in (record.last_error or "")
    assert not (box.root / "hsl").exists()
    assert (box.root / "interim" / DAY).exists()


def test_source_changed_before_swap_is_not_moved(make_box: Callable[..., Box]) -> None:
    """A raw file that changed after it was converted keeps the originals in place."""
    box = make_box()

    def touch_source(name: str) -> None:
        if name == "state:uploaded":
            write_gz(box.root / "raw" / "all" / DAY / f"1001-{DAY}T09.txt.gz", "late line\n")

    run(box, checkpoint=touch_source)
    assert box.db.require(DAY).state == DayState.NEEDS_REVIEW
    assert (box.root / "raw" / "all" / DAY).exists()
    assert not (box.root / "_superseded").exists()


def test_day_without_interim(make_box: Callable[..., Box]) -> None:
    """A raw day with no interim directory converts normally."""
    box = make_box(interim=False)
    run(box)
    manifest = assert_converted(box)
    assert manifest["interim"] is None


def test_tar_only_when_c8_fails(make_box: Callable[..., Box]) -> None:
    """With the on-c8-failure policy a day whose files all rebuild keeps no tar."""
    box = make_box(raw_archive=RawArchivePolicy.ON_C8_FAILURE)
    run(box)
    manifest = assert_converted(box)
    assert manifest["raw_archive"] is None
    assert not (box.root / "hsl" / "hfp-raw").exists()


def test_missing_final_newline_keeps_tar(make_box: Callable[..., Box]) -> None:
    """A file C8 cannot rebuild (no final newline) makes the day keep its tar, whatever the policy."""
    box = make_box(raw_archive=RawArchivePolicy.ON_C8_FAILURE)
    path = box.root / "raw" / "all" / DAY / f"9995-{DAY}T09.txt.gz"
    write_gz(path, gzip.decompress(path.read_bytes()).decode().rstrip("\n"))
    box.raw_before = snapshot(box.root / "raw" / "all" / DAY)
    run(box)
    manifest = assert_converted(box)
    assert manifest["checks"]["C8 raw rebuild"]["ok"] is False
    assert manifest["checks"]["C8 raw rebuild"]["examples"] == [f"9995-{DAY}T09.txt.gz"]
    assert manifest["raw_archive"] is not None


def test_unexpected_file_fails_the_day(make_box: Callable[..., Box]) -> None:
    """A file that is not one of the day's raw files stops the day."""
    box = make_box()
    (box.root / "raw" / "all" / DAY / "notes.txt").write_text("hello")
    run(box)
    assert box.db.require(DAY).state == DayState.FAILED
    assert (box.root / "raw" / "all" / DAY / "notes.txt").exists()


def test_retry_resumes_failed_day(make_box: Callable[..., Box]) -> None:
    """retry puts a failed day back where it stopped."""
    box = make_box()
    Pipeline(box.settings, CorruptingRemote(box.base), box.db).run([DAY])
    pipeline = Pipeline(box.settings, box.remote, box.db)
    pipeline.retry(DAY)
    assert box.db.require(DAY).state == DayState.VERIFIED
    pipeline.run([DAY])
    assert_converted(box)


def test_dry_run_changes_nothing_remotely(make_box: Callable[..., Box]) -> None:
    """A dry run converts and verifies locally and leaves the storage box as it was."""
    box = make_box()
    before = snapshot(box.root)
    Pipeline(box.settings, box.remote, box.db).run([DAY], dry_run=True)
    assert box.db.require(DAY).state == DayState.VERIFIED
    assert snapshot(box.root) == before


def test_lock_prevents_parallel_runs(tmp_path: Path) -> None:
    """A second run cannot start while the first holds the lock."""
    with RunLock(tmp_path / "run.lock"), pytest.raises(AlreadyRunningError), RunLock(tmp_path / "run.lock"):
        pass


def test_cutoff_and_interim_only_days(make_box: Callable[..., Box]) -> None:
    """Days after the cutoff and days with only interim data are never selected."""
    box = make_box()
    make_interim_day(box.base, day="2025-01-01")
    inv = inventory(box.remote)
    assert inv.interim_only == ["2025-01-01"]
    assert select_days(inv, box.db, cutoff="2026-01-04") == []
    assert select_days(inv, box.db, cutoff="2026-01-05") == [DAY]

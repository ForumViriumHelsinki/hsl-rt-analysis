"""Copying finished days to a blob container, against a fake storage box and a fake container."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from hsl_rt_analysis.hfp_archive.blobstore import LocalBlobStore
from hsl_rt_analysis.hfp_archive.cli import main
from hsl_rt_analysis.hfp_archive.config import SENTINEL, hfp_day
from hsl_rt_analysis.hfp_archive.guards import StopFlag
from hsl_rt_analysis.hfp_archive.manifest import MANIFEST_NAME
from hsl_rt_analysis.hfp_archive.pipeline import Pipeline
from hsl_rt_analysis.hfp_archive.remote import SentinelMissingError, sha256_of
from hsl_rt_analysis.hfp_archive.replicate import (
    BlobLayout,
    ReplicaConflictError,
    Replicator,
    normalize_prefix,
    select_replica_days,
)
from hsl_rt_analysis.hfp_archive.state import DayState, ReplicaState
from tests.hfp_archive.conftest import DAY, Box


TAR_BLOB = f"hsl/hfp-raw/2026/01/05/hfp-raw-{DAY}.tar"
LAYOUT = BlobLayout("hsl/hfp/", "hsl/hfp-raw/")


class CrashError(Exception):
    """Stands in for the process dying in the middle of a run."""


class CrashingStore(LocalBlobStore):
    """A fake container that dies before its n-th upload."""

    def __init__(self, base: Path, crash_before: int) -> None:
        """Set up the store.

        Args:
            base: Directory.
            crash_before: Index of the upload that dies (0-based).
        """
        super().__init__(base)
        self.crash_before, self.uploads = crash_before, 0

    def upload(self, local_path: Path, name: str, sha256: str, md5: bytes, overwrite: bool = False) -> None:
        """Upload, or die on the chosen upload.

        Args:
            local_path: Local file.
            name: Blob name.
            sha256: SHA-256.
            md5: MD5.
            overwrite: Replace an existing blob.

        Raises:
            CrashError: On the chosen upload.
        """
        if self.uploads == self.crash_before:
            raise CrashError(name)
        self.uploads += 1
        super().upload(local_path, name, sha256, md5, overwrite)


@pytest.fixture
def done_box(make_box: Callable[..., Box]) -> Box:
    """Return a box whose day is converted and done.

    Args:
        make_box: Box factory.

    Returns:
        The box.
    """
    box = make_box()
    Pipeline(box.settings, box.remote, box.db).run([DAY])
    assert box.db.require(DAY).state == DayState.DONE
    return box


def replicator(box: Box, store: LocalBlobStore, layout: BlobLayout = LAYOUT) -> Replicator:
    """Return a replicator between the box and a store.

    Args:
        box: The fake storage box.
        store: The fake container.
        layout: Container layout.

    Returns:
        The replicator.
    """
    return Replicator(box.settings, box.remote, store, layout, box.db)


def source_manifest(box: Box) -> dict:
    """Return the day's manifest on the storage box.

    Args:
        box: The fake storage box.

    Returns:
        The manifest.
    """
    return json.loads((box.root / hfp_day(DAY) / MANIFEST_NAME).read_text())


def assert_complete(box: Box, store: LocalBlobStore, layout: BlobLayout = LAYOUT) -> None:
    """Check that every file of the day is in the container with the storage box bytes.

    Args:
        box: The fake storage box.
        store: The fake container.
        layout: Container layout.
    """
    src = box.root / hfp_day(DAY)
    for f in sorted(src.iterdir()):
        blob = store.base / layout.day_dir(DAY) / f.name
        assert blob.read_bytes() == f.read_bytes(), f.name
        if f.name != MANIFEST_NAME:
            info = store.props(f"{layout.day_dir(DAY)}{f.name}")
            assert info is not None
            assert info.sha256 == sha256_of(f)
    if layout.raw_prefix:
        tar = box.root / source_manifest(box)["raw_archive"]["path"]
        assert (store.base / TAR_BLOB).read_bytes() == tar.read_bytes()


def test_copies_a_day_and_is_idempotent(done_box: Box, tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "azure")
    assert replicator(done_box, store).replicate_day(DAY) > 0
    assert_complete(done_box, store)
    assert done_box.db.replicas(f"{store.describe()}/hsl/hfp/")[DAY].state == ReplicaState.DONE
    assert replicator(done_box, store).replicate_day(DAY, verify=True) == 0
    assert not (done_box.settings.work_dir / "_replicate" / DAY).exists()


def test_no_tar_without_raw_prefix(done_box: Box, tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "azure")
    layout = BlobLayout.from_env({"AZURE_HFP_PREFIX": "/data/hfp"})
    replicator(done_box, store, layout).replicate_day(DAY)
    assert_complete(done_box, store, layout)
    assert store.list_names("hsl/") == []
    assert all(n.startswith(f"data/hfp/dt={DAY}/") for n in store.list_names(""))


def test_layout_from_env() -> None:
    assert BlobLayout.from_env({}) == BlobLayout("hsl/hfp/", None)
    assert BlobLayout.from_env({"AZURE_HFP_RAW_PREFIX": "  "}).raw_prefix is None
    layout = BlobLayout.from_env({"AZURE_HFP_PREFIX": "/x/y/", "AZURE_HFP_RAW_PREFIX": "/hsl/hfp-raw/"})
    assert layout.day_dir(DAY) == f"x/y/dt={DAY}/"
    assert layout.tar_name(DAY) == TAR_BLOB
    assert normalize_prefix("/") == ""
    for bad in ("/a/../b/", "a//b"):
        with pytest.raises(ValueError, match="bad blob prefix"):
            normalize_prefix(bad)


@pytest.mark.parametrize("crash_before", range(12))
def test_crash_at_every_upload_resumes(done_box: Box, tmp_path: Path, crash_before: int) -> None:
    store = CrashingStore(tmp_path / "azure", crash_before)
    n_uploads = len(list((done_box.root / hfp_day(DAY)).iterdir())) + 1  # data files + manifest + tar
    if crash_before >= n_uploads:
        pytest.skip(f"the day has only {n_uploads} uploads")
    with pytest.raises(CrashError):
        replicator(done_box, store).replicate_day(DAY)
    assert store.props(f"hsl/hfp/dt={DAY}/{MANIFEST_NAME}") is None, "the manifest must come last"
    assert DAY not in done_box.db.replicas(f"{store.describe()}/hsl/hfp/")
    store.crash_before = -1
    replicator(done_box, store).replicate_day(DAY)
    assert_complete(done_box, store)


def test_existing_blob_with_other_content_is_not_overwritten(done_box: Box, tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "azure")
    first = sorted((done_box.root / hfp_day(DAY)).glob("part-*.parquet"))[0].name
    other = tmp_path / "other"
    other.write_bytes(b"not the parquet file")
    store.upload(other, f"hsl/hfp/dt={DAY}/{first}", "0" * 64, b"")
    summary = replicator(done_box, store).run([DAY])
    assert summary.failed == [DAY]
    assert (store.base / f"hsl/hfp/dt={DAY}/{first}").read_bytes() == b"not the parquet file"
    assert store.props(f"hsl/hfp/dt={DAY}/{MANIFEST_NAME}") is None
    record = done_box.db.replicas(f"{store.describe()}/hsl/hfp/")[DAY]
    assert record.state == ReplicaState.FAILED
    assert "exists" in (record.last_error or "")


def test_damaged_storage_box_file_is_not_copied(done_box: Box, tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "azure")
    part = sorted((done_box.root / hfp_day(DAY)).glob("part-*.parquet"))[-1]
    part.write_bytes(part.read_bytes()[:-1] + b"\0")
    summary = replicator(done_box, store).run([DAY])
    assert summary.failed == [DAY]
    assert store.props(f"hsl/hfp/dt={DAY}/{part.name}") is None
    assert "differs from the manifest" in (done_box.db.replicas(f"{store.describe()}/hsl/hfp/")[DAY].last_error or "")


def test_unfinished_day_is_skipped(done_box: Box, tmp_path: Path) -> None:
    path = done_box.root / hfp_day(DAY) / MANIFEST_NAME
    manifest = source_manifest(done_box)
    manifest["archive_state"]["state"] = "uploaded"
    path.write_text(json.dumps(manifest))
    store = LocalBlobStore(tmp_path / "azure")
    summary = replicator(done_box, store).run([DAY])
    assert DAY in summary.skipped
    assert store.list_names("") == []


def test_stop_between_files(done_box: Box, tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "azure")
    stop = StopFlag()
    stop.requested = True
    summary = Replicator(done_box.settings, done_box.remote, store, LAYOUT, done_box.db, stop).run([DAY])
    assert summary.stopped_reason == "stop requested"
    assert store.list_names("") == []


def test_missing_sentinel_stops(done_box: Box, tmp_path: Path) -> None:
    (done_box.root / SENTINEL).unlink()
    with pytest.raises(SentinelMissingError):
        replicator(done_box, LocalBlobStore(tmp_path / "azure")).run([DAY])


def test_verify_detects_a_changed_blob(done_box: Box, tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "azure")
    replicator(done_box, store).replicate_day(DAY)
    blob = sorted((store.base / f"hsl/hfp/dt={DAY}").glob("part-*.parquet"))[0]
    blob.write_bytes(blob.read_bytes()[:-1] + b"\0")
    with pytest.raises(ReplicaConflictError, match="downloaded SHA-256"):
        replicator(done_box, store).replicate_day(DAY, verify=True)


def test_select_replica_days() -> None:
    days = {"2026-01-01", "2026-01-02", "2026-01-03"}
    assert select_replica_days(days, {"2026-01-01"}) == ["2026-01-02", "2026-01-03"]
    assert select_replica_days(days, set(), last="2026-01-02") == ["2026-01-01", "2026-01-02"]
    assert select_replica_days(days, {"2026-01-01"}, only=["2026-01-01", "2026-02-01"]) == ["2026-01-01"]


def test_cli_replicate_status_and_check(
    done_box: Box, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AZURE_HFP_RAW_PREFIX", "/hsl/hfp-raw/")
    store = LocalBlobStore(tmp_path / "azure")
    common = ["--remote", "local", "--state-dir", str(done_box.settings.state_dir), "--work-dir",
              str(done_box.settings.work_dir)]  # fmt: skip
    for command in ("azure-check", "replicate", "replica-status"):
        with pytest.raises(SystemExit) as exc:
            main([command, *common], remote=done_box.remote, store=store)
        assert exc.value.code == 0, command
    assert_complete(done_box, store)
    assert "1 copied, 0 failed, 0 not yet" in capsys.readouterr().out

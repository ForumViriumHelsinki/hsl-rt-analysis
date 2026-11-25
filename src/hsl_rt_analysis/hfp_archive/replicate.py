"""Copy finished days from the storage box to an Azure Blob container, file by file, verified.

A day is copied only when its storage box manifest says `done` (or `purged`) and every required check
passed. Every file is downloaded here, its SHA-256 compared with the manifest, and uploaded as a new
blob with the SHA-256 in its metadata. The manifest goes last: a day directory in the container without
`_manifest.json` is incomplete, and the next run finishes it. An existing blob is never overwritten,
except the manifest itself.

Layout in the container (environment variables, see docs/hfp-archive-converter.md):

- `AZURE_HFP_PREFIX` (default `/hsl/hfp/`): `<container><prefix>dt=YYYY-MM-DD/part-*.parquet`, ...
- `AZURE_HFP_RAW_PREFIX` (default empty): when set, the raw tar goes to
  `<container><prefix>YYYY/MM/DD/hfp-raw-YYYY-MM-DD.tar`; when empty, no tar is copied.
"""

import hashlib
import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hsl_rt_analysis.hfp_archive.blobstore import BlobExistsError, BlobStore
from hsl_rt_analysis.hfp_archive.config import Settings, hfp_day, raw_tar_name
from hsl_rt_analysis.hfp_archive.guards import StopFlag
from hsl_rt_analysis.hfp_archive.manifest import MANIFEST_NAME
from hsl_rt_analysis.hfp_archive.remote import Remote, local_free_bytes
from hsl_rt_analysis.hfp_archive.state import ReplicaState, StateDB


logger = logging.getLogger(__name__)

DEFAULT_DATA_PREFIX = "/hsl/hfp/"
FINISHED_STATES = {"done", "purged"}
MAX_FAILURES_IN_A_ROW = 3


class ReplicaConflictError(Exception):
    """A blob exists with other content than the storage box file: a human must look."""


class SourceDamagedError(Exception):
    """A file downloaded from the storage box does not match its manifest."""


class NotFinishedError(Exception):
    """The day is not finished on the storage box (or failed a check), so it is not copied."""


class ReplicaStoppedError(Exception):
    """A stop was requested between two files."""


def normalize_prefix(value: str) -> str:
    """Turn a prefix such as `/hsl/hfp/` into a blob name prefix (`hsl/hfp/`).

    Args:
        value: Prefix from the environment; leading and trailing slashes are optional.

    Returns:
        The prefix without a leading slash and with a trailing one, or "" for the container root.

    Raises:
        ValueError: If it holds `..` or empty path segments.
    """
    parts = value.strip().strip("/").split("/")
    if parts == [""]:
        return ""
    if any(p in {"", ".", ".."} for p in parts):
        msg = f"bad blob prefix: {value!r}"
        raise ValueError(msg)
    return "/".join(parts) + "/"


@dataclass(frozen=True)
class BlobLayout:
    """Where a day's files go in the container."""

    data_prefix: str = "hsl/hfp/"
    raw_prefix: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> BlobLayout:
        """Read the layout from AZURE_HFP_PREFIX and AZURE_HFP_RAW_PREFIX.

        Args:
            env: Environment (os.environ).

        Returns:
            The layout.
        """
        data = env.get("AZURE_HFP_PREFIX") or DEFAULT_DATA_PREFIX
        raw = env.get("AZURE_HFP_RAW_PREFIX", "").strip()
        return cls(normalize_prefix(data), normalize_prefix(raw) if raw else None)

    def day_dir(self, day: str) -> str:
        """Return the blob prefix of a day's Parquet directory.

        Args:
            day: The day.

        Returns:
            Prefix ending with '/'.
        """
        return f"{self.data_prefix}dt={day}/"

    def tar_name(self, day: str) -> str | None:
        """Return the blob name of a day's raw tar, or None if tars are not copied.

        Args:
            day: The day.

        Returns:
            Blob name or None.
        """
        if self.raw_prefix is None:
            return None
        y, m, d = day.split("-")
        return f"{self.raw_prefix}{y}/{m}/{d}/{raw_tar_name(day)}"


@dataclass(frozen=True)
class Item:
    """One file to copy."""

    source: str  # path relative to the storage box archive root
    blob: str
    size: int
    sha256: str


@dataclass
class ReplicaSummary:
    """What a replication run did."""

    done: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    uploaded_bytes: int = 0
    stopped_reason: str | None = None

    @property
    def exit_code(self) -> int:
        """Return 1 if any day failed or the run stopped early, else 0."""
        return 1 if self.failed or self.stopped_reason else 0


def digests(path: Path) -> tuple[str, bytes]:
    """Return the hex SHA-256 and the raw MD5 of a file in one pass.

    Args:
        path: File.

    Returns:
        (sha256 hex, md5 bytes).
    """
    sha, md5 = hashlib.sha256(), hashlib.md5()  # MD5 only for Azure Content-MD5
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.digest()


def day_items(day: str, manifest: dict[str, Any], layout: BlobLayout) -> list[Item]:
    """List the files of a day to copy, data files first (the manifest is handled separately).

    Args:
        day: The day.
        manifest: The day's storage box manifest.
        layout: The container layout.

    Returns:
        Items.
    """
    entries = [*manifest["parquet"], manifest.get("repeats"), manifest.get("rejects")]
    items = [
        Item(f"{hfp_day(day)}/{e['path']}", f"{layout.day_dir(day)}{e['path']}", e["size"], e["sha256"])
        for e in entries
        if e
    ]
    tar, tar_blob = manifest.get("raw_archive"), layout.tar_name(day)
    if tar and tar_blob:
        items.append(Item(tar["path"], tar_blob, tar["size"], tar["sha256"]))
    elif tar_blob:
        logger.info("%s keeps no raw tar: nothing to copy to %s", day, tar_blob)
    return items


def check_finished(day: str, manifest: dict[str, Any]) -> None:
    """Refuse a day that is not finished on the storage box.

    Args:
        day: The day.
        manifest: Its storage box manifest.

    Raises:
        NotFinishedError: If its state is not done/purged or a required check failed.
    """
    state = manifest.get("archive_state", {}).get("state")
    if state not in FINISHED_STATES:
        msg = f"{day} is {state} on the storage box, not done"
        raise NotFinishedError(msg)
    if manifest.get("all_checks_ok") is not True or manifest.get("dt") != day:
        msg = f"{day}: the manifest does not report all checks passed"
        raise NotFinishedError(msg)


class Replicator:
    """Copies days from the storage box to one blob container."""

    def __init__(
        self,
        settings: Settings,
        remote: Remote,
        store: BlobStore,
        layout: BlobLayout,
        db: StateDB,
        stop: StopFlag | None = None,
    ) -> None:
        """Set up a replication run.

        Args:
            settings: Run settings (work dir, local space).
            remote: The storage box.
            store: The blob container.
            layout: Where the files go in the container.
            db: State database.
            stop: Stop flag checked between files.
        """
        self.settings, self.remote, self.store, self.layout, self.db = settings, remote, store, layout, db
        self.stop = stop
        self.target = f"{store.describe()}/{layout.data_prefix}"
        self.uploaded_bytes = 0

    def _scratch(self, day: str) -> Path:
        """Return (and create) the local scratch directory of a day's copy.

        Args:
            day: The day.

        Returns:
            Directory.
        """
        path = self.settings.work_dir / "_replicate" / day
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _check_stop(self) -> None:
        """Raise if a stop was requested.

        Raises:
            ReplicaStoppedError: If SIGINT / SIGTERM arrived.
        """
        if self.stop is not None and self.stop.requested:
            msg = "stop requested"
            raise ReplicaStoppedError(msg)

    def _fetch(self, source: str, local: Path, size: int, sha256: str) -> bytes:
        """Download one storage box file and confirm it against the manifest.

        Args:
            source: Path relative to the archive root.
            local: Local destination.
            size: Expected size.
            sha256: Expected SHA-256.

        Returns:
            The file's MD5.

        Raises:
            OSError: If the local disk is too full.
            SourceDamagedError: If the downloaded bytes differ from the manifest.
        """
        free = local_free_bytes(local.parent)
        if free < size + self.settings.min_local_free_gb * 1e9:
            msg = f"local disk too full for {source}: {free / 1e9:.1f} GB free"
            raise OSError(msg)
        self.remote.download_file(source, local)
        got, md5 = digests(local)
        if got != sha256 or local.stat().st_size != size:
            msg = f"{source}: downloaded SHA-256 {got} differs from the manifest {sha256}"
            raise SourceDamagedError(msg)
        return md5

    def _copy_item(self, item: Item, scratch: Path) -> bool:
        """Make sure one file is in the container with the right content.

        Args:
            item: The file.
            scratch: Local scratch directory.

        Returns:
            True if it was uploaded now, False if it was already there.

        Raises:
            ReplicaConflictError: If a blob with other content is in the way.
        """
        info = self.store.props(item.blob)
        if info is not None:
            if info.size == item.size and info.sha256 == item.sha256:
                return False
            msg = f"{item.blob} exists with size {info.size} and SHA-256 {info.sha256}, expected {item.sha256}"
            raise ReplicaConflictError(msg)
        local = scratch / Path(item.blob).name
        try:
            md5 = self._fetch(item.source, local, item.size, item.sha256)
            self._check_stop()
            try:
                self.store.upload(local, item.blob, item.sha256, md5)
            except BlobExistsError:
                # Another writer committed it between props() and upload(); check what it wrote.
                return self._copy_item(item, scratch)
        finally:
            local.unlink(missing_ok=True)
        after = self.store.props(item.blob)
        if after is None or after.size != item.size or after.sha256 != item.sha256:
            msg = f"{item.blob}: read-back after upload shows {after}, expected size {item.size}"
            raise ReplicaConflictError(msg)
        self.uploaded_bytes += item.size
        return True

    def _verify_item(self, item: Item, scratch: Path) -> None:
        """Download a blob back and compare its SHA-256 (costs Azure egress, so only on request).

        Args:
            item: The file.
            scratch: Local scratch directory.

        Raises:
            ReplicaConflictError: If the downloaded bytes differ.
        """
        local = scratch / f"verify-{Path(item.blob).name}"
        try:
            self.store.download(item.blob, local)
            got, _ = digests(local)
        finally:
            local.unlink(missing_ok=True)
        if got != item.sha256:
            msg = f"{item.blob}: downloaded SHA-256 {got} differs from {item.sha256}"
            raise ReplicaConflictError(msg)

    def _put_manifest(self, day: str, scratch: Path) -> dict[str, Any]:
        """Copy the storage box manifest last, which marks the day complete in the container.

        Args:
            day: The day.
            scratch: Local scratch directory.

        Returns:
            The manifest.
        """
        local = scratch / MANIFEST_NAME
        self.remote.download_file(f"{hfp_day(day)}/{MANIFEST_NAME}", local)
        manifest = json.loads(local.read_text())
        sha, md5 = digests(local)
        self.store.upload(local, f"{self.layout.day_dir(day)}{MANIFEST_NAME}", sha, md5, overwrite=True)
        return manifest

    def replicate_day(self, day: str, verify: bool = False) -> int:
        """Copy one day and record the outcome.

        Args:
            day: The day.
            verify: Also download every blob back and compare its SHA-256.

        Returns:
            Number of files uploaded now.
        """
        scratch = self._scratch(day)
        try:
            self.remote.check_sentinel()
            local = scratch / f"source-{MANIFEST_NAME}"
            self.remote.download_file(f"{hfp_day(day)}/{MANIFEST_NAME}", local)
            manifest = json.loads(local.read_text())
            check_finished(day, manifest)
            items = day_items(day, manifest, self.layout)
            uploaded = 0
            for item in items:
                self._check_stop()
                uploaded += self._copy_item(item, scratch)
            if verify:
                for item in items:
                    self._check_stop()
                    self._verify_item(item, scratch)
            self._put_manifest(day, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        size = sum(i.size for i in items)
        self.db.set_replica(day, self.target, ReplicaState.DONE, files=len(items), bytes=size, verified=verify)
        logger.info("%s: copied to %s (%d files, %d new, %.2f GB)", day, self.target, len(items), uploaded, size / 1e9)
        return uploaded

    def run(self, days: list[str], verify: bool = False) -> ReplicaSummary:
        """Copy days in order, stopping on a stop request or several failures in a row.

        Args:
            days: Days to copy, oldest first.
            verify: Download every blob back.

        Returns:
            The summary.
        """
        summary, in_a_row = ReplicaSummary(), 0
        for day in days:
            try:
                self.replicate_day(day, verify)
            except ReplicaStoppedError:
                summary.stopped_reason = "stop requested"
                break
            except NotFinishedError as exc:
                summary.skipped[day] = str(exc)
                logger.info("Skipping %s", exc)
                continue
            except (ReplicaConflictError, SourceDamagedError, OSError) as exc:
                logger.error("%s: %s", day, exc)
                self.db.set_replica(day, self.target, ReplicaState.FAILED, str(exc))
                summary.failed.append(day)
                in_a_row += 1
                if in_a_row >= MAX_FAILURES_IN_A_ROW:
                    summary.stopped_reason = f"{in_a_row} days failed in a row"
                    break
                continue
            in_a_row = 0
            summary.done.append(day)
            self._check_stop_between_days(summary)
            if summary.stopped_reason:
                break
        summary.uploaded_bytes = self.uploaded_bytes
        return summary

    def _check_stop_between_days(self, summary: ReplicaSummary) -> None:
        """Record a stop request that arrived during the last day.

        Args:
            summary: The run summary to update.
        """
        if self.stop is not None and self.stop.requested:
            summary.stopped_reason = "stop requested"


def select_replica_days(
    hfp_days: set[str],
    done: set[str],
    first: str | None = None,
    last: str | None = None,
    only: list[str] | None = None,
) -> list[str]:
    """Return the days to copy: converted on the storage box, in range, not yet copied.

    Failed days are tried again; an explicitly listed day is copied (re-checked) even if done.

    Args:
        hfp_days: Days with a dt= directory on the storage box.
        done: Days already copied to this target.
        first: First day of the range.
        last: Last day of the range.
        only: Explicit days.

    Returns:
        Days, oldest first.
    """
    if only is not None:
        return sorted(d for d in only if d in hfp_days)
    return sorted(d for d in hfp_days - done if not (first and d < first) and not (last and d > last))

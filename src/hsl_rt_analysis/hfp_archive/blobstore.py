"""Access to the Azure Blob container that receives a copy of the archive: the SDK, or a directory in tests.

Blob storage has no rename, but a block blob only appears when its block list is committed, so an upload
is atomic: a blob either exists with all its bytes or not at all. Every data blob is uploaded with
`overwrite=False` and carries its SHA-256 in the blob metadata.
"""

import base64
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.storage.blob import ContainerClient, ContentSettings, StandardBlobTier


logger = logging.getLogger(__name__)

HTTP_FORBIDDEN = 403
MAX_BACKOFF_S = 900.0


class BlobUnavailableError(Exception):
    """Azure cannot be reached (network, DNS, sleep). Transient: waiting may help."""


class BlobAuthError(Exception):
    """Azure refused the request: the SAS token is wrong, lacks a permission or has expired. Never retried."""


class BlobExistsError(Exception):
    """A blob that must be new already exists."""


@dataclass(frozen=True)
class BlobInfo:
    """What the converter needs to know about an existing blob."""

    size: int
    sha256: str | None


class BlobStore(Protocol):
    """The operations the replication needs on the container. Names are blob names (no leading slash)."""

    def describe(self) -> str:
        """Return a printable name of the container, without any secret."""

    def props(self, name: str) -> BlobInfo | None:
        """Return the size and SHA-256 metadata of a blob, or None if it does not exist."""

    def upload(self, local_path: Path, name: str, sha256: str, md5: bytes, overwrite: bool = False) -> None:
        """Upload a file to the Cool tier; raise BlobExistsError if it exists and overwrite is False."""

    def download(self, name: str, local_path: Path) -> None:
        """Download a blob to a local file."""

    def list_names(self, prefix: str) -> list[str]:
        """Return the names of the blobs that start with a prefix."""


def describe_sas_url(sas_url: str) -> str:
    """Return the account host and container of a SAS URL, dropping the token.

    Args:
        sas_url: Container URL with a SAS query string.

    Returns:
        `host/container`, safe to log.
    """
    parts = urlsplit(sas_url)
    return f"{parts.netloc}{parts.path}"


class AzureBlobStore:
    """A container reached with a container SAS URL (needs the read, create, write and list permissions)."""

    def __init__(self, sas_url: str, max_wait_s: float = 6 * 3600.0, concurrency: int = 4) -> None:
        """Connect to the container (no request is made yet).

        Args:
            sas_url: Container URL with a SAS token. It is never logged.
            max_wait_s: How long a lost connection is waited out before giving up.
            concurrency: Parallel block transfers per blob.
        """
        self._container = ContainerClient.from_container_url(sas_url)
        self._name = describe_sas_url(sas_url)
        self._max_wait_s = max_wait_s
        self._concurrency = concurrency

    def describe(self) -> str:
        """Return the account host and container.

        Returns:
            Printable name.
        """
        return self._name

    def _retrying[T](self, call: Callable[[], T], what: str) -> T:
        """Run one SDK call, waiting out lost connections with a growing pause.

        The SDK already retries a few times on its own; this outer loop covers longer outages such as a
        sleeping laptop or a network break.

        Args:
            call: The call.
            what: Description for the log.

        Returns:
            The call's result.

        Raises:
            BlobUnavailableError: If Azure stays unreachable for longer than max_wait_s.
            BlobAuthError: If Azure refuses the SAS token.
        """
        start, pause = time.monotonic(), 30.0
        while True:
            try:
                return call()
            except ClientAuthenticationError as exc:
                raise BlobAuthError(f"{what}: {exc.message}") from exc
            except (ServiceRequestError, ServiceResponseError) as exc:
                if time.monotonic() - start > self._max_wait_s:
                    raise BlobUnavailableError(f"{what}: {exc}") from exc
                logger.warning("Azure unreachable (%s): retrying in %.0f s", what, pause)
                time.sleep(pause)
                pause = min(pause * 2, MAX_BACKOFF_S)
            except HttpResponseError as exc:
                if exc.status_code == HTTP_FORBIDDEN:
                    raise BlobAuthError(f"{what}: {exc.reason}") from exc
                raise

    def props(self, name: str) -> BlobInfo | None:
        """Return a blob's size and SHA-256 metadata.

        Args:
            name: Blob name.

        Returns:
            The info, or None if the blob does not exist.
        """

        def call() -> BlobInfo | None:
            try:
                p = self._container.get_blob_client(name).get_blob_properties()
            except ResourceNotFoundError:
                return None
            return BlobInfo(p.size, (p.metadata or {}).get("sha256"))

        return self._retrying(call, f"properties of {name}")

    def upload(self, local_path: Path, name: str, sha256: str, md5: bytes, overwrite: bool = False) -> None:
        """Upload a file to the Cool tier, checking the MD5 of every block on the way.

        Args:
            local_path: Local file.
            name: Blob name.
            sha256: SHA-256 of the file, stored in the blob metadata.
            md5: MD5 of the whole file, stored as Content-MD5 for later readers.
            overwrite: Replace an existing blob (only for the manifest).

        Raises:
            BlobExistsError: If the blob exists and overwrite is False.
        """

        def call() -> None:
            with local_path.open("rb") as fh:
                self._container.get_blob_client(name).upload_blob(
                    fh,
                    overwrite=overwrite,
                    metadata={"sha256": sha256},
                    standard_blob_tier=StandardBlobTier.COOL,
                    content_settings=ContentSettings(content_md5=bytearray(md5)),
                    validate_content=True,
                    max_concurrency=self._concurrency,
                )

        try:
            self._retrying(call, f"upload of {name}")
        except ResourceExistsError as exc:
            raise BlobExistsError(name) from exc

    def download(self, name: str, local_path: Path) -> None:
        """Download a blob.

        Args:
            name: Blob name.
            local_path: Destination file (overwritten).
        """

        def call() -> None:
            with local_path.open("wb") as fh:
                self._container.get_blob_client(name).download_blob(max_concurrency=self._concurrency).readinto(fh)

        self._retrying(call, f"download of {name}")

    def list_names(self, prefix: str) -> list[str]:
        """List blob names under a prefix.

        Args:
            prefix: Name prefix.

        Returns:
            Names, sorted.
        """
        return self._retrying(
            lambda: sorted(b.name for b in self._container.list_blobs(name_starts_with=prefix)), f"list {prefix}"
        )


class LocalBlobStore:
    """A directory standing in for a container in tests; metadata lives in a sidecar directory."""

    META = ".blobmeta"

    def __init__(self, base: Path) -> None:
        """Use a directory as the container.

        Args:
            base: Directory (created).
        """
        self.base = base
        base.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        """Return the directory.

        Returns:
            Printable name.
        """
        return str(self.base)

    def _meta(self, name: str) -> Path:
        """Return the sidecar file of a blob.

        Args:
            name: Blob name.

        Returns:
            Path.
        """
        return self.base / self.META / f"{name}.json"

    def props(self, name: str) -> BlobInfo | None:
        """Return a blob's size and SHA-256 metadata.

        Args:
            name: Blob name.

        Returns:
            The info, or None if the blob does not exist.
        """
        path = self.base / name
        if not path.is_file():
            return None
        meta = json.loads(self._meta(name).read_text()) if self._meta(name).exists() else {}
        return BlobInfo(path.stat().st_size, meta.get("sha256"))

    def upload(self, local_path: Path, name: str, sha256: str, md5: bytes, overwrite: bool = False) -> None:
        """Copy a file in atomically, like a committed block list.

        Args:
            local_path: Local file.
            name: Blob name.
            sha256: SHA-256 stored in the metadata.
            md5: MD5 stored in the metadata.
            overwrite: Replace an existing blob.

        Raises:
            BlobExistsError: If the blob exists and overwrite is False.
        """
        dst = self.base / name
        if dst.exists() and not overwrite:
            raise BlobExistsError(name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.uploading")
        tmp.write_bytes(local_path.read_bytes())
        self._meta(name).parent.mkdir(parents=True, exist_ok=True)
        self._meta(name).write_text(json.dumps({"sha256": sha256, "md5": base64.b64encode(md5).decode()}))
        tmp.replace(dst)

    def download(self, name: str, local_path: Path) -> None:
        """Copy a blob out.

        Args:
            name: Blob name.
            local_path: Destination file.

        Raises:
            FileNotFoundError: If the blob does not exist.
        """
        src = self.base / name
        if not src.is_file():
            raise FileNotFoundError(name)
        local_path.write_bytes(src.read_bytes())

    def list_names(self, prefix: str) -> list[str]:
        """List blob names under a prefix.

        Args:
            prefix: Name prefix.

        Returns:
            Names, sorted.
        """
        names = (str(p.relative_to(self.base)) for p in self.base.rglob("*") if p.is_file())
        return sorted(n for n in names if n.startswith(prefix) and not n.startswith(self.META))

"""Access to the archive root on the storage box: SSH + rsync for real runs, a local directory for tests.

The storage box offers only a restricted shell (no pipes or redirects), so every operation is one
command. Checksums are computed on the server (`sha256sum`), which makes check C6 a true read-back:
nothing on this machine can answer for the server.
"""

import hashlib
import logging
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hsl_rt_analysis.hfp_archive.config import SENTINEL


logger = logging.getLogger(__name__)

SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_.=/-]+$")
SSH_UNREACHABLE = 255


class RemoteUnavailableError(Exception):
    """The storage box cannot be reached (network, sleep, DNS). Transient: waiting may help."""


class SentinelMissingError(Exception):
    """The storage box answers but the sentinel file is missing: wrong root or wrong account. Never retried."""


class RemoteCommandError(Exception):
    """A command on the storage box failed although the box is reachable."""


@dataclass(frozen=True)
class RemoteEntry:
    """One directory entry on the remote side."""

    name: str
    size: int
    mtime: int
    is_dir: bool


class Remote(Protocol):
    """The operations the converter needs on the archive root. Paths are relative to the root."""

    def check_sentinel(self) -> None:
        """Raise unless the archive root is reachable and holds the sentinel file."""

    def exists(self, rel: str) -> bool:
        """Return whether a file or directory exists."""

    def list_entries(self, rel: str) -> list[RemoteEntry]:
        """Return the entries of a directory, without . and .."""

    def sha256_dir(self, rel: str) -> dict[str, str]:
        """Return file name mapped to SHA-256 for every file directly in a directory."""

    def sha256_file(self, rel: str) -> str:
        """Return the SHA-256 of one file."""

    def read_range(self, rel: str, offset: int, length: int) -> bytes:
        """Return `length` bytes of a file starting at `offset`."""

    def download_dir(self, rel: str, local_dir: Path, include: str | None = None) -> None:
        """Copy the files of a directory (optionally only names matching a glob) into a local directory."""

    def download_file(self, rel: str, local_path: Path) -> None:
        """Copy one file to a local path."""

    def upload_file(self, local_path: Path, rel: str) -> None:
        """Copy a local file to a remote path whose parent directory exists."""

    def mkdir_p(self, rel: str) -> None:
        """Create a directory and its parents."""

    def move(self, src: str, dst: str) -> None:
        """Rename a file or directory; the destination must not exist."""

    def replace_file(self, src: str, dst: str) -> None:
        """Rename a file over an existing one (only for the converter's own manifest)."""

    def remove_tree(self, rel: str) -> None:
        """Delete a file or directory tree (only the purge command uses this)."""

    def free_bytes(self) -> int:
        """Return the free space of the storage box in bytes."""


def check_safe(rel: str) -> str:
    """Refuse a remote path that would need shell quoting, so globs and moves stay predictable.

    Args:
        rel: Path relative to the archive root.

    Returns:
        The same path.

    Raises:
        ValueError: If the path holds characters outside the safe set.
    """
    if not SAFE_PATH_RE.match(rel) or ".." in rel.split("/"):
        msg = f"unsafe remote path: {rel!r}"
        raise ValueError(msg)
    return rel


def parse_sha256_output(text: str, prefix: str) -> dict[str, str]:
    """Parse `sha256sum` output into file name mapped to digest.

    Args:
        text: Command output, one `<digest>  <path>` line per file (names may contain spaces).
        prefix: Directory part to strip from every path, ending with '/'.

    Returns:
        File name mapped to hex digest.

    Raises:
        RemoteCommandError: If a line does not have the expected form.
    """
    result = {}
    for line in text.splitlines():
        if not line:
            continue
        digest, sep, path = line.partition("  ")
        if not sep or len(digest) != 64 or not path.startswith(prefix):
            msg = f"unexpected sha256sum line: {line!r}"
            raise RemoteCommandError(msg)
        result[path.removeprefix(prefix)] = digest
    return result


def parse_ls_output(text: str) -> list[RemoteEntry]:
    """Parse `ls -la --time-style=+%s` output.

    Args:
        text: Command output.

    Returns:
        Directory entries without . and ..
    """
    entries = []
    for line in text.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7 or line.startswith("total "):
            continue
        name = parts[6]
        if name in {".", ".."}:
            continue
        entries.append(RemoteEntry(name=name, size=int(parts[4]), mtime=int(parts[5]), is_dir=parts[0][0] == "d"))
    return entries


class SshRemote:
    """The storage box over SSH (restricted shell) and rsync."""

    def __init__(self, target: str, root: str, port: int = 23, max_wait_s: float = 6 * 3600.0) -> None:
        """Create the remote.

        Args:
            target: SSH destination, user@host.
            root: Archive root directory relative to the SSH home.
            port: SSH port (23 is the storage box's port with the extended service set).
            max_wait_s: How long to keep waiting for an unreachable storage box before giving up.
        """
        self.target = target
        self.root = check_safe(root)
        self.port = port
        self.max_wait_s = max_wait_s

    # --- plumbing ---------------------------------------------------------------------------------

    def _ssh_base(self) -> list[str]:
        """Return the ssh command prefix with keep-alives that notice a dead link within a minute.

        Returns:
            Argument list.
        """
        return [
            "ssh", "-p", str(self.port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
        ]  # fmt: skip

    def _path(self, rel: str) -> str:
        """Return the remote path of a root-relative path.

        Args:
            rel: Path relative to the archive root.

        Returns:
            Path relative to the SSH home.
        """
        return f"{self.root}/{rel}" if rel else self.root

    def _exec(self, args: list[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
        """Run a local command in its own session, so a Ctrl-C meant for the converter does not kill it.

        Args:
            args: Command and arguments.
            timeout: Seconds before the command is abandoned.

        Returns:
            The finished process.
        """
        try:
            return subprocess.run(args, capture_output=True, timeout=timeout, start_new_session=True, check=False)
        except subprocess.TimeoutExpired:
            logger.warning("Timed out after %.0f s: %s", timeout, " ".join(args[:8]))
            return subprocess.CompletedProcess(args, SSH_UNREACHABLE, b"", b"timeout")

    def _reachable(self) -> bool:
        """Return whether the storage box answers a trivial command.

        Returns:
            True if `pwd` succeeds.
        """
        return self._exec([*self._ssh_base(), self.target, "pwd"], timeout=60).returncode == 0

    def _wait_until_reachable(self) -> None:
        """Block with growing pauses until the storage box answers, or give up after max_wait_s.

        Raises:
            RemoteUnavailableError: If it stays unreachable for max_wait_s.
        """
        deadline = time.monotonic() + self.max_wait_s
        pause = 10.0
        while not self._reachable():
            if time.monotonic() > deadline:
                msg = f"storage box unreachable for {self.max_wait_s:.0f} s"
                raise RemoteUnavailableError(msg)
            logger.warning("Storage box unreachable, retrying in %.0f s", pause)
            time.sleep(pause)
            pause = min(pause * 2, 300.0)

    def _retrying(self, run: Callable[[], subprocess.CompletedProcess[bytes]], what: str) -> bytes:
        """Run an idempotent operation, waiting out connection losses, and return its stdout.

        Args:
            run: Runs the operation once.
            what: Description for logs and errors.

        Returns:
            Standard output of the successful run.

        Raises:
            RemoteCommandError: If it fails three times while the box is reachable.
        """
        failures = 0
        while True:
            proc = run()
            if proc.returncode == 0:
                return proc.stdout
            if not self._reachable():
                self._wait_until_reachable()
                continue
            failures += 1
            if failures >= 3:
                msg = f"{what} failed (exit {proc.returncode}): {proc.stderr.decode(errors='replace').strip()}"
                raise RemoteCommandError(msg)
            time.sleep(5)

    def _cmd(self, command: str, timeout: float = 600) -> bytes:
        """Run one restricted-shell command with retries.

        Args:
            command: The command line (quoted by the caller).
            timeout: Seconds per attempt.

        Returns:
            Standard output.
        """
        return self._retrying(lambda: self._exec([*self._ssh_base(), self.target, command], timeout), command)

    def _probe(self, command: str) -> int:
        """Run a command whose non-zero exit is an answer, not an error (e.g. ls of a missing path).

        Args:
            command: The command line.

        Returns:
            Exit code; connection losses are waited out first.
        """
        while True:
            proc = self._exec([*self._ssh_base(), self.target, command], timeout=120)
            if proc.returncode != SSH_UNREACHABLE:
                return proc.returncode
            self._wait_until_reachable()

    def _rsync(self, src: str, dst: str, extra: list[str], timeout: float) -> None:
        """Run rsync over the same SSH settings, with retries; rsync resumes where it stopped.

        Args:
            src: Source (local path or target:path).
            dst: Destination.
            extra: Extra rsync options.
            timeout: Seconds per attempt.
        """
        ssh = " ".join(shlex.quote(a) for a in self._ssh_base())
        args = ["rsync", "-a", "--partial", "-e", ssh, *extra, src, dst]
        self._retrying(lambda: self._exec(args, timeout), f"rsync {src} {dst}")

    # --- operations ---------------------------------------------------------------------------------

    def check_sentinel(self) -> None:
        """Raise unless the archive root is reachable and holds the sentinel file.

        Raises:
            RemoteUnavailableError: If the box stays unreachable.
            SentinelMissingError: If the box answers but the sentinel is missing.
        """
        if not self._reachable():
            self._wait_until_reachable()
        if self._probe(f"ls -d {self._path(SENTINEL)}") != 0:
            msg = f"sentinel {self._path(SENTINEL)} missing on {self.target}"
            raise SentinelMissingError(msg)

    def exists(self, rel: str) -> bool:
        """Return whether a file or directory exists.

        Args:
            rel: Path relative to the archive root.

        Returns:
            True if it exists.
        """
        return self._probe(f"ls -d {shlex.quote(self._path(rel))}") == 0

    def list_entries(self, rel: str) -> list[RemoteEntry]:
        """Return the entries of a directory.

        Args:
            rel: Directory relative to the archive root.

        Returns:
            Entries without . and ..
        """
        out = self._cmd(f"ls -la --time-style=+%s {shlex.quote(self._path(check_safe(rel)))}")
        return parse_ls_output(out.decode())

    def sha256_dir(self, rel: str) -> dict[str, str]:
        """Return file name mapped to SHA-256 for every file in a directory, computed on the server.

        Args:
            rel: Directory relative to the archive root.

        Returns:
            File name mapped to hex digest; empty for an empty directory.
        """
        path = self._path(check_safe(rel))
        if not [e for e in self.list_entries(rel) if not e.is_dir]:
            return {}
        out = self._cmd(f"sha256sum {path}/*", timeout=3600)
        return parse_sha256_output(out.decode(), f"{path}/")

    def sha256_file(self, rel: str) -> str:
        """Return the SHA-256 of one file, computed on the server.

        Args:
            rel: File relative to the archive root.

        Returns:
            Hex digest.
        """
        path = self._path(rel)
        out = self._cmd(f"sha256sum {shlex.quote(path)}", timeout=3600).decode()
        return parse_sha256_output(out, "")[path]

    def read_range(self, rel: str, offset: int, length: int) -> bytes:
        """Return a byte range of a file, read with dd on the server.

        Args:
            rel: File relative to the archive root.
            offset: Start offset.
            length: Number of bytes.

        Returns:
            The bytes.

        Raises:
            RemoteCommandError: If fewer bytes come back.
        """
        path = shlex.quote(self._path(rel))
        data = self._cmd(f"dd if={path} iflag=skip_bytes,count_bytes skip={offset} count={length} bs=64k status=none")
        if len(data) != length:
            msg = f"read {len(data)} of {length} bytes from {rel}"
            raise RemoteCommandError(msg)
        return data

    def download_dir(self, rel: str, local_dir: Path, include: str | None = None) -> None:
        """Copy the files of a directory into a local directory, keeping modification times.

        Args:
            rel: Directory relative to the archive root.
            local_dir: Local destination directory (created).
            include: Optional glob; only matching names are copied.
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        extra = ["--include", include, "--exclude", "*"] if include else []
        self._rsync(f"{self.target}:{self._path(check_safe(rel))}/", f"{local_dir}/", extra, timeout=4 * 3600)

    def download_file(self, rel: str, local_path: Path) -> None:
        """Copy one remote file to a local path.

        Args:
            rel: File relative to the archive root.
            local_path: Local destination.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._rsync(f"{self.target}:{self._path(check_safe(rel))}", str(local_path), [], timeout=3600)

    def upload_file(self, local_path: Path, rel: str) -> None:
        """Copy a local file to a remote path; rsync writes a hidden temporary and renames it.

        Args:
            local_path: Local source file.
            rel: Destination relative to the archive root; its directory must exist.
        """
        self._rsync(str(local_path), f"{self.target}:{self._path(check_safe(rel))}", [], timeout=4 * 3600)

    def mkdir_p(self, rel: str) -> None:
        """Create a directory and its parents.

        Args:
            rel: Directory relative to the archive root.
        """
        self._cmd(f"mkdir -p {self._path(check_safe(rel))}")

    def move(self, src: str, dst: str) -> None:
        """Rename a file or directory on the server; refuses to move onto an existing path.

        Args:
            src: Source relative to the archive root.
            dst: Destination relative to the archive root.

        Raises:
            RemoteCommandError: If the destination exists or the move did not take effect.
        """
        if self.exists(dst):
            msg = f"refusing to move {src} onto existing {dst}"
            raise RemoteCommandError(msg)
        self._cmd(f"mv {self._path(check_safe(src))} {self._path(check_safe(dst))}")
        if not self.exists(dst) or self.exists(src):
            msg = f"move {src} -> {dst} did not take effect"
            raise RemoteCommandError(msg)

    def replace_file(self, src: str, dst: str) -> None:
        """Rename a file over an existing one (only for the converter's own manifest).

        Args:
            src: Source relative to the archive root.
            dst: Destination relative to the archive root.
        """
        self._cmd(f"mv -f {self._path(check_safe(src))} {self._path(check_safe(dst))}")

    def remove_tree(self, rel: str) -> None:
        """Delete a file or directory tree.

        Args:
            rel: Path relative to the archive root; must be under _superseded/.

        Raises:
            ValueError: If the path is outside _superseded/.
        """
        if not check_safe(rel).startswith("_superseded/"):
            msg = f"refusing to delete outside _superseded/: {rel}"
            raise ValueError(msg)
        self._cmd(f"rm -r {self._path(rel)}", timeout=3600)

    def free_bytes(self) -> int:
        """Return the free space of the storage box.

        Returns:
            Available bytes.
        """
        lines = self._cmd("df -B1").decode().splitlines()
        return int(lines[-1].split()[3])


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a local file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class LocalRemote:
    """A directory standing in for the storage box: used by the tests and for local trial runs."""

    def __init__(self, base: Path, root: str = "2025_HSL_RT", free: int = 10**15) -> None:
        """Create the remote.

        Args:
            base: Directory standing in for the SSH home.
            root: Archive root below it.
            free: Free space to report.
        """
        self.base = base
        self.root = root
        self.free = free

    def _p(self, rel: str) -> Path:
        """Return the local path of a root-relative path.

        Args:
            rel: Path relative to the archive root.

        Returns:
            Local path.
        """
        return self.base / self.root / rel if rel else self.base / self.root

    def check_sentinel(self) -> None:
        """Raise unless the sentinel exists.

        Raises:
            SentinelMissingError: If the sentinel is missing.
        """
        if not self._p(SENTINEL).is_file():
            msg = f"sentinel {self._p(SENTINEL)} missing"
            raise SentinelMissingError(msg)

    def exists(self, rel: str) -> bool:
        """Return whether a path exists.

        Args:
            rel: Path relative to the archive root.

        Returns:
            True if it exists.
        """
        return self._p(rel).exists()

    def list_entries(self, rel: str) -> list[RemoteEntry]:
        """Return the entries of a directory.

        Args:
            rel: Directory relative to the archive root.

        Returns:
            Entries.

        Raises:
            RemoteCommandError: If the directory does not exist.
        """
        path = self._p(rel)
        if not path.is_dir():
            msg = f"no such directory: {rel}"
            raise RemoteCommandError(msg)
        return [
            RemoteEntry(e.name, e.stat().st_size, int(e.stat().st_mtime), e.is_dir()) for e in sorted(path.iterdir())
        ]

    def sha256_dir(self, rel: str) -> dict[str, str]:
        """Return file name mapped to SHA-256 for every file in a directory.

        Args:
            rel: Directory relative to the archive root.

        Returns:
            File name mapped to hex digest.
        """
        # Like `sha256sum dir/*` on the server, hidden files are left out.
        entries = [e for e in self.list_entries(rel) if not e.is_dir and not e.name.startswith(".")]
        return {e.name: sha256_of(self._p(rel) / e.name) for e in entries}

    def sha256_file(self, rel: str) -> str:
        """Return the SHA-256 of one file.

        Args:
            rel: File relative to the archive root.

        Returns:
            Hex digest.
        """
        return sha256_of(self._p(rel))

    def read_range(self, rel: str, offset: int, length: int) -> bytes:
        """Return a byte range of a file.

        Args:
            rel: File relative to the archive root.
            offset: Start offset.
            length: Number of bytes.

        Returns:
            The bytes.
        """
        with self._p(rel).open("rb") as f:
            f.seek(offset)
            return f.read(length)

    def download_dir(self, rel: str, local_dir: Path, include: str | None = None) -> None:
        """Copy the files of a directory into a local directory.

        Args:
            rel: Directory relative to the archive root.
            local_dir: Local destination directory.
            include: Optional glob of names to copy.
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(self._p(rel).iterdir()):
            if src.is_file() and (include is None or src.match(include)):
                shutil.copy2(src, local_dir / src.name)

    def download_file(self, rel: str, local_path: Path) -> None:
        """Copy one file to a local path.

        Args:
            rel: File relative to the archive root.
            local_path: Local destination.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._p(rel), local_path)

    def upload_file(self, local_path: Path, rel: str) -> None:
        """Copy a local file to a remote path through a hidden temporary, as rsync does.

        Args:
            local_path: Local source file.
            rel: Destination relative to the archive root.

        Raises:
            RemoteCommandError: If the parent directory does not exist.
        """
        dst = self._p(rel)
        if not dst.parent.is_dir():
            msg = f"no such directory: {dst.parent}"
            raise RemoteCommandError(msg)
        tmp = dst.with_name(f".{dst.name}.rsync")
        shutil.copy2(local_path, tmp)
        tmp.replace(dst)

    def mkdir_p(self, rel: str) -> None:
        """Create a directory and its parents.

        Args:
            rel: Directory relative to the archive root.
        """
        self._p(rel).mkdir(parents=True, exist_ok=True)

    def move(self, src: str, dst: str) -> None:
        """Rename a path; the destination must not exist.

        Args:
            src: Source relative to the archive root.
            dst: Destination relative to the archive root.

        Raises:
            RemoteCommandError: If the destination exists.
        """
        if self._p(dst).exists():
            msg = f"refusing to move {src} onto existing {dst}"
            raise RemoteCommandError(msg)
        self._p(src).rename(self._p(dst))

    def replace_file(self, src: str, dst: str) -> None:
        """Rename a file over an existing one.

        Args:
            src: Source relative to the archive root.
            dst: Destination relative to the archive root.
        """
        self._p(src).replace(self._p(dst))

    def remove_tree(self, rel: str) -> None:
        """Delete a path under _superseded/.

        Args:
            rel: Path relative to the archive root.

        Raises:
            ValueError: If the path is outside _superseded/.
        """
        if not rel.startswith("_superseded/"):
            msg = f"refusing to delete outside _superseded/: {rel}"
            raise ValueError(msg)
        path = self._p(rel)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def free_bytes(self) -> int:
        """Return the configured free space.

        Returns:
            Bytes.
        """
        return self.free


def local_free_bytes(path: Path) -> int:
    """Return the free space of the file system holding a path.

    Args:
        path: Any existing path (its nearest existing parent is used).

    Returns:
        Free bytes.
    """
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free

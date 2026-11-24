"""Run-level safety: a lock against parallel runs and a clean stop on SIGINT / SIGTERM."""

import fcntl
import logging
import os
import signal
from pathlib import Path
from types import FrameType, TracebackType
from typing import IO, Self


logger = logging.getLogger(__name__)


class AlreadyRunningError(Exception):
    """Another run holds the lock."""


class RunLock:
    """An exclusive flock on a file; the kernel releases it if the process dies, so it never goes stale."""

    def __init__(self, path: Path) -> None:
        """Create the lock (not yet taken).

        Args:
            path: Lock file.
        """
        self.path = path
        self._fh: IO[str] | None = None

    def __enter__(self) -> Self:
        """Take the lock.

        Returns:
            The lock.

        Raises:
            AlreadyRunningError: If another process holds it.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fh.seek(0)
            holder = fh.read().strip()
            fh.close()
            msg = f"another run holds {self.path} (pid {holder or '?'})"
            raise AlreadyRunningError(msg) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Release the lock.

        Args:
            exc_type: Exception type, if any.
            exc: Exception, if any.
            tb: Traceback, if any.
        """
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


class StopFlag:
    """Turns the first SIGINT / SIGTERM into a request to stop at the next step boundary.

    A second signal raises KeyboardInterrupt at once. Steps are idempotent, so even that loses nothing:
    the interrupted step is simply redone on the next run.
    """

    def __init__(self) -> None:
        """Create the flag (handlers not installed)."""
        self.requested = False
        self._previous: dict[int, object] = {}

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        """Record a stop request, or abort on the second one.

        Args:
            signum: Signal number.
            frame: Current frame.

        Raises:
            KeyboardInterrupt: On the second signal.
        """
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        logger.warning("%s received: stopping after the current step", signal.Signals(signum).name)

    def __enter__(self) -> Self:
        """Install the handlers.

        Returns:
            The flag.
        """
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Restore the previous handlers.

        Args:
            exc_type: Exception type, if any.
            exc: Exception, if any.
            tb: Traceback, if any.
        """
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)  # type: ignore[arg-type]

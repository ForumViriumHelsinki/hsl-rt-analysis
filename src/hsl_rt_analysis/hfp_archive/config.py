"""Settings and the archive layout (relative paths under the archive root)."""

import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path


DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SENTINEL = ".hfp-archive-sentinel"
SCHEMA_VERSION = 1


class RawArchivePolicy(StrEnum):
    """When the per-day raw tar is written."""

    ALWAYS = "always"
    ON_C8_FAILURE = "on-c8-failure"


@dataclass(frozen=True)
class Settings:
    """Everything a run needs to know; built by the CLI from its options."""

    remote: str
    state_dir: Path
    work_dir: Path
    remote_port: int = 23
    remote_root: str = "2025_HSL_RT"
    memory_limit: str = "8GB"
    threads: int = 0  # 0 lets DuckDB use every core
    cutoff_days: int = 14
    min_remote_free_gb: float = 200.0
    min_local_free_gb: float = 4.0
    # Peak local disk of a day as a multiple of its raw size (measured: see docs/hfp-archive-converter.md).
    local_space_factor: float = 5.0
    raw_archive: RawArchivePolicy = RawArchivePolicy.ALWAYS
    max_wait_s: float = 6 * 3600.0

    @property
    def state_db(self) -> Path:
        """Return the path of the SQLite state database."""
        return self.state_dir / "state.sqlite"

    @property
    def lock_file(self) -> Path:
        """Return the path of the lock file that keeps runs from overlapping."""
        return self.state_dir / "run.lock"

    def day_work_dir(self, day: str) -> Path:
        """Return the local working directory of one day.

        Args:
            day: The day, YYYY-MM-DD.

        Returns:
            Directory path (not created).
        """
        return self.work_dir / day


def validate_day(day: str) -> str:
    """Check that a string is a real YYYY-MM-DD date.

    Args:
        day: Candidate day string.

    Returns:
        The same string.

    Raises:
        ValueError: If it is not a valid date.
    """
    if not DAY_RE.match(day):
        msg = f"not a YYYY-MM-DD day: {day!r}"
        raise ValueError(msg)
    date.fromisoformat(day)
    return day


# --- archive layout, relative to the archive root -------------------------------------------------


def raw_day(day: str) -> str:
    """Return the raw day directory of the current layout.

    Args:
        day: The day, YYYY-MM-DD.

    Returns:
        Path relative to the archive root.
    """
    return f"raw/all/{day}"


def interim_day(day: str) -> str:
    """Return the interim day directory of the current layout.

    Args:
        day: The day, YYYY-MM-DD.

    Returns:
        Path relative to the archive root.
    """
    return f"interim/{day}"


def hfp_day(day: str) -> str:
    """Return the Parquet day directory of the archive format.

    Args:
        day: The day, YYYY-MM-DD.

    Returns:
        Path relative to the archive root.
    """
    return f"hsl/hfp/dt={day}"


def hfp_day_tmp(day: str) -> str:
    """Return the temporary upload directory of a Parquet day; the dot keeps readers away from it.

    Args:
        day: The day, YYYY-MM-DD.

    Returns:
        Path relative to the archive root.
    """
    return f"hsl/hfp/.dt={day}.tmp"


def raw_tar_dir(day: str) -> str:
    """Return the directory of a day's raw tar.

    Args:
        day: The day, YYYY-MM-DD.

    Returns:
        Path relative to the archive root.
    """
    y, m, d = day.split("-")
    return f"hsl/hfp-raw/{y}/{m}/{d}"


def raw_tar_name(day: str) -> str:
    """Return the file name of a day's raw tar.

    Args:
        day: The day, YYYY-MM-DD.

    Returns:
        Path relative to the archive root.
    """
    return f"hfp-raw-{day}.tar"


def superseded(rel: str) -> str:
    """Return where a replaced source path is kept until it is purged.

    Args:
        rel: Path relative to the archive root.

    Returns:
        Path relative to the archive root.
    """
    return f"_superseded/{rel}"

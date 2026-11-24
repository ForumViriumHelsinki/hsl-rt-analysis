"""Find the days on the storage box and decide which of them a run may touch."""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from hsl_rt_analysis.hfp_archive.config import DAY_RE, Settings, superseded
from hsl_rt_analysis.hfp_archive.remote import Remote
from hsl_rt_analysis.hfp_archive.state import DayState, StateDB


logger = logging.getLogger(__name__)


@dataclass
class Inventory:
    """Which days exist where on the storage box."""

    raw: set[str] = field(default_factory=set)
    interim: set[str] = field(default_factory=set)
    hfp: set[str] = field(default_factory=set)
    superseded_raw: set[str] = field(default_factory=set)
    superseded_interim: set[str] = field(default_factory=set)

    @property
    def interim_only(self) -> list[str]:
        """Return the days with interim files but no raw files, never converted automatically."""
        return sorted(self.interim - self.raw - self.hfp)

    @property
    def reappeared(self) -> list[str]:
        """Return converted days whose original directories exist again (e.g. after a manual rsync)."""
        return sorted((self.raw | self.interim) & self.hfp)


def _day_dirs(remote: Remote, rel: str) -> set[str]:
    """List the YYYY-MM-DD (or dt=YYYY-MM-DD) subdirectories of a remote directory.

    Args:
        remote: The storage box.
        rel: Directory relative to the root.

    Returns:
        Day strings; empty if the directory does not exist.
    """
    if not remote.exists(rel):
        return set()
    days = set()
    for e in remote.list_entries(rel):
        name = e.name.removeprefix("dt=")
        if e.is_dir and DAY_RE.match(name):
            days.add(name)
    return days


def inventory(remote: Remote) -> Inventory:
    """List the days of every area of the archive root.

    Args:
        remote: The storage box.

    Returns:
        The inventory.
    """
    return Inventory(
        raw=_day_dirs(remote, "raw/all"),
        interim=_day_dirs(remote, "interim"),
        hfp=_day_dirs(remote, "hsl/hfp"),
        superseded_raw=_day_dirs(remote, superseded("raw/all")),
        superseded_interim=_day_dirs(remote, superseded("interim")),
    )


def cutoff_day(settings: Settings, today: date | None = None) -> str:
    """Return the last day a run may convert: older days can no longer receive files from dckr3's rsync.

    dckr3 syncs only yesterday's directories (16:11 and 16:15 EET) and deletes its own copies after
    8 days, so a day is final within about 10 days; cutoff_days adds a margin on top.

    Args:
        settings: Run settings.
        today: Today (default: the real today).

    Returns:
        The day, YYYY-MM-DD.
    """
    return ((today or date.today()) - timedelta(days=settings.cutoff_days)).isoformat()


def select_days(
    inv: Inventory,
    db: StateDB,
    cutoff: str,
    first: str | None = None,
    last: str | None = None,
    only: list[str] | None = None,
) -> list[str]:
    """Return the days a run should advance, oldest first.

    A day qualifies if it has raw files (or is mid-way through the pipeline), is not after the cutoff,
    is within the requested range, and is not finished or waiting for a human.

    Args:
        inv: The storage box inventory.
        db: State database.
        cutoff: Last allowed day.
        first: First day of the range.
        last: Last day of the range.
        only: Explicit list of days (still subject to the cutoff).

    Returns:
        Days in order.
    """
    stopped = {DayState.DONE, DayState.NEEDS_REVIEW, DayState.FAILED}
    in_flight = {r.day for r in db.all() if r.state not in stopped and r.state != DayState.PENDING}
    candidates = (inv.raw | in_flight) if only is None else set(only)
    days = []
    for day in sorted(candidates):
        if day > cutoff:
            if only is not None:
                logger.warning("Skipping %s: after the cutoff %s (rsync may still write to it)", day, cutoff)
            continue
        if (first and day < first) or (last and day > last):
            continue
        record = db.get(day)
        if record is not None and record.state in stopped:
            continue
        if day not in inv.raw and day not in in_flight:
            logger.warning("Skipping %s: no raw directory", day)
            continue
        days.append(day)
    return days

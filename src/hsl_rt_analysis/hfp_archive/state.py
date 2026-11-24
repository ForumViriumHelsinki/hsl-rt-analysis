"""The per-day state machine, kept in a local SQLite database.

The storage box manifest repeats the state from `uploaded` on, so the archive itself says how far each
day got even if this database is lost; `audit` reads only the storage box.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class DayState(StrEnum):
    """Where one day is. The forward path is PENDING ... DONE; the other two stop the day for a human."""

    PENDING = "pending"
    FETCHED = "fetched"
    CONVERTED = "converted"
    VERIFIED = "verified"
    UPLOADED = "uploaded"
    SWAPPED = "swapped"
    DONE = "done"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


FORWARD = [
    DayState.PENDING,
    DayState.FETCHED,
    DayState.CONVERTED,
    DayState.VERIFIED,
    DayState.UPLOADED,
    DayState.SWAPPED,
    DayState.DONE,
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS days (
    day TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    info TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    at TEXT NOT NULL,
    state TEXT NOT NULL,
    message TEXT
);
CREATE TABLE IF NOT EXISTS audits (
    day TEXT NOT NULL,
    at TEXT NOT NULL,
    ok INTEGER NOT NULL,
    deep INTEGER NOT NULL,
    details TEXT NOT NULL
);
"""


def now_iso() -> str:
    """Return the current UTC time as ISO 8601 with seconds.

    Returns:
        Timestamp string.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class DayRecord:
    """One row of the days table."""

    day: str
    state: DayState
    updated_at: str
    attempts: int
    last_error: str | None
    info: dict[str, Any]


class StateDB:
    """Read and write day states. Every write is its own committed transaction."""

    def __init__(self, path: Path) -> None:
        """Open (and create) the database.

        Args:
            path: SQLite file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, isolation_level=None)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=FULL")
        self.con.executescript(SCHEMA)

    def close(self) -> None:
        """Close the database."""
        self.con.close()

    def get(self, day: str) -> DayRecord | None:
        """Return one day's record.

        Args:
            day: The day.

        Returns:
            The record, or None if the day is unknown.
        """
        row = self.con.execute(
            "SELECT day, state, updated_at, attempts, last_error, info FROM days WHERE day = ?", [day]
        ).fetchone()
        return self._record(row) if row else None

    def require(self, day: str) -> DayRecord:
        """Return one day's record, which must exist.

        Args:
            day: The day.

        Returns:
            The record.

        Raises:
            KeyError: If the day is unknown.
        """
        record = self.get(day)
        if record is None:
            raise KeyError(day)
        return record

    def all(self) -> list[DayRecord]:
        """Return every known day, oldest first.

        Returns:
            Records.
        """
        rows = self.con.execute("SELECT day, state, updated_at, attempts, last_error, info FROM days ORDER BY day")
        return [self._record(r) for r in rows]

    @staticmethod
    def _record(row: tuple[Any, ...]) -> DayRecord:
        """Turn a database row into a record.

        Args:
            row: The selected columns.

        Returns:
            The record.
        """
        return DayRecord(row[0], DayState(row[1]), row[2], row[3], row[4], json.loads(row[5]))

    def ensure(self, day: str) -> DayRecord:
        """Register a day as pending unless it is already known.

        Args:
            day: The day.

        Returns:
            The day's record.
        """
        self.con.execute(
            "INSERT OR IGNORE INTO days (day, state, updated_at) VALUES (?, ?, ?)", [day, DayState.PENDING, now_iso()]
        )
        record = self.get(day)
        assert record is not None
        return record

    def set_state(self, day: str, state: DayState, message: str | None = None, **info: Any) -> None:  # noqa: ANN401
        """Move a day to a new state and merge extra info into its record.

        Args:
            day: The day.
            state: The new state.
            message: Optional note for the event log (an error for FAILED / NEEDS_REVIEW).
            **info: Values merged into the day's info JSON.
        """
        record = self.ensure(day)
        merged = {**record.info, **info}
        error = message if state in {DayState.FAILED, DayState.NEEDS_REVIEW} else None
        self.con.execute("BEGIN IMMEDIATE")
        self.con.execute(
            "UPDATE days SET state = ?, updated_at = ?, last_error = ?, info = ? WHERE day = ?",
            [state, now_iso(), error, json.dumps(merged, default=str), day],
        )
        self.con.execute(
            "INSERT INTO events (day, at, state, message) VALUES (?, ?, ?, ?)", [day, now_iso(), state, message]
        )
        self.con.execute("COMMIT")

    def bump_attempts(self, day: str) -> None:
        """Count one more attempt at a day.

        Args:
            day: The day.
        """
        self.ensure(day)
        self.con.execute("UPDATE days SET attempts = attempts + 1 WHERE day = ?", [day])

    def events(self, day: str) -> list[tuple[str, str, str | None]]:
        """Return a day's event log.

        Args:
            day: The day.

        Returns:
            (at, state, message) tuples, oldest first.
        """
        return self.con.execute("SELECT at, state, message FROM events WHERE day = ? ORDER BY id", [day]).fetchall()

    def record_audit(self, day: str, ok: bool, deep: bool, details: dict[str, Any]) -> None:
        """Store the result of an audit of one day.

        Args:
            day: The day.
            ok: Whether every audit check passed.
            deep: Whether it was a deep audit.
            details: Check results.
        """
        self.con.execute(
            "INSERT INTO audits (day, at, ok, deep, details) VALUES (?, ?, ?, ?, ?)",
            [day, now_iso(), int(ok), int(deep), json.dumps(details, default=str)],
        )

    def last_audit(self, day: str) -> tuple[str, bool] | None:
        """Return the time and result of a day's latest audit.

        Args:
            day: The day.

        Returns:
            (at, ok), or None if never audited.
        """
        row = self.con.execute("SELECT at, ok FROM audits WHERE day = ? ORDER BY at DESC LIMIT 1", [day]).fetchone()
        return (row[0], bool(row[1])) if row else None

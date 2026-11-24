"""The day's _manifest.json: counts, checksums, check results and the archive state."""

import json
from pathlib import Path
from typing import Any

import duckdb

from hsl_rt_analysis import hfp_archive
from hsl_rt_analysis.hfp_archive.config import SCHEMA_VERSION, interim_day, raw_day, raw_tar_dir, raw_tar_name
from hsl_rt_analysis.hfp_archive.state import now_iso


MANIFEST_NAME = "_manifest.json"
# C8 is required only when no raw tar is kept: with the tar, the raw bytes are safe whatever C8 says.
REQUIRED_CHECKS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def all_checks_ok(checks: dict[str, dict[str, Any]], raw_tar_kept: bool) -> bool:
    """Return whether every required check passed.

    Args:
        checks: Check name mapped to result (names start with C1 ... C8).
        raw_tar_kept: Whether the day's raw tar is part of the archive.

    Returns:
        True if all required checks are ok.
    """
    required = REQUIRED_CHECKS if raw_tar_kept else [*REQUIRED_CHECKS, "C8"]
    by_id = {name.split()[0]: result for name, result in checks.items()}
    return all(by_id.get(c, {}).get("ok") is True for c in required)


def build(
    day: str,
    source: list[dict[str, Any]],
    interim: list[dict[str, Any]] | None,
    files: list[dict[str, Any]],
    raw_archive: dict[str, Any] | None,
    counts: dict[str, Any],
    checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the manifest of a day.

    Args:
        day: The day.
        source: Raw source files with name, size, mtime, sha256 and content_sha256.
        interim: Interim files of the day (name, size, mtime, sha256), or None if there were none.
        files: Uploaded data files of the dt= directory (path, size, sha256).
        raw_archive: The raw tar entry, or None if the day keeps no tar.
        counts: Line and row counts.
        checks: Check results C1-C8.

    Returns:
        Manifest dictionary.
    """
    return {
        "format": "hsl-hfp-archive",
        "schema_version": SCHEMA_VERSION,
        "dt": day,
        "created_at": now_iso(),
        "converter": {
            "name": "hsl_rt_analysis.hfp_archive",
            "version": hfp_archive.__version__,
            "duckdb": duckdb.__version__,
        },
        "source": {"path": raw_day(day), "files": source},
        "interim": {"path": interim_day(day), "files": interim} if interim is not None else None,
        "parquet": [f for f in files if f["path"].startswith("part-")],
        "rejects": next((f for f in files if f["path"] == "_rejects.json.gz"), None),
        "repeats": next((f for f in files if f["path"] == "_repeats.parquet"), None),
        "raw_archive": raw_archive,
        "counts": counts,
        "checks": checks,
        "all_checks_ok": all_checks_ok(checks, raw_archive is not None),
        "archive_state": {"state": "uploaded", "history": [{"state": "uploaded", "at": now_iso()}]},
    }


def raw_archive_entry(day: str, tar_path: Path, sha256: str, members: int) -> dict[str, Any]:
    """Return the manifest entry of a day's raw tar.

    Args:
        day: The day.
        tar_path: Local tar.
        sha256: Its SHA-256.
        members: Number of members.

    Returns:
        The entry.
    """
    return {
        "path": f"{raw_tar_dir(day)}/{raw_tar_name(day)}",
        "size": tar_path.stat().st_size,
        "sha256": sha256,
        "members": members,
    }


def advance(manifest: dict[str, Any], state: str, **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return a copy of the manifest moved to a new archive state.

    Args:
        manifest: The manifest.
        state: The new state (uploaded, swapped, done, purged).
        **extra: Fields merged into archive_state (e.g. superseded paths).

    Returns:
        The updated manifest.
    """
    archive_state = dict(manifest["archive_state"])
    if archive_state["state"] != state:
        archive_state["history"] = [*archive_state["history"], {"state": state, "at": now_iso()}]
    archive_state.update(state=state, **extra)
    return {**manifest, "archive_state": archive_state}


def dumps(manifest: dict[str, Any]) -> str:
    """Serialize a manifest.

    Args:
        manifest: The manifest.

    Returns:
        JSON text.
    """
    return json.dumps(manifest, indent=2, default=str) + "\n"

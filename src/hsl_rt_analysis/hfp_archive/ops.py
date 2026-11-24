"""Operator commands: progress per day, the report of days that need a decision, audit and purge."""

import json
import logging
import shutil
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from hsl_rt_analysis.hfp_archive import manifest as mf
from hsl_rt_analysis.hfp_archive.config import Settings, hfp_day, interim_day, raw_day, superseded
from hsl_rt_analysis.hfp_archive.convert import REJECTS_NAME, REPEATS_NAME, connect
from hsl_rt_analysis.hfp_archive.discover import Inventory
from hsl_rt_analysis.hfp_archive.remote import Remote
from hsl_rt_analysis.hfp_archive.restore import check_c8, restored_contents
from hsl_rt_analysis.hfp_archive.state import DayRecord, DayState, StateDB


logger = logging.getLogger(__name__)


def _duration(record: DayRecord, db: StateDB) -> float | None:
    """Return the seconds a finished day took from its first to its last event.

    Args:
        record: The day.
        db: State database.

    Returns:
        Seconds, or None if not finished.
    """
    events = db.events(record.day)
    if record.state != DayState.DONE or len(events) < 2:
        return None
    total = 0.0
    for _at, _state, message in events:
        if message and message.endswith(" s") and " took " in message:
            total += float(message.rsplit(" ", 2)[1])
    return total


def status_lines(db: StateDB, pending: list[str], show_all: bool = False) -> list[str]:
    """Return the progress table, one line per day, plus a summary.

    Args:
        db: State database.
        pending: Eligible days not yet in the database.
        show_all: Include days that are done.

    Returns:
        Lines to print.
    """
    records = db.all()
    lines = [f"{'day':<10}  {'state':<12}  {'updated (UTC)':<19}  {'rows':>12}  {'took':>6}  note"]
    durations = []
    for r in records:
        took = _duration(r, db)
        if took:
            durations.append(took)
        if r.state == DayState.DONE and not show_all:
            continue
        rows = r.info.get("rows", "")
        note = (r.last_error or "")[:90]
        took_s = f"{took / 60:.0f} m" if took else ""
        lines.append(f"{r.day:<10}  {r.state:<12}  {r.updated_at[:19]:<19}  {rows!s:>12}  {took_s:>6}  {note}")
    counts = Counter(r.state.value for r in records)
    todo = len(pending) + sum(
        counts[s] for s in ("pending", "fetched", "converted", "verified", "uploaded", "swapped")
    )
    lines.append("")
    lines.append("  ".join(f"{s}: {n}" for s, n in sorted(counts.items())) + f"  not started: {len(pending)}")
    if durations:
        avg = sum(durations) / len(durations)
        lines.append(f"average {avg / 60:.1f} min per day; {todo} days to go = about {todo * avg / 3600:.0f} h")
    return lines


def report_lines(inv: Inventory, db: StateDB) -> list[str]:
    """Return the days that need a human decision.

    Args:
        inv: Storage box inventory.
        db: State database.

    Returns:
        Lines to print.
    """
    out = []
    interim_only = inv.interim_only
    out.append(f"Interim only, not converted ({len(interim_only)} days):")
    out.extend(f"  {start} .. {end} ({n} days)" for start, end, n in _ranges(interim_only))
    states = {r.day: r for r in db.all()}
    reappeared = [d for d in inv.reappeared if d in states and states[d].state in {DayState.SWAPPED, DayState.DONE}]
    out.append(f"Originals reappeared after the swap ({len(reappeared)}): {', '.join(reappeared) or '-'}")
    for state in (DayState.NEEDS_REVIEW, DayState.FAILED):
        days = [r for r in states.values() if r.state == state]
        out.append(f"{state} ({len(days)}):")
        out.extend(f"  {r.day} (stopped in {r.info.get('resume_from', '?')}): {r.last_error}" for r in days)
    noted = [r for r in states.values() if r.info.get("rejected") or r.info.get("tar_reason")]
    out.append(f"Days with rejected lines or a C8 tar fallback ({len(noted)}):")
    out.extend(
        f"  {r.day}: rejected {r.info.get('rejected', 0)} {r.info.get('tar_reason', '')}".rstrip() for r in noted
    )
    return out


def _ranges(days: list[str]) -> list[tuple[str, str, int]]:
    """Collapse sorted days into runs of consecutive days.

    Args:
        days: Sorted YYYY-MM-DD strings.

    Returns:
        (first, last, count) per run.
    """
    runs: list[tuple[str, str, int]] = []
    for day in days:
        d = datetime.fromisoformat(day).date()
        if runs and datetime.fromisoformat(runs[-1][1]).date() + timedelta(days=1) == d:
            runs[-1] = (runs[-1][0], day, runs[-1][2] + 1)
        else:
            runs.append((day, day, 1))
    return runs


# --- audit -------------------------------------------------------------------------------------------


def _files_match(actual: dict[str, str], expected: list[dict[str, Any]], key: str = "name") -> bool:
    """Return whether a directory's server-side digests equal the manifest's list.

    Args:
        actual: Name mapped to SHA-256 on the storage box.
        expected: Manifest file entries.
        key: Field holding the file name.

    Returns:
        True if both name sets and every digest agree.
    """
    want = {f[key]: f["sha256"] for f in expected if not f.get("is_dir")}
    return actual == want


def _sizes_match(remote: Remote, rel: str, expected: list[dict[str, Any]]) -> bool:
    """Return whether a directory holds exactly the expected file names with the expected sizes.

    Args:
        remote: The storage box.
        rel: Directory relative to the root.
        expected: Manifest file entries with name and size.

    Returns:
        True if names and sizes agree.
    """
    actual = {e.name: e.size for e in remote.list_entries(rel) if not e.is_dir}
    return actual == {f["name"]: f["size"] for f in expected if not f.get("is_dir")}


def audit_day(remote: Remote, settings: Settings, day: str, deep: bool, full: bool = False) -> dict[str, Any]:
    """Check one converted day on the storage box alone, without trusting the local state.

    The new files are always checked by their SHA-256 computed on the server. The superseded originals
    were checked by SHA-256 when they were moved; here they are compared by name and size, or by
    SHA-256 again with full=True (about a minute per GB on the storage box).

    Args:
        remote: The storage box.
        settings: Settings (for the scratch directory and memory limit).
        day: The day.
        deep: Also download the Parquet files and rebuild every raw file (C8 again).
        full: Also recompute the SHA-256 of the superseded originals.

    Returns:
        Results per audit check, with an overall "ok".
    """
    scratch = settings.work_dir / f"_audit-{day}"
    shutil.rmtree(scratch, ignore_errors=True)
    remote.download_file(f"{hfp_day(day)}/{mf.MANIFEST_NAME}", scratch / mf.MANIFEST_NAME)
    m = json.loads((scratch / mf.MANIFEST_NAME).read_text())
    r: dict[str, Any] = {"manifest checks": m.get("all_checks_ok") is True and m.get("schema_version") == 1}
    server = remote.sha256_dir(hfp_day(day))
    data = [*m["parquet"], *(f for f in (m.get("rejects"), m.get("repeats")) if f)]
    r["parquet sha256"] = _files_match({k: v for k, v in server.items() if k != mf.MANIFEST_NAME}, data, "path")
    if m.get("raw_archive"):
        r["raw tar sha256"] = remote.sha256_file(m["raw_archive"]["path"]) == m["raw_archive"]["sha256"]
    r["originals gone"] = not remote.exists(raw_day(day)) and not remote.exists(interim_day(day))
    originals = [(superseded(raw_day(day)), m["source"]["files"], "raw")]
    if m.get("interim"):
        originals.append((superseded(interim_day(day)), m["interim"]["files"], "interim"))
    for rel, files, what in originals:
        if not remote.exists(rel):
            continue  # already purged
        if full:
            r[f"superseded {what} sha256"] = _files_match(remote.sha256_dir(rel), files)
        else:
            r[f"superseded {what} sizes"] = _sizes_match(remote, rel, files)
    if deep:
        r["deep C8"] = _deep_rebuild(remote, settings, m, scratch)["ok"]
    shutil.rmtree(scratch, ignore_errors=True)
    r["ok"] = all(v is True for v in r.values())
    return r


def _deep_rebuild(remote: Remote, settings: Settings, m: dict[str, Any], scratch: Path) -> dict[str, Any]:
    """Download a day's Parquet files and compare every rebuilt raw file with the manifest's content digest.

    Args:
        remote: The storage box.
        settings: Settings.
        m: The day's manifest.
        scratch: Local scratch directory.

    Returns:
        The C8 result.
    """
    remote.download_dir(hfp_day(m["dt"]), scratch / "dt")
    parts = sorted((scratch / "dt").glob("part-*.parquet"))
    repeats = scratch / "dt" / REPEATS_NAME
    rejects = scratch / "dt" / REJECTS_NAME
    con = connect(settings.memory_limit, settings.threads, scratch / "tmp")
    try:
        restored = restored_contents(
            con, parts, repeats if repeats.exists() else None, rejects if rejects.exists() else None
        )
    finally:
        con.close()
    original = {f["name"]: {"content_sha256": f["content_sha256"]} for f in m["source"]["files"]}
    result = check_c8(original, restored)
    logger.info("%s deep audit: %s", m["dt"], result)
    return result


# --- purge -------------------------------------------------------------------------------------------


def purge_candidates(
    db: StateDB, inv: Inventory, before: str, audit_max_age: timedelta
) -> tuple[list[str], list[str]]:
    """Split the days with _superseded/ data before a date into purgeable and blocked ones.

    Args:
        db: State database.
        inv: Storage box inventory.
        before: Only days strictly before this day.
        audit_max_age: How recent a passed audit must be.

    Returns:
        (purgeable days, blocked days with the reason).
    """
    days = sorted(d for d in inv.superseded_raw | inv.superseded_interim if d < before)
    ok, blocked = [], []
    now = datetime.now(UTC)
    for day in days:
        record = db.get(day)
        audit = db.last_audit(day)
        if record is None or record.state != DayState.DONE:
            blocked.append(f"{day}: state {record.state if record else 'unknown'}")
        elif audit is None or not audit[1]:
            blocked.append(f"{day}: no passed audit")
        elif now - datetime.fromisoformat(audit[0]) > audit_max_age:
            blocked.append(f"{day}: audit older than {audit_max_age.days} days")
        else:
            ok.append(day)
    return ok, blocked


def purge(remote: Remote, inv: Inventory, days: list[str], execute: bool) -> int:
    """Delete (or with execute=False only list) the _superseded/ directories of the given days.

    Args:
        remote: The storage box.
        inv: Storage box inventory.
        days: Days whose superseded data may go.
        execute: Really delete.

    Returns:
        Bytes freed (or that would be freed).
    """
    total = 0
    for day in days:
        for rel, present in (
            (superseded(raw_day(day)), day in inv.superseded_raw),
            (superseded(interim_day(day)), day in inv.superseded_interim),
        ):
            if not present:
                continue
            entries = remote.list_entries(rel)
            size = sum(e.size for e in entries)
            total += size
            if execute:
                remote.remove_tree(rel)
                logger.info("Deleted %s (%d files, %.2f GB)", rel, len(entries), size / 1e9)
            else:
                logger.info("[dry-run] would delete %s (%d files, %.2f GB)", rel, len(entries), size / 1e9)
    return total

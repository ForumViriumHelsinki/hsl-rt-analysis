"""The per-day state machine: fetch -> convert -> verify -> upload -> swap -> done.

Every step is idempotent and first looks at what is already there, so a run that stops anywhere
(signal, crash, sleep, lost connection) continues from the last recorded state. Nothing is overwritten
on the storage box: new files go to temporary names, are checked by their server-side SHA-256 (C6) and
renamed; the originals move to _superseded/ only after every check has passed and only if they are
still exactly what was converted.
"""

import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import duckdb

from hsl_rt_analysis.hfp_archive import manifest as mf
from hsl_rt_analysis.hfp_archive.checks import build_tar, check_c5, check_c7
from hsl_rt_analysis.hfp_archive.config import (
    RawArchivePolicy,
    Settings,
    hfp_day,
    hfp_day_tmp,
    interim_day,
    raw_day,
    raw_tar_dir,
    raw_tar_name,
    superseded,
)
from hsl_rt_analysis.hfp_archive.convert import (
    CONVERSION_NAME,
    FILE_STATS_NAME,
    REJECTS_NAME,
    REPEATS_NAME,
    convert_day,
)
from hsl_rt_analysis.hfp_archive.guards import StopFlag
from hsl_rt_analysis.hfp_archive.remote import (
    Remote,
    RemoteEntry,
    RemoteUnavailableError,
    SentinelMissingError,
    local_free_bytes,
    sha256_of,
)
from hsl_rt_analysis.hfp_archive.restore import check_c8, original_contents, restored_contents
from hsl_rt_analysis.hfp_archive.state import DayState, StateDB


logger = logging.getLogger(__name__)

GB = 1e9


class StopRequestedError(Exception):
    """A signal asked the run to stop; raised at a step boundary."""


class RemoteFullError(Exception):
    """The storage box has less free space than the configured minimum: purge _superseded/ first."""


class LocalSpaceError(Exception):
    """The local disk has too little free space for the next step."""


class CheckFailedError(Exception):
    """A check failed: the day stops in FAILED for a human to look at."""


class ReviewNeededError(Exception):
    """The day needs a manual decision (C7, a changed source, a conflict on the storage box)."""


@dataclass
class DayPaths:
    """The local working files of one day."""

    root: Path

    @property
    def raw(self) -> Path:
        """Local copy of the raw day directory."""
        return self.root / "raw"

    @property
    def out(self) -> Path:
        """The future dt= directory."""
        return self.root / "out"

    @property
    def stage(self) -> Path:
        """Conversion scratch space."""
        return self.root / "stage"

    @property
    def tar_dir(self) -> Path:
        """Where the raw tar is built."""
        return self.root / "tar"

    @property
    def scratch(self) -> Path:
        """Downloads for C7."""
        return self.root / "scratch"

    def meta(self, name: str) -> Path:
        """Return a metadata file of the day.

        Args:
            name: File name.

        Returns:
            Path.
        """
        return self.root / name


def read_json(path: Path) -> Any:  # noqa: ANN401
    """Read a JSON file.

    Args:
        path: File.

    Returns:
        The parsed value.
    """
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:  # noqa: ANN401
    """Write a JSON file atomically.

    Args:
        path: File.
        value: Value to write.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


@dataclass
class RunSummary:
    """What a run did."""

    finished: dict[str, DayState] = field(default_factory=dict)
    stopped_reason: str | None = None
    exit_code: int = 0


class Pipeline:
    """Moves days through the state machine."""

    def __init__(
        self,
        settings: Settings,
        remote: Remote,
        db: StateDB,
        stop: StopFlag | None = None,
        checkpoint: Callable[[str], None] | None = None,
    ) -> None:
        """Create the pipeline.

        Args:
            settings: Run settings.
            remote: The storage box.
            db: State database.
            stop: Stop flag set by signal handlers.
            checkpoint: Test hook called with a name at every point where an interruption is interesting.
        """
        self.s = settings
        self.remote = remote
        self.db = db
        self.stop = stop
        self.checkpoint = checkpoint or (lambda name: None)

    # --- guards -----------------------------------------------------------------------------------

    def _guard(self, day: str, step: str, local_need: float) -> None:
        """Check, before a step, the stop flag, the storage box (sentinel) and the local disk.

        Args:
            day: The day.
            step: Name of the step about to run.
            local_need: Bytes of local disk the step may need.

        Raises:
            StopRequestedError: If a stop was requested.
            LocalSpaceError: If the local disk is too full.
        """
        if self.stop is not None and self.stop.requested:
            raise StopRequestedError
        self.remote.check_sentinel()
        free = local_free_bytes(self.s.work_dir)
        need = local_need + self.s.min_local_free_gb * GB
        if free < need:
            msg = f"{day} {step}: {free / GB:.1f} GB free locally, need {need / GB:.1f} GB"
            raise LocalSpaceError(msg)

    def _raw_size(self, day: str) -> int:
        """Return the raw size of a day from the fetch record (0 before the fetch).

        Args:
            day: The day.

        Returns:
            Bytes.
        """
        path = DayPaths(self.s.day_work_dir(day)).meta("source.json")
        return sum(f["size"] for f in read_json(path)) if path.exists() else 0

    # --- steps ------------------------------------------------------------------------------------

    def fetch(self, day: str) -> DayState:
        """Copy the raw day here and confirm every file against its SHA-256 on the storage box.

        Args:
            day: The day.

        Returns:
            FETCHED.

        Raises:
            CheckFailedError: If the copy does not match the storage box.
        """
        p = DayPaths(self.s.day_work_dir(day))
        entries = self.remote.list_entries(raw_day(day))
        if any(e.is_dir for e in entries):
            msg = f"directory inside {raw_day(day)}"
            raise CheckFailedError(msg)
        size = sum(e.size for e in entries)
        self._guard(day, "fetch", size * self.s.local_space_factor)
        self.remote.download_dir(raw_day(day), p.raw)
        self.checkpoint("fetch:downloaded")
        server = self.remote.sha256_dir(raw_day(day))
        local_names = {f.name for f in p.raw.iterdir()}
        if local_names != set(server) or set(server) != {e.name for e in entries}:
            msg = f"file list changed during the fetch of {day}"
            raise CheckFailedError(msg)
        bad = [n for n, sha in server.items() if sha256_of(p.raw / n) != sha]
        if bad:
            for n in bad:
                (p.raw / n).unlink()
            self.remote.download_dir(raw_day(day), p.raw)
            bad = [n for n, sha in server.items() if sha256_of(p.raw / n) != sha]
        if bad:
            msg = f"{len(bad)} files differ from the storage box after re-download, e.g. {bad[0]}"
            raise CheckFailedError(msg)
        by_name = {e.name: e for e in entries}
        source = [
            {"name": n, "size": by_name[n].size, "mtime": int((p.raw / n).stat().st_mtime), "sha256": server[n]}
            for n in sorted(server)
        ]
        write_json(p.meta("source.json"), source)
        write_json(p.meta("interim.json"), self._interim_inventory(day))
        return DayState.FETCHED

    def _interim_inventory(self, day: str) -> list[dict[str, Any]] | None:
        """List the day's interim files with their server-side SHA-256.

        Args:
            day: The day.

        Returns:
            The files, or None if the day has no interim directory.
        """
        if not self.remote.exists(interim_day(day)):
            return None
        entries = self.remote.list_entries(interim_day(day))
        shas = self.remote.sha256_dir(interim_day(day))
        return [
            {"name": e.name, "size": e.size, "mtime": e.mtime, "sha256": shas.get(e.name), "is_dir": e.is_dir}
            for e in entries
        ]

    def convert(self, day: str) -> DayState:
        """Convert the local raw copy and run C1-C4.

        Args:
            day: The day.

        Returns:
            CONVERTED.

        Raises:
            CheckFailedError: If any of C1-C4 fails.
        """
        p = DayPaths(self.s.day_work_dir(day))
        self._guard(day, "convert", self._raw_size(day) * (self.s.local_space_factor - 1))
        result = convert_day(day, p.raw, p.out, p.stage, self.s.memory_limit, self.s.threads)
        failed = [name for name, r in result.checks.items() if r.get("ok") is not True]
        logger.info(
            "%s converted in %.0f s: %d rows, checks %s",
            day,
            result.seconds,
            result.counts["rows"],
            "ok" if not failed else f"FAILED {failed}",
        )
        if failed:
            msg = f"conversion checks failed: {failed}"
            raise CheckFailedError(msg)
        return DayState.CONVERTED

    def verify(self, day: str) -> DayState:
        """Run C8 (raw rebuild), build and check the tar (C5) and compare with interim (C7).

        Args:
            day: The day.

        Returns:
            VERIFIED.

        Raises:
            CheckFailedError: If C5 fails, or C8 fails while no tar would be kept.
            ReviewNeededError: If C7 fails.
        """
        p = DayPaths(self.s.day_work_dir(day))
        raw_size = self._raw_size(day)
        self._guard(day, "verify", raw_size * 1.1)
        source = read_json(p.meta("source.json"))
        checks: dict[str, dict[str, Any]] = {}
        contents = original_contents([p.raw / f["name"] for f in source])
        parts = sorted(p.out.glob("part-*.parquet"))
        repeats = p.out / REPEATS_NAME if (p.out / REPEATS_NAME).exists() else None
        rejects = p.out / REJECTS_NAME if (p.out / REJECTS_NAME).exists() else None
        con = duckdb.connect()
        try:
            con.execute(f"SET memory_limit = '{self.s.memory_limit}'; SET temp_directory = '{p.stage}'")
            restored = restored_contents(con, parts, repeats, rejects)
        finally:
            con.close()
        shutil.rmtree(p.stage, ignore_errors=True)
        checks["C8 raw rebuild"] = check_c8(contents, restored)
        for f in source:
            f["content_sha256"] = contents[f["name"]]["content_sha256"]
        write_json(p.meta("source.json"), source)
        keep_tar = self.s.raw_archive == RawArchivePolicy.ALWAYS or not checks["C8 raw rebuild"]["ok"]
        if keep_tar:
            tar = build_tar(p.raw, [f["name"] for f in source], p.tar_dir / raw_tar_name(day), day)
            checks["C5 raw tar"] = check_c5(tar, {f["name"]: f["sha256"] for f in source}, day)
        else:
            shutil.rmtree(p.tar_dir, ignore_errors=True)
            checks["C5 raw tar"] = {"ok": True, "tar": False, "reason": "C8 passed; policy on-c8-failure"}
        interim = read_json(p.meta("interim.json"))
        if interim is None:
            checks["C7 interim sanity"] = {"ok": True, "interim": False}
        else:
            entries = [RemoteEntry(f["name"], f["size"], f["mtime"], f["is_dir"]) for f in interim]
            checks["C7 interim sanity"] = check_c7(self.remote, day, entries, p.meta(FILE_STATS_NAME), p.scratch)
        write_json(p.meta("verify.json"), {"checks": checks, "keep_tar": keep_tar})
        for name, result in checks.items():
            logger.log(logging.INFO if result["ok"] else logging.ERROR, "%s %s: %s", day, name, result)
        if not checks["C5 raw tar"]["ok"] or (not keep_tar and not checks["C8 raw rebuild"]["ok"]):
            msg = f"raw archive checks failed: C5 {checks['C5 raw tar']}, C8 {checks['C8 raw rebuild']}"
            raise CheckFailedError(msg)
        if not checks["C7 interim sanity"]["ok"]:
            msg = f"C7 failed: interim holds data the raw files lack: {checks['C7 interim sanity']}"
            raise ReviewNeededError(msg)
        return DayState.VERIFIED

    def _upload_checked(self, local: Path, final: str, tmp: str) -> dict[str, Any]:
        """Upload one file to a temporary name, read its SHA-256 back from the server (C6) and rename it.

        Args:
            local: Local file.
            final: Final remote path.
            tmp: Temporary remote path in the same directory.

        Returns:
            The C6 result for this file.

        Raises:
            ReviewNeededError: If the final path holds a different file.
            CheckFailedError: If the uploaded copy does not match after a second try.
        """
        sha = sha256_of(local)
        if self.remote.exists(final):
            if self.remote.sha256_file(final) != sha:
                msg = f"{final} already exists with different content"
                raise ReviewNeededError(msg)
            return {"path": final, "sha256": sha, "ok": True}
        for _attempt in range(2):
            self.remote.upload_file(local, tmp)
            self.checkpoint(f"upload:tmp:{local.name}")
            if self.remote.sha256_file(tmp) == sha:
                self.remote.move(tmp, final)
                return {"path": final, "sha256": sha, "ok": True}
            logger.warning("C6 mismatch for %s, uploading again", tmp)
        msg = f"C6: {tmp} does not match the local file after two uploads"
        raise CheckFailedError(msg)

    def upload(self, day: str) -> DayState:
        """Put the tar and the dt= directory on the storage box, checked (C6) and renamed into place.

        Args:
            day: The day.

        Returns:
            UPLOADED.

        Raises:
            RemoteFullError: If the storage box is too full.
            ReviewNeededError: If the final directory exists with other content.
            CheckFailedError: If C6 fails.
        """
        p = DayPaths(self.s.day_work_dir(day))
        self._guard(day, "upload", 0)
        data_files = sorted(f for f in p.out.iterdir() if not f.name.startswith("."))
        tar = p.tar_dir / raw_tar_name(day)
        upload_size = sum(f.stat().st_size for f in data_files) + (tar.stat().st_size if tar.exists() else 0)
        free = self.remote.free_bytes()
        if free - upload_size < self.s.min_remote_free_gb * GB:
            msg = f"storage box has {free / GB:.0f} GB free, minimum {self.s.min_remote_free_gb:.0f} GB: purge first"
            raise RemoteFullError(msg)
        c6: list[dict[str, Any]] = []
        raw_archive = None
        if tar.exists():
            self.remote.mkdir_p(raw_tar_dir(day))
            final = f"{raw_tar_dir(day)}/{raw_tar_name(day)}"
            c6.append(self._upload_checked(tar, final, f"{raw_tar_dir(day)}/.{raw_tar_name(day)}.tmp"))
            members = len(read_json(p.meta("source.json")))
            raw_archive = mf.raw_archive_entry(day, tar, c6[-1]["sha256"], members)
        final_dir, tmp_dir = hfp_day(day), hfp_day_tmp(day)
        local = {f.name: sha256_of(f) for f in data_files}
        if self.remote.exists(final_dir):
            server = self.remote.sha256_dir(final_dir)
            if any(server.get(n) != sha for n, sha in local.items()) or mf.MANIFEST_NAME not in server:
                msg = f"{final_dir} already exists with different content"
                raise ReviewNeededError(msg)
            return DayState.UPLOADED
        self.remote.mkdir_p(tmp_dir)
        for f in data_files:
            self.remote.upload_file(f, f"{tmp_dir}/{f.name}")
            self.checkpoint(f"upload:file:{f.name}")
        server = self.remote.sha256_dir(tmp_dir)
        for name, sha in local.items():
            c6.append({"path": f"{final_dir}/{name}", "sha256": sha, "ok": server.get(name) == sha})
        if not all(r["ok"] for r in c6):
            msg = f"C6 failed: {[r['path'] for r in c6 if not r['ok']]}"
            raise CheckFailedError(msg)
        manifest = self._manifest(day, data_files, local, raw_archive, c6)
        if not manifest["all_checks_ok"]:
            msg = f"not all checks passed: {list(manifest['checks'])}"
            raise CheckFailedError(msg)
        self._put_manifest(manifest, tmp_dir, p)
        self.checkpoint("upload:manifest")
        self.remote.move(tmp_dir, final_dir)
        return DayState.UPLOADED

    def _manifest(
        self,
        day: str,
        data_files: list[Path],
        local: dict[str, str],
        raw_archive: dict[str, Any] | None,
        c6: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the manifest from the day's local records.

        Args:
            day: The day.
            data_files: Files of the dt= directory.
            local: Their SHA-256.
            raw_archive: The tar entry or None.
            c6: Per-file C6 results.

        Returns:
            The manifest.
        """
        p = DayPaths(self.s.day_work_dir(day))
        conversion = read_json(p.meta(CONVERSION_NAME))
        verify = read_json(p.meta("verify.json"))
        checks = {
            **conversion["checks"],
            "C5 raw tar": verify["checks"]["C5 raw tar"],
            "C6 storage read-back": {
                "files": len(c6),
                "mismatched": sum(not r["ok"] for r in c6),
                "ok": all(r["ok"] for r in c6),
            },
            "C7 interim sanity": verify["checks"]["C7 interim sanity"],
            "C8 raw rebuild": verify["checks"]["C8 raw rebuild"],
        }
        files = [{"path": f.name, "size": f.stat().st_size, "sha256": local[f.name]} for f in data_files]
        return mf.build(
            day,
            read_json(p.meta("source.json")),
            read_json(p.meta("interim.json")),
            files,
            raw_archive,
            conversion["counts"],
            checks,
        )

    def _put_manifest(self, manifest: dict[str, Any], remote_dir: str, p: DayPaths) -> None:
        """Write the manifest locally and to the storage box via a temporary name, checked by SHA-256.

        Args:
            manifest: The manifest.
            remote_dir: Remote dt= directory (temporary or final).
            p: The day's local paths.

        Raises:
            CheckFailedError: If the uploaded manifest does not match.
        """
        local = p.meta(mf.MANIFEST_NAME)
        local.write_text(mf.dumps(manifest))
        tmp = f"{remote_dir}/.{mf.MANIFEST_NAME}.tmp"
        self.remote.upload_file(local, tmp)
        if self.remote.sha256_file(tmp) != sha256_of(local):
            msg = f"C6: manifest upload to {tmp} does not match"
            raise CheckFailedError(msg)
        self.remote.replace_file(tmp, f"{remote_dir}/{mf.MANIFEST_NAME}")

    def _load_manifest(self, day: str, p: DayPaths) -> dict[str, Any]:
        """Return the day's manifest, from the local copy or else from the storage box.

        Args:
            day: The day.
            p: The day's local paths.

        Returns:
            The manifest.
        """
        local = p.meta(mf.MANIFEST_NAME)
        if not local.exists():
            self.remote.download_file(f"{hfp_day(day)}/{mf.MANIFEST_NAME}", local)
        return read_json(local)

    def _swap_one(self, original: str, expected: list[dict[str, Any]]) -> None:
        """Move one original directory to _superseded/ if it is still exactly what was converted.

        Args:
            original: Original directory relative to the root.
            expected: Its files as recorded at fetch time.

        Raises:
            ReviewNeededError: If it changed, or both copies exist.
            CheckFailedError: If neither exists.
        """
        dst = superseded(original)
        want = {f["name"]: f["sha256"] for f in expected if not f.get("is_dir")}
        here, there = self.remote.exists(original), self.remote.exists(dst)
        if here and there:
            msg = f"both {original} and {dst} exist: files reappeared after the swap?"
            raise ReviewNeededError(msg)
        if not here and not there:
            msg = f"neither {original} nor {dst} exists"
            raise CheckFailedError(msg)
        current = self.remote.sha256_dir(dst if there else original)
        if current != want or any(f.get("is_dir") for f in expected):
            msg = f"{dst if there else original} differs from what was converted ({len(current)} vs {len(want)} files)"
            raise ReviewNeededError(msg)
        if here:
            self.remote.mkdir_p(dst.rsplit("/", 1)[0])
            self.remote.move(original, dst)
            logger.info("Moved %s -> %s", original, dst)

    def swap(self, day: str) -> DayState:
        """Move the day's raw and interim directories to _superseded/ and record it in the manifest.

        Args:
            day: The day.

        Returns:
            SWAPPED.
        """
        p = DayPaths(self.s.day_work_dir(day))
        self._guard(day, "swap", 0)
        manifest = self._load_manifest(day, p)
        self._swap_one(raw_day(day), manifest["source"]["files"])
        self.checkpoint("swap:raw")
        if manifest["interim"] is not None:
            self._swap_one(interim_day(day), manifest["interim"]["files"])
        self.checkpoint("swap:interim")
        paths = {"raw": superseded(raw_day(day))}
        if manifest["interim"] is not None:
            paths["interim"] = superseded(interim_day(day))
        self._put_manifest(mf.advance(manifest, "swapped", superseded=paths), hfp_day(day), p)
        return DayState.SWAPPED

    def finish(self, day: str) -> DayState:
        """Mark the manifest done and remove the local working files.

        Args:
            day: The day.

        Returns:
            DONE.
        """
        p = DayPaths(self.s.day_work_dir(day))
        self._guard(day, "finish", 0)
        manifest = self._load_manifest(day, p)
        self._put_manifest(mf.advance(manifest, "done"), hfp_day(day), p)
        shutil.rmtree(p.root, ignore_errors=True)
        return DayState.DONE

    # --- driving ----------------------------------------------------------------------------------

    STEPS: ClassVar[dict[DayState, str]] = {
        DayState.PENDING: "fetch",
        DayState.FETCHED: "convert",
        DayState.CONVERTED: "verify",
        DayState.VERIFIED: "upload",
        DayState.UPLOADED: "swap",
        DayState.SWAPPED: "finish",
    }

    def _plan(self, day: str) -> None:
        """Log what a real run would do on the storage box after a dry run reached VERIFIED.

        Args:
            day: The day.
        """
        p = DayPaths(self.s.day_work_dir(day))
        tar = p.tar_dir / raw_tar_name(day)
        if tar.exists():
            logger.info(
                "[dry-run] would upload %s (%.0f MB) to %s/", tar.name, tar.stat().st_size / 1e6, raw_tar_dir(day)
            )
        for f in sorted(p.out.iterdir()):
            logger.info("[dry-run] would upload %s (%.1f MB) to %s/", f.name, f.stat().st_size / 1e6, hfp_day(day))
        logger.info("[dry-run] would move %s and %s to _superseded/", raw_day(day), interim_day(day))

    def _info(self, day: str, state: DayState) -> dict[str, Any]:
        """Return the figures of a finished step that status and report show.

        Args:
            day: The day.
            state: The state the step reached.

        Returns:
            Values for the day's info record.
        """
        p = DayPaths(self.s.day_work_dir(day))
        if state == DayState.FETCHED:
            return {"raw_bytes": self._raw_size(day)}
        if state == DayState.CONVERTED:
            counts = read_json(p.meta(CONVERSION_NAME))["counts"]
            return {
                "rows": counts["rows"],
                "rejected": counts["rejected"],
                "payload_raw": counts["rows_with_payload_raw"],
            }
        if state == DayState.VERIFIED:
            v = read_json(p.meta("verify.json"))
            c8 = v["checks"]["C8 raw rebuild"]
            return {"c8_ok": c8["ok"], "tar": v["keep_tar"], "tar_reason": "" if c8["ok"] else "C8 failed: tar kept"}
        return {}

    def run_day(self, day: str, dry_run: bool = False) -> DayState:
        """Advance one day as far as it goes.

        Args:
            day: The day.
            dry_run: Stop after VERIFIED and only log the storage box changes.

        Returns:
            The state the day ended in.
        """
        state = self.db.ensure(day).state
        while state in self.STEPS:
            if dry_run and state == DayState.VERIFIED:
                self._plan(day)
                break
            step = self.STEPS[state]
            started = time.monotonic()
            self.db.bump_attempts(day)
            state = getattr(self, step)(day)
            self.db.set_state(day, state, f"{step} took {time.monotonic() - started:.0f} s", **self._info(day, state))
            self.checkpoint(f"state:{state}")
        return state

    def run(self, days: list[str], dry_run: bool = False, max_consecutive_failures: int = 3) -> RunSummary:
        """Advance many days in order, stopping cleanly on a signal or a problem that affects every day.

        Problems of one day (a failed check, an unexpected file, a conflict) put that day in FAILED or
        NEEDS_REVIEW and the run moves on; several failures in a row stop the run, since they point to a
        common cause.

        Args:
            days: Days to process, in order.
            dry_run: Stop every day after VERIFIED.
            max_consecutive_failures: Failures in a row that stop the run.

        Returns:
            What happened.
        """
        summary = RunSummary()
        failures = 0
        for day in days:
            try:
                state = self.run_day(day, dry_run)
            except StopRequestedError:
                summary.stopped_reason = "stop requested"
                break
            except (ReviewNeededError, CheckFailedError) as exc:
                stuck = self.db.ensure(day).state
                new = DayState.NEEDS_REVIEW if isinstance(exc, ReviewNeededError) else DayState.FAILED
                self.db.set_state(day, new, str(exc), resume_from=str(stuck))
                logger.error("%s -> %s: %s", day, new, exc)
                summary.finished[day] = new
                failures += 1
            except (RemoteFullError, LocalSpaceError) as exc:
                logger.error("Stopping: %s", exc)
                summary.stopped_reason, summary.exit_code = str(exc), 5
                break
            except SentinelMissingError, RemoteUnavailableError:
                raise  # the storage box itself is gone or wrong: nothing else can run either
            except Exception as exc:
                # Unknown errors (a strict cast, a bug, a server error) fail the day loudly; the state it
                # failed in is kept so that retry resumes there.
                stuck = self.db.ensure(day).state
                self.db.set_state(day, DayState.FAILED, f"{type(exc).__name__}: {exc}", resume_from=str(stuck))
                logger.exception("%s failed in state %s", day, stuck)
                summary.finished[day] = DayState.FAILED
                failures += 1
            else:
                summary.finished[day] = state
                failures = 0
                logger.info("%s: %s", day, state)
            if failures >= max_consecutive_failures:
                summary.stopped_reason, summary.exit_code = f"{failures} failures in a row", 6
                logger.error("Stopping after %d failures in a row", failures)
                break
        return summary

    def retry(self, day: str) -> None:
        """Put a FAILED or NEEDS_REVIEW day back into the state it stopped in, so the next run resumes it.

        Args:
            day: The day.
        """
        record = self.db.get(day)
        if record is None or record.state not in {DayState.FAILED, DayState.NEEDS_REVIEW}:
            return
        resume = DayState(record.info.get("resume_from", DayState.PENDING))
        self.db.set_state(day, resume, f"retry from {record.state}")

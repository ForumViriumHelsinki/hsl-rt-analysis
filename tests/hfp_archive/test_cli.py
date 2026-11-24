"""Smoke tests of the command line against the fake storage box."""

from collections.abc import Callable
from pathlib import Path

import pytest

from hsl_rt_analysis.hfp_archive.cli import main
from hsl_rt_analysis.hfp_archive.state import DayState, StateDB
from tests.hfp_archive.conftest import DAY, Box


def cli(box: Box, *args: str) -> int:
    """Run the CLI with the box's settings and return its exit code.

    Args:
        box: The fake storage box.
        *args: Command and options.

    Returns:
        Exit code.
    """
    common = ["--remote", "local", "--state-dir", str(box.settings.state_dir), "--work-dir",
              str(box.settings.work_dir), "--memory-limit", "1GB", "--threads", "2"]  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        main([args[0], *common, *args[1:]], remote=box.remote)
    return int(exc.value.code or 0)


def test_cli_end_to_end(make_box: Callable[..., Box], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """dry run, real run, status, report, audit, purge (dry and real) and restore work from the CLI."""
    box = make_box()
    box.db.close()
    assert cli(box, "run", "--days", DAY, "--dry-run", "--cutoff-days", "0", "--min-remote-free-gb", "0") == 0
    assert StateDB(box.settings.state_db).require(DAY).state == DayState.VERIFIED
    assert cli(box, "run", "--days", DAY, "--cutoff-days", "0", "--min-remote-free-gb", "0") == 0
    assert StateDB(box.settings.state_db).require(DAY).state == DayState.DONE
    assert cli(box, "status", "--all") == 0
    assert DAY in capsys.readouterr().out
    assert cli(box, "report") == 0
    assert cli(box, "purge-superseded", "--before", "2026-02-01") == 0
    assert (box.root / "_superseded" / "raw" / "all" / DAY).exists()  # blocked: no audit yet
    assert cli(box, "audit", "--days", DAY) == 0
    assert cli(box, "purge-superseded", "--before", "2026-02-01") == 0
    assert (box.root / "_superseded" / "raw" / "all" / DAY).exists()  # dry run by default
    assert cli(box, "purge-superseded", "--before", "2026-02-01", "--execute") == 0
    assert not (box.root / "_superseded" / "raw" / "all" / DAY).exists()
    assert cli(box, "restore", DAY, "--out", str(tmp_path / "restored")) == 0
    assert sorted(p.name for p in (tmp_path / "restored" / DAY).iterdir()) == sorted(box.raw_before)


def test_cli_stops_without_sentinel(make_box: Callable[..., Box]) -> None:
    """A missing sentinel ends the run with its own exit code and changes nothing."""
    box = make_box()
    (box.root / ".hfp-archive-sentinel").unlink()
    assert cli(box, "run", "--cutoff-days", "0") == 3

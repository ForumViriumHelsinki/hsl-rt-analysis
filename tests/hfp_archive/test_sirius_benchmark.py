"""The Sirius benchmark script, run on the CPU over a converted synthetic day."""

import importlib.util
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from hsl_rt_analysis.hfp_archive.pipeline import Pipeline
from tests.hfp_archive.conftest import DAY, Box


SCRIPT = Path(__file__).parents[2] / "exploration" / "sirius_benchmark.py"


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    """Import the standalone script as a module.

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location("sirius_benchmark", SCRIPT)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while the class is built
    spec.loader.exec_module(module)
    return module


def test_cpu_run_over_a_converted_day(
    bench: ModuleType, make_box: Callable[..., Box], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    box = make_box()
    Pipeline(box.settings, box.remote, box.db).run([DAY])
    out = tmp_path / "results.json"
    bench.main(["--data", str(box.root / "hsl" / "hfp"), "--db", str(tmp_path / "b.duckdb"), "--engines", "cpu",
                "--repeat", "2", "--output", str(out)])  # fmt: skip
    results = json.loads(out.read_text())["results"]
    expected = {(q, s) for q in bench.QUERIES for s in ("parquet", "table")}
    assert {(r["query"], r["source"]) for r in results} == expected
    assert all(r["error"] is None and len(r["runs_s"]) == 2 for r in results), results
    counts = {r["source"]: r["rows"] for r in results if r["query"] == "event_counts"}
    assert counts["parquet"] == counts["table"] > 0
    assert "event_counts" in caplog.text


def test_same_rows(bench: ModuleType) -> None:
    assert bench.same_rows([("a", 1, 0.1 + 0.2)], [("a", 1, 0.3)])
    assert not bench.same_rows([("a", 1, 0.3)], [("a", 1, 0.31)])
    assert not bench.same_rows([("a", 1)], [("a", 2)])
    assert not bench.same_rows([("a", 1)], [])


def test_bad_arguments(bench: ModuleType, tmp_path: Path) -> None:
    assert bench.parse_settings(["a_b=1"]) == {"a_b": "1"}
    with pytest.raises(Exception, match="KEY=VALUE"):
        bench.parse_settings(["nope"])
    with pytest.raises(SystemExit, match="no part-"):
        bench.main(["--data", str(tmp_path), "--engines", "cpu"])
    with pytest.raises(SystemExit, match="unknown queries"):
        bench.main(["--data", str(tmp_path), "--queries", "nope"])

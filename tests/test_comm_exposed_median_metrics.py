"""Tests for the median-based comm exposure view (sec 17 v2, judge batch 5, full input+comm loop)."""
import sys
import types
from pathlib import Path

import pandas as pd
import importlib.util

_ANALYZER_DIR = Path(__file__).resolve().parents[1] / "python" / "dftracer" / "analyzer"

_dftracer_pkg = types.ModuleType("dftracer")
_dftracer_pkg.__path__ = [str(_ANALYZER_DIR.parent)]
sys.modules["dftracer"] = _dftracer_pkg
_analyzer_pkg = types.ModuleType("dftracer.analyzer")
_analyzer_pkg.__path__ = [str(_ANALYZER_DIR)]
sys.modules["dftracer.analyzer"] = _analyzer_pkg

_spec = importlib.util.spec_from_file_location(
    "dftracer.analyzer.metrics", _ANALYZER_DIR / "metrics.py",
    submodule_search_locations=[str(_ANALYZER_DIR)],
)
metrics = importlib.util.module_from_spec(_spec)
sys.modules["dftracer.analyzer.metrics"] = metrics
_spec.loader.exec_module(metrics)

attach_comm_exposed_median_metrics = metrics.attach_comm_exposed_median_metrics


def _window_view(index):
    return pd.DataFrame(index=pd.Index(index, name="window"))


def _comm_row(window, step, pid, time_val):
    return {"window": window, "step": step, "pid": pid, "time": time_val,
            "cat": "comm", "func_name": "comm.exposed"}


def _fetch_row(window, step, pid, time_val=0.1):
    """The authoritative step-population tag (sec 12/16's own family) -- every test that expects a
    real (non-NA) reduction must tag every step it wants counted, since a step present only in
    `comm.exposed` and absent from `fetch_iter` is not part of the population."""
    return {"window": window, "step": step, "pid": pid, "time": time_val,
            "cat": "dataloader", "func_name": "fetch.iter"}


def test_median_of_per_step_max_times_step_count():
    # Steps 1..5, two ranks each. Per-step max: [5, 1, 9, 2, 3]. median([5,1,9,2,3]) = 3 -> 3*5=15.
    rows = []
    per_step_vals = {1: (5, 1), 2: (1, 0.5), 3: (9, 4), 4: (2, 1), 5: (3, 0.1)}
    for s, (a, b) in per_step_vals.items():
        rows.append(_comm_row(1, s, 0, a))
        rows.append(_comm_row(1, s, 1, b))
        rows.append(_fetch_row(1, s, 0))
    traces = pd.DataFrame(rows)
    out = attach_comm_exposed_median_metrics({("window",): _window_view([1])}, traces)
    row = out[("window",)].loc[1]
    assert row["domain_comm_pure_exposure_time_v2"] == 15.0
    assert row["domain_comm_exposure_step_nunique"] == 5


def test_median_is_robust_to_a_single_spike_step():
    # 9 normal steps at max=1.0, one spike step at max=100.0: median stays 1.0 (not dragged by the
    # spike the way a sum/mean would be) -- the exact problem statement this design fixes.
    rows = [_comm_row(1, s, 0, 1.0) for s in range(1, 10)]
    rows.append(_comm_row(1, 10, 0, 100.0))
    rows += [_fetch_row(1, s, 0) for s in range(1, 11)]
    traces = pd.DataFrame(rows)
    out = attach_comm_exposed_median_metrics({("window",): _window_view([1])}, traces)
    row = out[("window",)].loc[1]
    assert row["domain_comm_pure_exposure_time_v2"] == 10.0  # median=1.0 * 10 steps
    # the trimmed-mean alternate is reported too, and differs (a spike still pulls it up somewhat).
    assert row["domain_comm_exposure_trimmed10_time_v2"] is not None


def test_hidden_steps_are_zero_filled_from_the_fetch_iter_step_population():
    """The orchestrator's exact fix: a step tagged by fetch_iter but with NO comm.exposed row is
    fully hidden and counts as 0, not as missing -- half the window's steps hidden should visibly
    pull the median down, not just shrink step_nunique."""
    # 10 steps total (fetch_iter tags all of them). Only steps 1-5 have a comm.exposed row (value 4.0
    # each); steps 6-10 are fully hidden (no comm row at all). Zero-filled per-step values:
    # [4,4,4,4,4,0,0,0,0,0] -> median = 2.0 -> domain value = 2.0 * 10 = 20.0.
    rows = [_fetch_row(1, s, 0) for s in range(1, 11)]
    rows += [_comm_row(1, s, 0, 4.0) for s in range(1, 6)]
    traces = pd.DataFrame(rows)
    out = attach_comm_exposed_median_metrics({("window",): _window_view([1])}, traces)
    row = out[("window",)].loc[1]
    assert row["domain_comm_exposure_step_nunique"] == 10          # the fetch population, not 5
    assert row["domain_comm_pure_exposure_time_v2"] == 20.0        # median(zero-filled) * 10


def test_no_fetch_iter_tags_at_all_leaves_the_fact_na_even_with_comm_rows_present():
    """No authoritative step population exists for this window -- suppression, never a fabricated
    population (and never a fabricated zero either, despite comm rows being present)."""
    traces = pd.DataFrame([_comm_row(1, 1, 0, 5.0)])   # comm row present, but NO fetch_iter row
    out = attach_comm_exposed_median_metrics({("window",): _window_view([1])}, traces)
    assert pd.isna(out[("window",)].loc[1, "domain_comm_pure_exposure_time_v2"])


def test_window_absent_is_a_noop():
    traces = pd.DataFrame([_comm_row(1, 1, 0, 1.0), _fetch_row(1, 1, 0)])
    assert attach_comm_exposed_median_metrics({}, traces) == {}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

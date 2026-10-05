"""Tests for the TOTAL-comm median view (sec 17 v3, orchestrator decision 2026-09-27): domain2's
exposed-comm validation found exposed comm tiny and HIGHER at cap400 while TOTAL comm was much
larger and lower at cap400 -- Step 0c's gain looks like comm/compute interference, not an
exposed-tail effect. This mirrors test_comm_exposed_median_metrics.py's shape exactly, applied to
the `comm.total` family, to confirm the shared `_comm_median_metrics` reduction (refactored out of
the v2 function) behaves identically for the new family and does not cross-contaminate it.
"""
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

attach_comm_total_median_metrics = metrics.attach_comm_total_median_metrics
attach_comm_exposed_median_metrics = metrics.attach_comm_exposed_median_metrics


def _window_view(index):
    return pd.DataFrame(index=pd.Index(index, name="window"))


def _total_row(window, step, pid, time_val):
    return {"window": window, "step": step, "pid": pid, "time": time_val,
            "cat": "comm", "func_name": "comm.total"}


def _exposed_row(window, step, pid, time_val):
    return {"window": window, "step": step, "pid": pid, "time": time_val,
            "cat": "comm", "func_name": "comm.exposed"}


def _fetch_row(window, step, pid, time_val=0.1):
    return {"window": window, "step": step, "pid": pid, "time": time_val,
            "cat": "dataloader", "func_name": "fetch.iter"}


def test_median_of_per_step_max_times_step_count():
    rows = []
    per_step_vals = {1: (50, 10), 2: (10, 5), 3: (90, 40)}
    for s, (a, b) in per_step_vals.items():
        rows.append(_total_row(1, s, 0, a))
        rows.append(_total_row(1, s, 1, b))
        rows.append(_fetch_row(1, s, 0))
    traces = pd.DataFrame(rows)
    out = attach_comm_total_median_metrics({("window",): _window_view([1])}, traces)
    row = out[("window",)].loc[1]
    # per-step max: [50, 10, 90] -> median = 50 -> 50 * 3 = 150.
    assert row["domain_comm_total_median_time_v3"] == 150.0
    assert row["domain_comm_total_step_nunique"] == 3


def test_hidden_steps_are_zero_filled_from_the_fetch_iter_step_population():
    rows = [_fetch_row(1, s, 0) for s in range(1, 11)]
    rows += [_total_row(1, s, 0, 40.0) for s in range(1, 6)]
    traces = pd.DataFrame(rows)
    out = attach_comm_total_median_metrics({("window",): _window_view([1])}, traces)
    row = out[("window",)].loc[1]
    assert row["domain_comm_total_step_nunique"] == 10
    # sorted zero-filled population: [0]*5 + [40]*5 -> median (average of the two middle values,
    # even-length) = (0 + 40) / 2 = 20 -> domain value = 20 * 10 = 200.
    assert row["domain_comm_total_median_time_v3"] == 200.0


def test_no_fetch_iter_tags_leaves_the_fact_na():
    traces = pd.DataFrame([_total_row(1, 1, 0, 5.0)])
    out = attach_comm_total_median_metrics({("window",): _window_view([1])}, traces)
    assert pd.isna(out[("window",)].loc[1, "domain_comm_total_median_time_v3"])


def test_total_family_does_not_cross_contaminate_exposed_family():
    """A window with BOTH comm.total and comm.exposed rows must keep the two reductions distinct --
    a shared groupby key or column name would silently merge them."""
    rows = [_fetch_row(1, s, 0) for s in (1, 2)]
    rows += [_total_row(1, 1, 0, 100.0), _total_row(1, 2, 0, 100.0)]
    rows += [_exposed_row(1, 1, 0, 3.0), _exposed_row(1, 2, 0, 3.0)]
    traces = pd.DataFrame(rows)
    flat_views = {("window",): _window_view([1])}
    flat_views = attach_comm_exposed_median_metrics(flat_views, traces)
    flat_views = attach_comm_total_median_metrics(flat_views, traces)
    row = flat_views[("window",)].loc[1]
    assert row["domain_comm_pure_exposure_time_v2"] == 6.0     # median(3,3)*2
    assert row["domain_comm_total_median_time_v3"] == 200.0    # median(100,100)*2


def test_window_absent_is_a_noop():
    traces = pd.DataFrame([_total_row(1, 1, 0, 1.0), _fetch_row(1, 1, 0)])
    assert attach_comm_total_median_metrics({}, traces) == {}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

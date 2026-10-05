"""Tests for the declared (window, step) sum-of-max input exposure view (E1, STORMER-LOOP-DESIGN sec 11/12/14).

Standalone: imports metrics.py directly by path so it runs without a full package install.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd

# The pinned venv already has an installed `dftracer.analyzer.metrics` (a different, unrelated
# copy), so a plain sys.path/PYTHONPATH prepend does not win. Register a synthetic
# `dftracer`/`dftracer.analyzer` package rooted at THIS sandbox's source directory so metrics.py's
# own `from .types import ...` resolves against the sandbox's sibling files, not site-packages.
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

attach_input_step_max_sum_metrics = metrics.attach_input_step_max_sum_metrics


def _window_view(index):
    return pd.DataFrame(index=pd.Index(index, name="window"))


def _fetch_row(window, step, pid, time):
    return {
        "window": window, "step": step, "pid": pid, "time": time,
        "cat": "dataloader", "func_name": "fetch.iter",
    }


def test_sum_of_max_matches_hand_computation():
    # Window 1 has two steps.  Step 1: ranks 0,1,2 wait 1,5,2 ms -> max=5.  Step 2: ranks wait
    # 3,1,1 -> max=3.  Sum-of-max for the window = 5+3 = 8, over 2 distinct steps.
    rows = [
        _fetch_row(1, 1, 0, 1.0), _fetch_row(1, 1, 1, 5.0), _fetch_row(1, 1, 2, 2.0),
        _fetch_row(1, 2, 0, 3.0), _fetch_row(1, 2, 1, 1.0), _fetch_row(1, 2, 2, 1.0),
    ]
    traces = pd.DataFrame(rows)
    flat_views = {("window",): _window_view([1])}
    out = attach_input_step_max_sum_metrics(flat_views, traces)
    row = out[("window",)].loc[1]
    assert row["domain_input_step_max_sum_time"] == 8.0
    assert row["domain_input_step_nunique"] == 2
    # The declared (window, step) view is also published, not just the reduced scalar.
    assert ("window", "step") in out
    per_step = out[("window", "step")]
    assert per_step.loc[(1, 1, 1), "fetch_iter_step_time_sum"] == 5.0


def test_non_fetch_rows_are_excluded_from_the_family():
    rows = [
        _fetch_row(1, 1, 0, 10.0),
        {"window": 1, "step": 1, "pid": 1, "time": 999.0, "cat": "comm", "func_name": "all_reduce"},
    ]
    traces = pd.DataFrame(rows)
    flat_views = {("window",): _window_view([1])}
    out = attach_input_step_max_sum_metrics(flat_views, traces)
    assert out[("window",)].loc[1, "domain_input_step_max_sum_time"] == 10.0


def test_missing_step_suppresses_rather_than_zeros():
    # No 'step' column at all on fetch_iter rows (pre-tag data / an app that never sets it):
    # the project rule is that missing normalization/attribution suppresses, never reads as zero.
    rows = [{"window": 1, "pid": 0, "time": 10.0, "cat": "dataloader", "func_name": "fetch.iter"}]
    traces = pd.DataFrame(rows)  # no "step" column
    flat_views = {("window",): _window_view([1])}
    out = attach_input_step_max_sum_metrics(flat_views, traces)
    assert pd.isna(out[("window",)].loc[1, "domain_input_step_max_sum_time"])


def test_rotating_straggler_is_not_diluted_like_a_window_median_would_be():
    # Same shape as the rotation finding (sec 11.2): a different rank is the argmax each step,
    # so no single rank's window total stands out, but sum-of-max still sees the full exposure.
    rows = [
        _fetch_row(1, s, s % 3, 100.0) if s % 3 == 0 else
        _fetch_row(1, s, s % 3, 100.0)
        for s in range(1, 10)
    ]
    # Every step has exactly one row (the rotating straggler) at 100ms; the other two ranks are
    # implicitly silent (no fetch.iter row that step -- e.g. they didn't wait).  Sum-of-max must
    # still total 900 (9 steps x 100ms), not something diluted by an all-rank average.
    traces = pd.DataFrame(rows)
    flat_views = {("window",): _window_view([1])}
    out = attach_input_step_max_sum_metrics(flat_views, traces)
    assert out[("window",)].loc[1, "domain_input_step_max_sum_time"] == 900.0
    assert out[("window",)].loc[1, "domain_input_step_nunique"] == 9


def test_window_absent_is_a_noop():
    traces = pd.DataFrame([_fetch_row(1, 1, 0, 1.0)])
    assert attach_input_step_max_sum_metrics({}, traces) == {}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

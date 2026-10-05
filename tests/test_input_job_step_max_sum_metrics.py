"""Tests for the cross-node job-wide sum-of-max view (sec 16, judge batch 5).

Standalone import, same technique as test_input_step_max_sum_metrics.py.
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

attach_input_job_step_max_sum_metrics = metrics.attach_input_job_step_max_sum_metrics


def _window_view(index):
    return pd.DataFrame(index=pd.Index(index, name="window"))


def _job_row(window, time_val, step_count):
    return {
        "window": window, "time": time_val, "step_count": step_count,
        "cat": "input_exposure_job", "func_name": "input_step_max_sum_job",
    }


def test_takes_max_not_sum_across_redundant_per_rank_rows():
    # Four ranks on one node all emit the SAME job-wide value that window (by construction, sec
    # 16.3) -- the reduction here must be idempotent (.max()), not .sum(), or it would multiply
    # the true value by the local rank count.
    rows = [_job_row(1, 4.2, 9) for _ in range(4)]
    traces = pd.DataFrame(rows)
    out = attach_input_job_step_max_sum_metrics({("window",): _window_view([1])}, traces)
    assert out[("window",)].loc[1, "domain_input_job_step_max_sum_time"] == 4.2
    assert out[("window",)].loc[1, "domain_input_job_step_count"] == 9


def test_absent_family_leaves_columns_missing_not_zero():
    traces = pd.DataFrame([{"window": 1, "time": 1.0, "cat": "dataloader", "func_name": "fetch.iter"}])
    out = attach_input_job_step_max_sum_metrics({("window",): _window_view([1])}, traces)
    assert pd.isna(out[("window",)].loc[1, "domain_input_job_step_max_sum_time"])


def test_window_absent_is_a_noop():
    traces = pd.DataFrame([_job_row(1, 1.0, 3)])
    assert attach_input_job_step_max_sum_metrics({}, traces) == {}


def test_missing_step_count_column_still_gives_the_time_value():
    # An older writer that emits the family without the step_count int_arg: time still attaches.
    rows = [{"window": 1, "time": 2.5, "cat": "input_exposure_job", "func_name": "input_step_max_sum_job"}]
    traces = pd.DataFrame(rows)
    out = attach_input_job_step_max_sum_metrics({("window",): _window_view([1])}, traces)
    assert out[("window",)].loc[1, "domain_input_job_step_max_sum_time"] == 2.5
    assert pd.isna(out[("window",)].loc[1, "domain_input_job_step_count"])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

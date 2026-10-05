"""Golden replay for native control-window grouping.

This fixture is deliberately small, but it represents two completed control
windows across a real workload epoch/step transition.  It contains a rank main
thread, a same-rank helper thread, and a data-loader worker.  The compatibility
path is frozen as a baseline; the native path may change only the grouping key,
not event membership or ordinary aggregates.
"""

import sys
import json
from pathlib import Path

import pandas as pd

import dftracer

_SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "python" / "dftracer"
if str(_SOURCE_PACKAGE) not in dftracer.__path__:
    dftracer.__path__.insert(0, str(_SOURCE_PACKAGE))

from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.fastpath import FastPathPlan
from dftracer.analyzer.metrics import attach_window_integrity_metrics
from dftracer.analyzer.types import AnalysisFact, AnalysisResult, FactScope, FactSeverity, FactWindow


class _LedgerAnalyzer(Analyzer):
    def read_trace(self, *args, **kwargs):
        raise NotImplementedError


def _plan(view_types):
    plan = FastPathPlan.__new__(FastPathPlan)
    plan.view_types = list(view_types)
    plan.terms = [
        ("window_wall", 'func_name == "window.stop"', False),
        ("compute", 'func_name == "compute"', False),
        ("comm", 'func_name == "comm"', False),
        ("reader_posix", 'func_name == "read"', False),
    ]
    plan._mask_cache = {}
    return plan


def _event(event_id, control_window, epoch, step, pid, proc_name, func_name, time):
    event = {
        "event_id": event_id,
        "pid": pid,
        "proc_name": proc_name,
        "cat": "window" if func_name == "window.stop" else "work",
        "io_cat": "",
        "func_name": func_name,
        "time": time,
        "size": 0,
    }
    if epoch is not None:
        event["epoch"] = epoch
    if step is not None:
        event["step"] = step
    return control_window, event


_TWO_WINDOW_REPLAY = [
    # Control window 9 belongs to workload epoch/step 3/17.
    _event("w9-main-compute", 9, 3, 17, 100, "rank0-main", "compute", 4.0),
    _event("w9-helper-comm", 9, 3, 17, 100, "rank0-helper", "comm", 1.0),
    _event("w9-worker-read", 9, None, None, 200, "loader-worker", "read", 2.0),
    _event("w9-stop", 9, 3, 17, 100, "rank0-main", "window.stop", 10.0),
    # Control window 10 crosses to the next genuine workload epoch/step.
    _event("w10-main-compute", 10, 4, 18, 100, "rank0-main", "compute", 5.0),
    _event("w10-helper-comm", 10, 4, 18, 100, "rank0-helper", "comm", 1.5),
    _event("w10-worker-read", 10, None, None, 200, "loader-worker", "read", 2.5),
    _event("w10-stop", 10, 4, 18, 100, "rank0-main", "window.stop", 11.0),
]


def _replay(native_window):
    attached = [
        Analyzer._attach_stream_window(event, control_window, native_window=native_window)
        for control_window, event in _TWO_WINDOW_REPLAY
    ]
    view_key = "window" if native_window else "epoch"
    view = _plan([view_key]).flat_views(
        pd.DataFrame(attached), control_pids=[100]
    )[(view_key,)]
    return attached, view


def test_native_window_replay_preserves_membership_metadata_and_ordinary_metrics():
    compatibility_events, compatibility = _replay(native_window=False)
    native_events, native = _replay(native_window=True)

    # All original records survive, and control-window membership is identical.
    assert [event["event_id"] for event in native_events] == [
        event["event_id"] for event in compatibility_events
    ]
    assert [event["window"] for event in native_events] == [9, 9, 9, 9, 10, 10, 10, 10]

    # Native grouping preserves genuine application metadata rather than aliasing
    # it to the control window.  The worker remains honestly unannotated.
    assert [(event.get("epoch"), event.get("step")) for event in native_events] == [
        (3, 17), (3, 17), (None, None), (3, 17),
        (4, 18), (4, 18), (None, None), (4, 18),
    ]
    assert {(event["epoch"], event["step"]) for event in compatibility_events} == {(9, 9), (10, 10)}

    # The compatibility view's alias keys and the native view's actual window
    # keys describe the same two control windows.  Ordinary aggregates and the
    # existing coherence outcome must not change merely because the key changes.
    assert list(compatibility.index) == [9, 10]
    assert list(native.index) == [9, 10]
    for column in (
        "compute_time_sum",
        "comm_time_sum",
        "reader_posix_time_sum",
        "window_wall_time_sum",
        "domain_coherent",
    ):
        pd.testing.assert_series_equal(
            compatibility[column], native[column], check_names=False
        )
    # Only application rows with genuine epoch/step values contribute; the
    # worker's absent step is not invented or counted.
    assert native.loc[9, "epoch_nunique"] == 1
    assert native.loc[9, "step_nunique"] == 1
    assert native.loc[9, "step_min"] == 17
    assert native.loc[9, "step_max"] == 17
    assert native.loc[10, "epoch_nunique"] == 1
    assert native.loc[10, "step_nunique"] == 1
    assert native.loc[10, "step_min"] == 18
    assert native.loc[10, "step_max"] == 18


def test_native_window_proc_name_view_is_diagnostic_not_control_identity():
    native_events, _ = _replay(native_window=True)
    views = _plan(["window", "proc_name"]).flat_views(
        pd.DataFrame(native_events), control_pids=[100]
    )

    assert list(views[("window",)].index) == [9, 10]
    # Three sources exist in the preserved input.  proc_name is a diagnostic
    # decomposition, not a replacement for the one-row-per-window control view.
    assert set(views[("proc_name",)].index) == {
        "rank0-main", "rank0-helper", "loader-worker"
    }


def test_fastpath_matrix_pair_codes_do_not_overflow_at_132():
    """Categorical codes are narrow; their arithmetic index must not be."""
    traces = pd.DataFrame([
        {
            "window": window, "pid": pid, "proc_name": f"rank-{pid}",
            "cat": "work", "io_cat": "", "func_name": "compute",
            "time": 1.0, "size": 0,
        }
        for window in range(34)
        for pid in range(4)
    ])

    view = _plan(["window"]).flat_views(traces)[("window",)]
    assert len(view) == 34
    assert view.loc[33, "compute_time_sum"] == 4.0


def test_native_window_without_app_step_metadata_is_missing_not_fabricated():
    _, event = _event("worker-only", 11, None, None, 200, "loader-worker", "read", 2.0)
    _, stop = _event("stop", 11, None, None, 200, "loader-worker", "window.stop", 3.0)
    view = _plan(["window"]).flat_views(pd.DataFrame([
        Analyzer._attach_stream_window(event, 11, native_window=True),
        Analyzer._attach_stream_window(stop, 11, native_window=True),
    ]), control_pids=[200])[("window",)]
    assert all(pd.isna(view.loc[11, name]) for name in
               ("epoch_nunique", "step_nunique", "step_min", "step_max"))


def test_control_integrity_ignores_worker_epoch_but_suppresses_rank_epoch_mix():
    views = {("window",): pd.DataFrame(index=pd.Index([2], name="window"))}
    traces = pd.DataFrame([
        # Delayed workers are observational, never control-identity evidence.
        {"window": 2, "pid": 200, "epoch": 1, "step": 1},
        {"window": 2, "pid": 200, "epoch": 1, "step": 2},
        # The rank/control path itself crosses epochs, so it must suppress.
        {"window": 2, "pid": 100, "epoch": 1, "step": 3},
        {"window": 2, "pid": 101, "epoch": 2, "step": 3},
    ])
    view = attach_window_integrity_metrics(
        views, traces, control_pids=[100, 101]
    )[("window",)]
    assert view.loc[2, "epoch_nunique"] == 2
    assert view.loc[2, "step_nunique"] == 1
    assert view.loc[2, "step_min"] == 3
    assert view.loc[2, "step_max"] == 3


def test_window_integrity_ledger_marks_only_genuine_epoch_step_rows(tmp_path):
    analyzer = _LedgerAnalyzer.__new__(_LedgerAnalyzer)
    analyzer.debug = True
    analyzer.debug_output_dir = str(tmp_path)
    analyzer._debug_window_provenance = {
        "window_index": 12,
        "rank_pids": [100],
        "boundary_ts_ns_by_pid": {100: 200},
    }
    traces = pd.DataFrame([
        {"window": 12, "pid": 100, "proc_name": "rank", "func_name": "compute",
         "epoch": 4, "step": 9, "time_start": 1, "time_end": 2},
        {"window": 12, "pid": 200, "proc_name": "worker", "func_name": "read",
         "epoch": 3, "step": 8, "time_start": 2, "time_end": 3},
    ])

    analyzer._write_debug_window_integrity_ledger(traces)

    ledger = pd.read_csv(tmp_path / "window_integrity_ledger_window_000012.csv")
    assert ledger["__liveflow_analysis_window"].tolist() == [12, 12]
    assert ledger["__liveflow_is_rank_pid"].tolist() == [True, False]
    assert ledger["__liveflow_integrity_contributes"].tolist() == [True, False]


def test_debug_fact_envelope_preserves_selected_worker_and_native_window(tmp_path):
    analyzer = _LedgerAnalyzer.__new__(_LedgerAnalyzer)
    analyzer.debug = True
    analyzer.debug_output_dir = str(tmp_path)
    analyzer._debug_window_provenance = {"window_index": 12}
    fact = AnalysisFact(
        fact_type="operation_family_time_disproportion",
        window=FactWindow(view_type="proc_name", window=12),
        scope=FactScope(layer="reader_posix", entity="reader_posix_read.worker_pool"),
        evidence={"metrics": {"selected_worker_pid": 200, "reader_posix_read_ops_slope": 0.7}},
        severity=FactSeverity(score=0.3, label="medium"),
    )
    result = AnalysisResult(
        _hlms={}, _main_views={}, _metric_boundaries={}, additional_metrics={},
        checkpoint_dir=str(tmp_path), flat_views={}, layers=["reader_posix"],
        raw_stats={}, view_types=["window", "proc_name"], views={},
        analysis_facts={("proc_name",): [fact]},
    )
    analyzer._write_debug_fact_envelope(result)
    payload = json.loads((tmp_path / "analysis_facts_window_000012.json").read_text())
    assert payload["facts"][0]["window"]["window"] == 12
    assert payload["facts"][0]["evidence"]["metrics"]["selected_worker_pid"] == 200

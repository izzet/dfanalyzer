"""Golden replay for native control-window grouping.

This fixture is deliberately small, but it represents two completed control
windows across a real workload epoch/step transition.  It contains a rank main
thread, a same-rank helper thread, and a data-loader worker.  The compatibility
path is frozen as a baseline; the native path may change only the grouping key,
not event membership or ordinary aggregates.
"""

import sys
from pathlib import Path

import pandas as pd

import dftracer

_SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "python" / "dftracer"
if str(_SOURCE_PACKAGE) not in dftracer.__path__:
    dftracer.__path__.insert(0, str(_SOURCE_PACKAGE))

from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.fastpath import FastPathPlan


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
    view = _plan([view_key]).flat_views(pd.DataFrame(attached))[(view_key,)]
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


def test_native_window_proc_name_view_is_diagnostic_not_control_identity():
    native_events, _ = _replay(native_window=True)
    views = _plan(["window", "proc_name"]).flat_views(pd.DataFrame(native_events))

    assert list(views[("window",)].index) == [9, 10]
    # Three sources exist in the preserved input.  proc_name is a diagnostic
    # decomposition, not a replacement for the one-row-per-window control view.
    assert set(views[("proc_name",)].index) == {
        "rank0-main", "rank0-helper", "loader-worker"
    }

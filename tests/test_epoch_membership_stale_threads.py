"""Window epoch membership comes from the control rank's own (main-thread) events.

Regression for the 2026-09-26 finding: on the Corona TF cell every `TFReader._parse_image` event (tf.data pool thread of
the rank process) carried a stale `epoch=1` in epochs 2-3, making `epoch_nunique == 2` and silently suppressing every
fact after epoch 1.  The single-epoch gate itself must stay strict.
"""
from pathlib import Path

import pandas as pd

import dftracer
_AP = str(Path(__file__).resolve().parents[1] / "python/dftracer")
if _AP not in list(dftracer.__path__):
    try:
        dftracer.__path__.insert(0, _AP)
    except AttributeError:
        dftracer.__path__ = [_AP] + list(dftracer.__path__)

from dftracer.analyzer.metrics import attach_window_integrity_metrics, authoritative_control_rows

PID = 100


def _views(windows):
    return {("window",): pd.DataFrame(index=pd.Index(list(windows), name="window"))}


def _row(window, tid, epoch, step, func="fetch.iter", pid=PID):
    return {"window": window, "pid": pid, "tid": tid, "proc_name": "app#h#%d#%d" % (pid, tid),
            "func_name": func, "epoch": epoch, "step": step}


def _epoch_stats(rows, control=(PID,)):
    v = attach_window_integrity_metrics(_views({r["window"] for r in rows}), pd.DataFrame(rows),
                                        control_pids=list(control))[("window",)]
    return v


def test_stale_thread_epoch_does_not_split_the_control_window():
    rows = [_row(30, PID, 2.0, s) for s in range(1, 6)]                       # rank main thread: real epoch 2
    rows += [_row(30, PID + 1, 1.0, s, func="TFReader._parse_image") for s in range(1, 6)]   # stale tf thread
    v = _epoch_stats(rows)
    assert v.loc[30, "epoch_nunique"] == 1
    assert v.loc[30, "step_min"] == 1 and v.loc[30, "step_max"] == 5


def test_two_epochs_on_the_authoritative_rows_still_fail_the_gate():
    rows = [_row(30, PID, 2.0, 1), _row(30, PID, 3.0, 2)]
    assert _epoch_stats(rows).loc[30, "epoch_nunique"] == 2                   # gate NOT weakened


def test_two_control_ranks_in_different_epochs_still_fail_the_gate():
    rows = [_row(30, PID, 2.0, 1), _row(30, PID + 10, 3.0, 1, pid=PID + 10)]
    assert _epoch_stats(rows, control=(PID, PID + 10)).loc[30, "epoch_nunique"] == 2


def test_pid_without_main_thread_rows_keeps_all_its_rows():
    rows = [_row(30, PID + 1, 2.0, 1), _row(30, PID + 2, 3.0, 2)]             # no tid == pid row at all
    assert _epoch_stats(rows).loc[30, "epoch_nunique"] == 2                   # no silent loss of a control rank


def test_without_thread_identity_the_filter_is_a_no_op():
    rows = [{"window": 30, "pid": PID, "epoch": 2.0, "step": 1}, {"window": 30, "pid": PID, "epoch": 1.0, "step": 2}]
    assert _epoch_stats(rows).loc[30, "epoch_nunique"] == 2


def test_worker_pids_remain_excluded_and_mask_is_per_row():
    df = pd.DataFrame([_row(1, PID, 2.0, 1), _row(1, PID + 1, 1.0, 1), _row(1, 500, 1.0, 1, pid=500)])
    m = authoritative_control_rows(df, [PID])
    assert list(m) == [True, False, False]

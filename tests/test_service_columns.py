"""LiveFlow v1 per-stage service columns (lf_v1_svc_*): busy seconds per step of the critical rank's stage (Little's-law input)."""
import importlib.util
import math
import sys
import types
from pathlib import Path

import pandas as pd

_ANALYZER_DIR = Path(__file__).resolve().parents[1] / "python" / "dftracer" / "analyzer"
_pkg = types.ModuleType("dftracer")
_pkg.__path__ = [str(_ANALYZER_DIR.parent)]
sys.modules.setdefault("dftracer", _pkg)
_apkg = types.ModuleType("dftracer.analyzer")
_apkg.__path__ = [str(_ANALYZER_DIR)]
sys.modules.setdefault("dftracer.analyzer", _apkg)
_spec = importlib.util.spec_from_file_location("dftracer.analyzer.liveflow_v1", _ANALYZER_DIR / "liveflow_v1.py")
lf = importlib.util.module_from_spec(_spec)
sys.modules["dftracer.analyzer.liveflow_v1"] = lf
_spec.loader.exec_module(lf)

N = 100


def _window(posix_sum=8.0, decode_sum=2.0):
    cols = {"step_nunique": float(N), "fetch_iter_time_proc_max": 4.0, "fetch_iter_count_sum": float(4 * N),
            "comm_time_proc_max": 2.0, "comm_count_sum": float(4 * N), "compute_time_proc_max": 15.0,
            "compute_count_sum": float(4 * N),
            "reader_posix_read_time_sum": posix_sum, "reader_posix_read_time_proc_max": 1.0, "reader_posix_read_count_sum": 1000.0,
            "reader_decode_time_sum": decode_sum, "reader_decode_time_proc_max": 0.5, "reader_decode_count_sum": 400.0}
    return pd.DataFrame([cols], index=pd.Index([1], name="window"))


def _procs(per_rank_threads):
    rows, idx = [], []
    for pid, threads in per_rank_threads.items():
        for tid, (p, d) in enumerate(threads):
            idx.append("app#node1#%d#%d" % (pid, 1000 + tid))
            rows.append({"reader_posix_read_time_sum": p, "reader_decode_time_sum": d})
    return pd.DataFrame(rows, index=pd.Index(idx, name="proc_name"))


def _prep(window, proc=None, ctx=None):
    views = {("window",): window}
    if proc is not None:
        views[("proc_name",)] = proc
    decl = lf.declaration("dlio-tf", tf_decode=True)
    return lf.prepare_views(views, decl, ctx)[("window",)].iloc[0]


def test_service_is_the_critical_ranks_busy_seconds_per_step_when_ranks_resolve():
    proc = _procs({10: [(1.0, 0.2), (1.0, 0.1)], 11: [(3.0, 0.4), (1.0, 0.1)], 12: [(0.5, 0.1)], 13: [(0.5, 0.1)]})
    row = _prep(_window(posix_sum=8.0, decode_sum=1.0), proc, {"rank_pids": [10, 11, 12, 13]})
    assert math.isclose(row["lf_v1_svc_storage"], 4.0 / N) and row["lf_v1_svc_storage_critical"] == 1.0   # rank 11
    assert math.isclose(row["lf_v1_svc_decode"], 0.5 / N) and row["lf_v1_svc_decode_critical"] == 1.0


def test_unmapped_worker_processes_fall_back_to_the_node_total_per_rank():
    proc = _procs({10: [(1.0, 0.2)], 11: [(1.0, 0.2)], 999: [(6.0, 0.6)]})        # a forked worker pid, not a rank pid
    row = _prep(_window(posix_sum=8.0, decode_sum=1.0), proc, {"rank_pids": [10, 11, 12, 13]})
    assert math.isclose(row["lf_v1_svc_storage"], 8.0 / 4 / N) and row["lf_v1_svc_storage_critical"] == 0.0


def test_window_view_alone_gives_the_per_rank_mean():
    row = _prep(_window(posix_sum=8.0, decode_sum=2.0))
    assert math.isclose(row["lf_v1_svc_storage"], 8.0 / 4 / N)            # ranks = fetch_iter_count_sum / steps
    assert math.isclose(row["lf_v1_svc_decode"], 2.0 / 4 / N)


def test_an_unmeasured_stage_is_absent_never_zero():
    w = _window().drop(columns=["reader_decode_time_sum", "reader_decode_count_sum", "reader_decode_time_proc_max"])
    row = _prep(w.assign(reader_decode_time_proc_max=1.0, reader_decode_count_sum=0.0))
    assert math.isnan(row["lf_v1_svc_decode"]) and not math.isnan(row["lf_v1_svc_storage"])

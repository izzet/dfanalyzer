"""LiveFlow v1 job-level layer: barrier wait explained by peer input stalls is attributed to input.

Standalone import (same technique as test_input_job_step_max_sum_metrics.py); no cluster needed.
"""
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


def _row(n=100, fetch=4.0, comm=20.0, compute=130.0, posix=3.0, decode=1.0, critical=None, own=None):
    cols = {
        "step_nunique": float(n),
        "fetch_iter_time_proc_max": fetch, "fetch_iter_count_sum": float(n),
        "comm_time_proc_max": comm, "comm_count_sum": float(n),
        "compute_time_proc_max": compute + comm, "compute_count_sum": float(n),
        "reader_posix_read_time_proc_max": posix, "reader_decode_time_proc_max": decode,
        "reader_posix_read_count_sum": 10.0, "reader_decode_count_sum": 10.0,
    }
    if critical is not None:
        cols["domain_input_job_step_max_sum_time"] = critical
    if own is not None:
        cols["domain_input_job_own_mean_time"] = own
    return pd.DataFrame([cols], index=pd.Index([1], name="window"))


def _decl(job_level, tf_decode=True):
    return lf.declaration("dlio-tf", tf_decode=tf_decode) | {"job_level": job_level}


def _prep(row, job_level, tf_decode=True):
    return lf.prepare_views({("window",): row}, _decl(job_level, tf_decode))[("window",)].iloc[0]


def test_peer_skew_below_comm_moves_exactly_the_skew_and_keeps_the_remainder():
    row = _row(critical=12.0, own=4.0)                     # skew 8 s, barrier wait 20 s
    node, job = _prep(row, False), _prep(row, True)
    assert math.isclose(job["lf_v1_job_attributed"], 8.0 / 100)
    assert math.isclose(job["lf_v1_comm_wait"], node["lf_v1_comm_wait"] - 0.08)
    moved = (job["lf_v1_storage_wait"] - node["lf_v1_storage_wait"]
             + job["lf_v1_decode_wait"] - node["lf_v1_decode_wait"])
    assert math.isclose(moved, 0.08)
    # closure: the attribution moves time between families, it never creates or removes any
    assert math.isclose(job["lf_v1_active_time"], node["lf_v1_active_time"])
    assert math.isclose(job["lf_v1_compute_time"], node["lf_v1_compute_time"])


def test_split_follows_service_share_posix_vs_decode():
    job = _prep(_row(critical=12.0, own=4.0, posix=3.0, decode=1.0), True)
    node = _prep(_row(critical=12.0, own=4.0, posix=3.0, decode=1.0), False)
    ds = job["lf_v1_storage_wait"] - node["lf_v1_storage_wait"]
    dd = job["lf_v1_decode_wait"] - node["lf_v1_decode_wait"]
    assert math.isclose(ds / (ds + dd), 0.75) and math.isclose(dd / (ds + dd), 0.25)


def test_skew_larger_than_wait_is_capped_by_the_measured_wait():
    job = _prep(_row(comm=5.0, critical=60.0, own=4.0), True)       # skew 56 > wait 5
    assert math.isclose(job["lf_v1_job_attributed"], 0.05)
    assert math.isclose(job["lf_v1_comm_wait"], 0.0, abs_tol=1e-12)


def test_uniform_input_has_no_peer_skew_and_nothing_moves():
    row = _row(critical=4.0, own=4.0)
    job, node = _prep(row, True), _prep(row, False)
    assert job["lf_v1_job_attributed"] == 0.0
    assert job["lf_v1_comm_wait"] == node["lf_v1_comm_wait"]
    assert job["lf_v1_storage_wait"] == node["lf_v1_storage_wait"]


def test_per_node_ablation_reports_the_columns_but_never_moves_time():
    row = _row(critical=12.0, own=4.0)
    abl = _prep(row, False)
    assert abl["lf_v1_job_available"] == 1.0 and abl["lf_v1_job_level"] == 0.0
    assert math.isclose(abl["lf_v1_job_peer_skew"], 0.08) and abl["lf_v1_job_attributed"] == 0.0
    assert math.isclose(abl["lf_v1_comm_wait"], 0.2)


def test_missing_job_columns_suppress_the_attribution():
    row = _row()
    got = _prep(row, True)
    assert got["lf_v1_job_available"] == 0.0 and got["lf_v1_job_attributed"] == 0.0
    assert math.isclose(got["lf_v1_comm_wait"], 0.2)
    got2 = _prep(_row(critical=12.0), True)                        # only one of the two columns
    assert got2["lf_v1_job_available"] == 0.0 and math.isclose(got2["lf_v1_comm_wait"], 0.2)


def test_single_family_takes_all_of_the_attributed_wait():
    job = _prep(_row(critical=12.0, own=4.0), True, tf_decode=False)
    node = _prep(_row(critical=12.0, own=4.0), False, tf_decode=False)
    assert math.isclose(job["lf_v1_storage_wait"] - node["lf_v1_storage_wait"], 0.08)
    assert "lf_v1_decode_wait" in job and job["lf_v1_decode_wait"] == 0.0


def test_job_attribution_is_a_pure_minimum_of_measurements():
    assert lf.job_attribution(None, 1.0, 5.0, {"storage": 1.0}) is None
    assert lf.job_attribution(3.0, 1.0, 5.0, {}) is None
    got = lf.job_attribution(3.0, 1.0, 5.0, {"storage": 0.5, "decode": 0.5})
    assert got["attributed"] == 2.0 and got["split"] == {"storage": 1.0, "decode": 1.0}
    assert lf.job_attribution(3.0, 1.0, 1.5, {"storage": 1.0})["attributed"] == 1.5
    assert lf.job_attribution(1.0, 3.0, 5.0, {"storage": 1.0})["attributed"] == 0.0   # critical < own: no skew

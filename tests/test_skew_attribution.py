"""Measured cross-rank input-skew attribution (coherent window vector).  Synthetic per-rank
matrices; offline accounting-identity checks, not detection quality or run evidence."""

from pathlib import Path

import dftracer
_SRC = str(Path(__file__).resolve().parents[1] / "python/dftracer")
if _SRC not in list(dftracer.__path__):
    dftracer.__path__.insert(0, _SRC)
import dftracer.analyzer as _ns  # noqa: E402
if str(Path(_SRC) / "analyzer") not in _ns.__path__:
    _ns.__path__.insert(0, str(Path(_SRC) / "analyzer"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dftracer.analyzer.metrics import (  # noqa: E402
    DOMAIN_VECTOR_COLUMNS, SKEW_COLUMNS, coherent_domain_vectors, set_window_decomposition_metrics)


def _mats(inp, comm, compute=None, decode=None):
    n = len(inp)
    return {
        "window_wall": np.full((1, n), 10.0),
        "stall_storage": np.array([inp], dtype=float),
        "stall_decode": np.array([decode or [0.0] * n], dtype=float),
        "comm": np.array([comm], dtype=float),
        "compute": np.array([compute or [3.0] * n], dtype=float) + np.array([comm], dtype=float),
    }


def _skew(inp, comm, **kw):
    expected = kw.pop("expected_procs", None)
    return coherent_domain_vectors(_mats(inp, comm, **kw), expected_procs=expected)


def test_columns_are_part_of_the_published_wire_contract():
    assert set(SKEW_COLUMNS) <= set(DOMAIN_VECTOR_COLUMNS)


def test_one_slow_reader_makes_peers_barrier_wait_input_up_to_its_excess():
    # rank 0 starved 4.0 s and arrived last (comm 1.0); three peers waited 5.0 s in the collective.
    c = _skew([4.0, 0.2, 0.2, 0.2], [1.0, 5.0, 5.0, 5.0])
    assert c["domain_skew_coverage"][0] == 1.0
    assert c["domain_input_skew_excess_time"][0] == pytest.approx(3.8)
    assert c["domain_comm_barrier_wait_time"][0] == pytest.approx(4.0)
    assert c["domain_input_skew_attributed_time"][0] == pytest.approx(3.8)     # min(E, W)
    assert c["domain_comm_residual_after_skew_time"][0] == pytest.approx(1.2)  # stays comm
    # accounting identity: peers' comm = late rank's comm + wait = attributed + residual + late comm
    assert (c["domain_skew_peer_median_comm_time"][0]
            == pytest.approx(c["domain_skew_late_proc_comm_time"][0] + c["domain_comm_barrier_wait_time"][0]))
    assert (c["domain_input_skew_attributed_time"][0] + c["domain_comm_residual_after_skew_time"][0]
            == pytest.approx(c["domain_skew_peer_median_comm_time"][0]))


def test_attribution_is_capped_by_the_measured_wait_when_excess_is_larger():
    c = _skew([9.0, 0.2, 0.2], [1.0, 3.0, 3.0])
    assert c["domain_input_skew_excess_time"][0] == pytest.approx(8.8)
    assert c["domain_input_skew_attributed_time"][0] == pytest.approx(2.0)     # = W
    assert c["domain_comm_residual_after_skew_time"][0] == pytest.approx(1.0)


def test_decode_estimate_counts_toward_the_measured_total_wait():
    c = _skew([1.0, 0.1, 0.1], [1.0, 4.0, 4.0], decode=[2.0, 0.1, 0.1])
    assert c["domain_input_skew_excess_time"][0] == pytest.approx(2.8)


@pytest.mark.parametrize("case", ("uniform_comm_slowness", "compute_caused_skew", "peers_slower_input"))
def test_negative_controls_do_not_attribute(case):
    if case == "uniform_comm_slowness":       # everyone waits equally, equal input: pure comm
        c = _skew([0.2] * 4, [5.0] * 4)
    elif case == "compute_caused_skew":       # rank 0 late (comm 1.0) but its INPUT equals its peers'
        c = _skew([0.2] * 4, [1.0, 5.0, 5.0, 5.0], compute=[9.0, 3.0, 3.0, 3.0])
    else:                                     # the late rank has LESS input than its peers
        c = _skew([0.1, 0.9, 0.9], [1.0, 5.0, 5.0])
    assert c["domain_skew_coverage"][0] == 1.0     # measurable, and the answer is zero
    assert c["domain_input_skew_attributed_time"][0] == 0.0
    assert c["domain_comm_residual_after_skew_time"][0] == pytest.approx(
        c["domain_skew_peer_median_comm_time"][0])


@pytest.mark.parametrize("case", (
    "missing_rank_vs_expected", "single_rank", "rank_without_comm_record",
    "no_comm_instrument", "no_input_instrument"))
def test_missing_data_suppresses_instead_of_inventing(case):
    kw = {}
    if case == "missing_rank_vs_expected":
        m = _mats([4.0, 0.2, 0.2], [1.0, 5.0, 5.0]); kw["expected_procs"] = 4
    elif case == "single_rank":
        m = _mats([4.0], [1.0])
    elif case == "rank_without_comm_record":   # an absent record must not read as "arrived last"
        m = _mats([0.2, 4.0, 0.2], [0.0, 5.0, 5.0])
    elif case == "no_comm_instrument":
        m = _mats([4.0, 0.2, 0.2], [1.0, 5.0, 5.0]); m.pop("comm")
    else:
        m = _mats([4.0, 0.2, 0.2], [1.0, 5.0, 5.0]); m.pop("stall_storage"); m.pop("stall_decode")
    c = coherent_domain_vectors(m, **kw)
    assert c["domain_skew_coverage"][0] == 0.0
    for name in SKEW_COLUMNS:
        assert c[name][0] == 0.0, name


def test_worker_rows_without_window_wall_are_not_ranks():
    m = _mats([4.0, 0.2, 0.2, 500.0], [1.0, 5.0, 5.0, 900.0])
    m["window_wall"][0, 3] = 0.0            # a DataLoader worker row
    c = coherent_domain_vectors(m, expected_procs=3)
    assert c["domain_skew_coverage"][0] == 1.0
    assert c["domain_input_skew_attributed_time"][0] == pytest.approx(3.8)


def test_columns_reach_the_window_frame():
    c = _skew([4.0, 0.2, 0.2, 0.2], [1.0, 5.0, 5.0, 5.0])
    frame = pd.DataFrame({"window_wall_time_proc_max": [10.0], "stall_storage_time_proc_max": [4.0],
                          "stall_decode_time_proc_max": [0.0], "comm_time_proc_max": [5.0],
                          "compute_time_proc_max": [8.0]}, index=[7])
    out = set_window_decomposition_metrics(frame, is_view_process_based=False, coherent=c)
    assert out.loc[7, "domain_input_skew_attributed_time"] == pytest.approx(3.8)
    legacy = set_window_decomposition_metrics(frame, is_view_process_based=False, coherent=None)
    assert legacy.loc[7, "domain_skew_coverage"] == 0.0     # no matrices: suppressed, columns exist


def _ev(pid, name, cat, func, seconds):
    return {"window": 7, "epoch": 1, "step": 1, "pid": pid, "proc_name": name, "cat": cat,
            "func_name": func, "io_cat": 0, "time": seconds, "size": 0}


def test_raw_fastpath_publishes_the_skew_columns_from_rank_events():
    from dftracer.analyzer.config import AnalyzerPresetConfigDLIOAILogging
    from dftracer.analyzer.fact_engine import FactEngine
    from dftracer.analyzer.fastpath import FastPathPlan
    preset = AnalyzerPresetConfigDLIOAILogging()
    rule = _SRC and (Path(_SRC) / "analyzer/configs/fact_rules/p3-tf-rank-thread-slope.yaml")
    plan = FastPathPlan.build(
        preset, ["window", "proc_name"], list(preset.layer_defs),
        FactEngine.from_rule_file(str(rule), strict_time_semantics=True,
                                  allow_mixed_time_aggregates=False))
    assert plan is not None
    rows = []
    for pid, stall, comm in ((10, 4.0, 1.0), (11, 0.2, 5.0), (12, 0.2, 5.0)):
        n = "rank-%d" % pid
        rows += [_ev(pid, n, "window", "window.stop", 10.0), _ev(pid, n, "compute", "compute", 8.0),
                 _ev(pid, n, "comm", "comm.all_reduce", comm), _ev(pid, n, "stall", "stall.storage", stall)]
    views = plan.flat_views(pd.DataFrame(rows), control_pids=[10, 11, 12])
    row = views[("window",)].loc[7]
    assert row["domain_skew_coverage"] == 1.0
    assert row["domain_input_skew_attributed_time"] == pytest.approx(3.8)
    assert row["domain_comm_residual_after_skew_time"] == pytest.approx(1.2)

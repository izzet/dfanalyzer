"""Regression test for the v4.5.10.1 1N finding (2026-09-28): the analyzer computed the loop's
exposure metrics (job/step input sum-of-max, exposed/total comm window medians) on the ("window",)
view, but a finding's key_metrics are exactly the fact's evidence metrics, which held only the
rule's own required/derived metrics -- so the planner logged input_job_sum_of_max=null,
comm_*_median=null and silently used the crit-reference tier those metrics were built to replace.

Drives the real StorMer loop rule file through FactEngine.evaluate() with a synthetic but
rule-satisfying proc_name frame and asserts the seven planner-read window metrics are forwarded
into the emitted fact's evidence when the window context carries them, are absent (not None, not a
failure) when it does not, and that the fact still forms either way.
"""
import importlib.util
import math
import sys
import types
from pathlib import Path

import pandas as pd

_ANALYZER_DIR = Path(__file__).resolve().parents[1] / "python" / "dftracer" / "analyzer"
_RULE_FILE = _ANALYZER_DIR / "configs" / "fact_rules" / "liveflow-stormer-loop.yaml"

_dftracer_pkg = types.ModuleType("dftracer")
_dftracer_pkg.__path__ = [str(_ANALYZER_DIR.parent)]
sys.modules["dftracer"] = _dftracer_pkg
_analyzer_pkg = types.ModuleType("dftracer.analyzer")
_analyzer_pkg.__path__ = [str(_ANALYZER_DIR)]
sys.modules["dftracer.analyzer"] = _analyzer_pkg


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(
        modname, _ANALYZER_DIR / filename, submodule_search_locations=[str(_ANALYZER_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_load("dftracer.analyzer.types", "types.py")
_load("dftracer.analyzer.fact_rules", "fact_rules.py")
fe = _load("dftracer.analyzer.fact_engine", "fact_engine.py")

HOST = "nodeA"
RANK_PIDS = [100, 101, 102, 103]

LOOP_VALUES = {
    "domain_input_step_max_sum_time": 0.96,
    "domain_input_step_nunique": 10.0,
    "domain_input_job_step_max_sum_time": 1.56,
    "domain_comm_pure_exposure_time_v2": 0.0046,
    "domain_comm_exposure_step_nunique": 10.0,
    "domain_comm_total_median_time_v3": 0.55,
    "domain_comm_total_step_nunique": 10.0,
}


def _window_metrics(with_loop_metrics):
    wm = {
        "domain_coherent": 1.0, "domain_procs_active": 4.0,
        "window_wall_time_proc_crit": 8.9, "domain_input_time_proc_crit": 0.74,
        "domain_decode_time_proc_crit": 0.0, "domain_comm_time_proc_crit": 0.65,
        "domain_compute_time_proc_crit": 0.88, "residual_time_proc_crit": 6.6,
        "epoch_nunique": 1.0, "step_nunique": 10.0, "step_min": 91.0, "step_max": 100.0,
        "domain_skew_coverage": 1.0, "domain_input_skew_attributed_time": 0.0,
        "domain_input_skew_excess_time": 0.0, "domain_comm_barrier_wait_time": 0.14,
        "domain_skew_late_proc_input_time": 0.1, "domain_skew_late_proc_comm_time": 0.5,
        "domain_comm_residual_after_skew_time": 0.64,
        "domain_input_time_proc_typ": 0.1, "domain_decode_time_proc_typ": 0.0,
        "domain_comm_time_proc_typ": 0.5, "domain_compute_time_proc_typ": 0.9,
        "window_wall_time_proc_typ": 8.9,
    }
    if with_loop_metrics:
        wm.update(LOOP_VALUES)
    return wm


def _frame_and_context(with_loop_metrics):
    # One row per rank main thread, carrying the fetch_iter family with slope < 1 on rank 100.
    slopes = {100: 0.4, 101: 3.0, 102: 2.5, 103: 1.5}
    rows, index, row_pid = [], [], {}
    for pid in RANK_PIDS:
        name = "app#%s#%d#%d" % (HOST, pid, pid)
        index.append(name)
        row_pid[name] = pid
        rows.append({
            "fetch_iter_count_frac_total": 0.25,
            "fetch_iter_time_proc_frac_total": 0.25 / slopes[pid],
            "fetch_iter_ops_slope": slopes[pid],
            "comm_all_reduce_count_frac_total": 0.25,
            "comm_all_reduce_time_proc_frac_total": 0.25,
            "comm_all_reduce_ops_slope": 0.9 if pid == 100 else 1.1,
        })
    frame = pd.DataFrame(rows, index=pd.Index(index, name="proc_name"))
    context = {
        "native_window": 10, "native_host": HOST, "rank_pids": RANK_PIDS,
        "worker_pids": [], "row_pid_by_index": row_pid,
        "window_metrics": _window_metrics(with_loop_metrics),
    }
    return {("proc_name",): frame}, context


def _engine():
    return fe.FactEngine.from_rule_file(
        str(_RULE_FILE), strict_time_semantics=False, allow_mixed_time_aggregates=True)


def _fetch_fact(result):
    facts = [f for fs in result.values() for f in fs]
    fetch = [f for f in facts if "fetch_iter" in str(f.scope.entity) or "fetch_iter" in str(f)]
    return fetch, facts


def _evidence(fact):
    return fact.evidence["metrics"] if hasattr(fact, "evidence") else fact.key_metrics


def test_loop_exposure_metrics_are_forwarded_into_fact_evidence():
    views, ctx = _frame_and_context(with_loop_metrics=True)
    result = _engine().evaluate(flat_views=views, raw_stats={}, context=ctx)
    fetch, facts = _fetch_fact(result)
    assert fetch, "fetch_iter rule must still form a fact; got %r" % (facts,)
    ev = _evidence(fetch[0])
    for name, expected in LOOP_VALUES.items():
        assert name in ev, "%s was not forwarded into the fact's evidence" % name
        assert math.isclose(float(ev[name]), expected), (name, ev[name])


def test_missing_loop_metrics_do_not_block_the_fact_or_appear_as_none():
    views, ctx = _frame_and_context(with_loop_metrics=False)
    result = _engine().evaluate(flat_views=views, raw_stats={}, context=ctx)
    fetch, facts = _fetch_fact(result)
    assert fetch, "absence of the optional loop metrics must not suppress the rule"
    ev = _evidence(fetch[0])
    for name in LOOP_VALUES:
        assert name not in ev, "%s must be absent (not None) when the window lacks it" % name


def test_analyzer_window_metrics_list_includes_the_loop_metrics():
    text = (_ANALYZER_DIR / "analyzer.py").read_text()
    assert "LOOP_EXPOSURE_WINDOW_METRICS" in text
    idx = text.index('fact_context["window_metrics"]')
    assert "LOOP_EXPOSURE_WINDOW_METRICS" in text[idx:idx + 400]

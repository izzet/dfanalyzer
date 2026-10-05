"""Offline controls for the parked worker-POSIX WiSIO diagnostic.

This is deliberately not a StorMer rule or action admission.  It uses the
existing DLIO worker-slope rule to characterize the current all-proc-name
normalization on a StorMer-shaped completed window.  In particular,
``worker_only`` selects after the shares and slope have already been calculated.
The generic Planner's default 0.5 severity admission is an old fixed actuation
gate, not a validated scientific safety boundary.
"""

import dataclasses as dc
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Match node_agent bootstrap for the source analyzer package.
import dftracer

_ROOT = Path(__file__).resolve().parents[2]
_ANALYZER_PACKAGE = _ROOT / "dfanalyzer/python/dftracer"
if str(_ANALYZER_PACKAGE) not in dftracer.__path__:
    dftracer.__path__.insert(0, str(_ANALYZER_PACKAGE))

from dftracer.analyzer.fact_engine import FactEngine
from dftracer.analyzer.fastpath import FastPathPlan


_RULE = _ROOT / "dfanalyzer/python/dftracer/analyzer/configs/fact_rules/dlio-wisio-worker-slope.yaml"
_CONTROL_COLUMNS = (
    "domain_coherent", "domain_procs_active", "window_wall_time_proc_crit",
    "domain_input_time_proc_crit", "domain_decode_time_proc_crit",
    "domain_comm_time_proc_crit", "domain_compute_time_proc_crit",
    "residual_time_proc_crit", "epoch_nunique", "step_nunique", "step_min", "step_max",
)


def _plan():
    """Use the same vectorized proc-name metric path as live analysis."""
    plan = FastPathPlan.__new__(FastPathPlan)
    plan.view_types = ["window", "proc_name"]
    plan.terms = [
        ("window_wall", 'cat == "window" & func_name == "window.stop"', False),
        ("compute", 'cat == "compute" & func_name == "compute"', False),
        ("stall_storage", 'cat == "stall" & func_name == "stall.storage"', False),
        ("reader_posix_read", 'cat == "posix_reader" & func_name == "read"', False),
    ]
    plan._mask_cache = {}
    return plan


def _event(pid, proc_name, cat, func_name, seconds):
    return {
        "window": 1, "epoch": 1, "step": 1, "pid": pid, "proc_name": proc_name,
        "cat": cat, "func_name": func_name, "io_cat": 0, "time": seconds, "size": 0,
    }


def _replay(*, parent_read_times=(), worker_read_times=(1.0, 1.0), input_exposure=1.0):
    """Return raw read rows, proc metrics, coherent window metrics, and parked-rule facts."""
    rows = [
        _event(100, "rank-main", "window", "window.stop", 10.0),
        _event(100, "rank-main", "compute", "compute", 10.0 - input_exposure),
        _event(100, "rank-main", "stall", "stall.storage", input_exposure),
    ]
    rows.extend(_event(100, "rank-main", "posix_reader", "read", value)
                for value in parent_read_times)
    worker_specs = ((200, "worker-a"), (201, "worker-b"), (202, "worker-c"))
    if len(worker_read_times) > len(worker_specs):
        raise ValueError("fixture supports at most three workers")
    active_workers = worker_specs[:len(worker_read_times)]
    rows.extend(_event(pid, name, "posix_reader", "read", value)
                for (pid, name), value in zip(active_workers, worker_read_times))
    traces = pd.DataFrame(rows)
    views = _plan().flat_views(traces, control_pids=[100])
    proc = views[("proc_name",)]
    window = views[("window",)].loc[1]
    context = {
        "native_window": 1,
        "worker_pids": [pid for pid, _name in active_workers],
        "row_pid_by_index": {"rank-main": 100, **{name: pid for pid, name in active_workers}},
        "window_metrics": {key: float(window[key]) for key in _CONTROL_COLUMNS},
    }
    engine = FactEngine.from_rule_file(str(_RULE), strict_time_semantics=True,
                                       allow_mixed_time_aggregates=False)
    facts = engine.evaluate({("proc_name",): proc}, raw_stats={"run_id": "offline-control"},
                            context=context)
    read_columns = [
        "reader_posix_read_count_sum", "reader_posix_read_time_sum",
        "reader_posix_read_count_frac_total", "reader_posix_read_time_proc_frac_total",
        "reader_posix_read_ops_slope",
    ]
    return traces, proc.loc[["rank-main", *[name for _pid, name in active_workers]], read_columns], window, facts


@pytest.mark.parametrize(
    "name,parent,workers,exposure,expected_slopes,expect_fact",
    [
        ("equal_durations", (1.0,), (1.0, 1.0), 1.0,
         {"rank-main": 1.0, "worker-a": 1.0, "worker-b": 1.0}, False),
        # Identical workers are made to look slow solely by a fast rank-parent read.
        ("fast_parent_reference", (0.1,), (1.0, 1.0), 1.0,
         {"rank-main": 7.0, "worker-a": 0.7, "worker-b": 0.7}, True),
        # The converse parent reference makes the same worker service look fast.
        ("slow_parent_reference", (10.0,), (1.0, 1.0), 1.0,
         {"rank-main": 0.4, "worker-a": 4.0, "worker-b": 4.0}, False),
        # Service-time imbalance alone can fire while rank-owned input exposure is negligible.
        ("unequal_workers_hidden_input", (), (3.0, 1.0), 0.01,
         {"rank-main": None, "worker-a": 2.0 / 3.0, "worker-b": 2.0}, True),
        # Uniformly slower service retains slope one even with substantial exposure.
        ("uniform_slowdown", (), (10.0, 10.0), 6.0,
         {"rank-main": None, "worker-a": 1.0, "worker-b": 1.0}, False),
    ],
)
def test_parked_worker_slope_controls(name, parent, workers, exposure, expected_slopes, expect_fact):
    traces, proc, window, facts = _replay(
        parent_read_times=parent, worker_read_times=workers, input_exposure=exposure,
    )

    # Preserve raw event PID/count/time evidence alongside the normalized rows.
    raw_reads = traces.loc[traces["func_name"] == "read", ["pid", "proc_name", "time"]]
    expected_counts = {
        "rank-main": len(parent), "worker-a": 1, "worker-b": 1,
    } if parent else {"worker-a": 1, "worker-b": 1}
    assert raw_reads.groupby("proc_name").size().to_dict() == expected_counts
    for row, slope in expected_slopes.items():
        if slope is None:
            assert pd.isna(proc.loc[row, "reader_posix_read_ops_slope"])
        else:
            assert proc.loc[row, "reader_posix_read_ops_slope"] == pytest.approx(slope)
    assert window["domain_coherent"] == 1.0
    assert window["domain_input_time_proc_crit"] == pytest.approx(exposure)

    emitted = facts.get(("proc_name",), [])
    assert bool(emitted) is expect_fact, name
    if expect_fact:
        selected = emitted[0].evidence["metrics"]
        assert selected["selected_worker_pid"] == 200
        assert selected["reader_posix_read_ops_slope"] == pytest.approx(
            expected_slopes["worker-a"])


def test_three_worker_population_crosses_legacy_gate_without_exposed_input(monkeypatch):
    """Show that the old fixed gate is population-sensitive, not protection.

    The current StorMer rule selections do not load this DLIO rule.  If it were
    deliberately wired to the ordinary Diagnoser/Planner path, this 9:1:1
    hidden-input population has a slow-worker slope 11/27 and deficit 16/27,
    crossing the generic Planner's old 0.5 gate.  That is a negative control,
    not an approved StorMer policy or a validated safety condition.
    """
    try:
        from dfdiagnoser.diagnoser import Diagnoser
        from dfoptimizer.planner.planner import Planner
        from dfoptimizer.types import DiagnosisFindingMsg, KnobDef, KnobResponse
    except ImportError as exc:  # Component test environments must not silently fake this path.
        pytest.fail("installed diagnoser/optimizer composition unavailable: %s" % exc)

    monkeypatch.setenv("DFDIAGNOSER_METRICS_FROM", "current")
    engine = FactEngine.from_rule_file(str(_RULE), strict_time_semantics=True,
                                       allow_mixed_time_aggregates=False)
    diagnoser = Diagnoser()
    planner = Planner()
    planner.register_knobs({
        "stormer.num_workers": KnobDef(
            id="stormer.num_workers", default=4, type=int, range=(0, 8),
            target_function="stormer.make_loader",
            responds_to={"reader_parallelism": KnobResponse(
                direction="increase", step_mode="evidence", min_persistence=2,
                cooldown_windows=3, apply_when="window_boundary",
            )},
        ),
    })
    plans = []
    for native_window in (1, 2):
        _traces, proc, window, _facts = _replay(
            worker_read_times=(9.0, 1.0, 1.0), input_exposure=0.01,
        )
        context = {
            "native_window": native_window, "worker_pids": [200, 201, 202],
            "row_pid_by_index": {
                "rank-main": 100, "worker-a": 200, "worker-b": 201, "worker-c": 202,
            },
            "window_metrics": {key: float(window[key]) for key in _CONTROL_COLUMNS},
        }
        facts = engine.evaluate({("proc_name",): proc}, raw_stats={}, context=context)[("proc_name",)]
        metrics = facts[0].evidence["metrics"]
        assert metrics["selected_worker_pid"] == 200
        assert metrics["reader_posix_read_ops_slope"] == pytest.approx(11.0 / 27.0)
        assert metrics["reader_posix_read_ops_slope_ref45_deficit"] == pytest.approx(16.0 / 27.0)
        payload = json.dumps({"view_type": "proc_name", "facts": [dc.asdict(f) for f in facts]}).encode()
        touched = diagnoser._handle_analysis_facts(type("Event", (), {"data": payload})(), metadata={})
        findings = diagnoser._build_control_findings(native_window, touched)
        for finding in findings:
            plans.extend(planner.process_finding(DiagnosisFindingMsg(
                finding_type=finding.finding_type, motif=finding.motif, severity=finding.severity,
                severity_score=finding.severity_score, confidence=finding.confidence,
                prevalence=finding.trend.prevalence, persistence=finding.trend.persistence,
                trend_direction=finding.trend.trend_direction,
                contributing_facts=finding.contributing_facts,
                recommendation_bundle=finding.recommendation_bundle,
                opportunity_tags=finding.opportunity_tags, summary=finding.summary,
                window_index=finding.trend.last_seen_window, key_metrics=finding.key_metrics,
            )))

    assert len(plans) == 1
    # Current evidence-mode Planner output is its capped 4 -> 8 proposal;
    # record it as behavior, not as an endorsed action size.
    assert (plans[0].knob_id, plans[0].old_value, plans[0].new_value) == (
        "stormer.num_workers", 4, 8)

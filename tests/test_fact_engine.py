import sys
from pathlib import Path

import pandas as pd
import pytest

# Match the node agent bootstrap: use this worktree's analyzer code while the
# extension package itself remains installed in the supported venv.
import dftracer

_SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "python" / "dftracer"
if str(_SOURCE_PACKAGE) not in dftracer.__path__:
    dftracer.__path__.insert(0, str(_SOURCE_PACKAGE))

from dftracer.analyzer.fact_engine import FactEngine
from dftracer.analyzer.fact_rules import FactRuleValidationError, build_fact_rule


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def test_fact_engine_emits_per_row_and_window_facts():
    per_row_rule = build_fact_rule(
        raw_rule={
            "id": "rule.fetch.pressure",
            "priority": 10,
            "source_view": "epoch",
            "fact_type": "fetch_pressure",
            "required_metrics": ["fetch_iter_time_frac_parent"],
            "when": "fetch_iter_time_frac_parent >= 0.5",
            "severity_score": "clip01(fetch_iter_time_frac_parent)",
            "confidence": "0.9",
            "opportunity_tags": ["reader_parallelism"],
        }
    )
    window_rule = build_fact_rule(
        raw_rule={
            "id": "rule.epoch.straggler",
            "priority": 20,
            "source_view": "proc_name",
            "emit_mode": "window",
            "fact_type": "epoch_straggler",
            "required_metrics": ["epoch_time_max"],
            "derived_metrics": {
                "epoch_straggler_ratio": "epoch_time_max / 3.0",
            },
            "when": "epoch_straggler_ratio > 2.0",
            "severity_score": "clip01((epoch_straggler_ratio - 1.0) / 2.0)",
            "confidence": "0.85",
            "opportunity_tags": ["rank_balance"],
        }
    )
    engine = FactEngine([per_row_rule, window_rule])
    flat_views = {
        ("epoch",): pd.DataFrame(
            {
                "fetch_iter_time_frac_parent": [0.20, 0.55, 0.80],
            },
            index=[1, 2, 3],
        ),
        ("proc_name",): pd.DataFrame(
            {
                "epoch_time_max": [10.0, 2.0, 5.0],
            },
            index=["r0", "r1", "r2"],
        ),
    }

    facts = engine.evaluate(flat_views=flat_views, raw_stats={"run_id": "run-123"})

    epoch_facts = facts[("epoch",)]
    assert len(epoch_facts) == 2
    assert {fact.window.epoch for fact in epoch_facts} == {2, 3}
    assert all(fact.scope.entity in {"2", "3"} for fact in epoch_facts)
    assert all(fact.confidence == pytest.approx(0.9) for fact in epoch_facts)
    assert all(fact.fact_id and fact.fact_id.startswith("af_") for fact in epoch_facts)

    proc_facts = facts[("proc_name",)]
    assert len(proc_facts) == 1
    assert proc_facts[0].scope.entity == "window"
    assert proc_facts[0].fact_type == "epoch_straggler"
    assert proc_facts[0].evidence["metrics"]["epoch_straggler_ratio"] == pytest.approx(10.0 / 3.0)


def test_fact_engine_preserves_native_window_view_identity():
    rule = build_fact_rule(
        raw_rule={
            "id": "rule.window.signal",
            "priority": 1,
            "source_view": "window",
            "fact_type": "window_signal",
            "required_metrics": ["active_time"],
            "when": "active_time > 0",
            "severity_score": "1.0",
        }
    )
    facts = FactEngine([rule]).evaluate(
        {("window",): pd.DataFrame({"active_time": [2.0]}, index=pd.Index([9], name="window"))},
        raw_stats={"run_id": "native-window"},
    )
    fact = facts[("window",)][0]
    assert fact.window.view_type == "window"
    assert fact.window.window == 9
    assert fact.window.epoch is None


def test_fact_engine_explicitly_rebinds_epoch_rules_for_native_window_only():
    rule = build_fact_rule({
        "id": "native.rebind", "priority": 1, "source_view": "epoch",
        "fact_type": "signal", "required_metrics": ["active_time"],
        "when": "active_time > 0", "severity_score": "1.0",
    })
    engine = FactEngine([rule])
    assert engine.rebind_source_view(from_view="epoch", to_view="window") == 1
    assert engine.evaluate({("epoch",): pd.DataFrame({"active_time": [1.0]}, index=[9])}, raw_stats={}) == {}
    facts = engine.evaluate({("window",): pd.DataFrame({"active_time": [1.0]}, index=[9])}, raw_stats={})
    fact = facts[("window",)][0]
    assert fact.window.view_type == "window"
    assert fact.window.window == 9


def test_proc_name_worker_slope_fact_uses_native_window_and_minimum_worker():
    rule = build_fact_rule({
        "id": "wisio.reader_posix_read.worker_pool.v1",
        "priority": 1,
        "source_view": "proc_name",
        "emit_mode": "window",
        "scope_entity": "reader_posix_read.worker_pool",
        "worker_only": True,
        "window_reducer": "min",
        "window_reducer_metric": "reader_posix_read_ops_slope",
        "fact_type": "operation_family_time_disproportion",
        "required_metrics": ["domain_coherent", "reader_posix_read_ops_slope"],
        "derived_metrics": {
            "reader_posix_read_ops_slope_ref45_deficit":
                "clip01(1.0 - reader_posix_read_ops_slope)",
        },
        "when": "domain_coherent == 1.0 and reader_posix_read_ops_slope > 0.0 and reader_posix_read_ops_slope_ref45_deficit > 0.0",
        "severity_score": "reader_posix_read_ops_slope_ref45_deficit",
        "opportunity_tags": ["reader_parallelism"],
    })
    # The diagnostic view contains family evidence only.  Control metrics are
    # supplied from the one completed native window, after worker selection.
    rows = pd.DataFrame({
        "reader_posix_read_ops_slope": [0.1, 0.8, 0.3, 1.2],
    }, index=["rank-main", "worker-a", "worker-b", "worker-c"])
    engine = FactEngine([rule])

    # No authoritative completed-window provenance means fail closed.
    assert engine.evaluate({("proc_name",): rows}, raw_stats={}) == {}

    facts = engine.evaluate(
        {("proc_name",): rows}, raw_stats={"run_id": "native-worker"},
        context={
            "native_window": 17,
            "worker_pids": [22, 23, 24],
            "row_pid_by_index": {"rank-main": 11, "worker-a": 22,
                                 "worker-b": 23, "worker-c": 24},
            "window_metrics": {
                "domain_coherent": 1.0, "domain_procs_active": 1.0,
                "window_wall_time_proc_crit": 2.0,
                "domain_input_time_proc_crit": 1.0,
                "domain_decode_time_proc_crit": 0.0,
                "domain_comm_time_proc_crit": 0.0,
                "domain_compute_time_proc_crit": 1.0,
                "residual_time_proc_crit": 0.0,
                "epoch_nunique": 1.0, "step_nunique": 2.0,
                "step_min": 1.0, "step_max": 2.0,
            },
        },
    )
    fact = facts[("proc_name",)][0]
    assert fact.window.view_type == "proc_name"
    assert fact.window.window == 17
    assert fact.scope.entity == "reader_posix_read.worker_pool"
    # worker-b is the minimum valid worker slope; rank-main is excluded.
    assert fact.evidence["metrics"]["reader_posix_read_ops_slope"] == pytest.approx(0.3)
    assert fact.evidence["metrics"]["selected_worker_pid"] == 23

    zero_rows = rows.copy()
    zero_rows.loc[["worker-a", "worker-b", "worker-c"], "reader_posix_read_ops_slope"] = 0.0
    assert engine.evaluate(
        {("proc_name",): zero_rows}, raw_stats={},
        context={
            "native_window": 18, "worker_pids": [22, 23, 24],
            "row_pid_by_index": {"rank-main": 11, "worker-a": 22,
                                 "worker-b": 23, "worker-c": 24},
            "window_metrics": {
                "domain_coherent": 1.0, "domain_procs_active": 1.0,
                "window_wall_time_proc_crit": 2.0, "domain_input_time_proc_crit": 1.0,
                "domain_decode_time_proc_crit": 0.0, "domain_comm_time_proc_crit": 0.0,
                "domain_compute_time_proc_crit": 1.0, "residual_time_proc_crit": 0.0,
                "epoch_nunique": 1.0, "step_nunique": 2.0,
                "step_min": 1.0, "step_max": 2.0,
            },
        },
    ) == {}
    assert fact.severity.score == pytest.approx(0.7)


def test_worker_rule_native_window_selector_comes_from_context_not_proc_row():
    rule = build_fact_rule({
        "id": "native.window.fixture", "priority": 1, "source_view": "proc_name",
        "emit_mode": "window", "worker_only": True,
        "window_reducer": "min", "window_reducer_metric": "reader_posix_read_ops_slope",
        "fact_type": "fixture", "required_metrics": [
            "__liveflow_window_index", "domain_coherent", "reader_posix_read_ops_slope",
        ],
        "when": "__liveflow_window_index == 17.0 and domain_coherent == 1.0 and reader_posix_read_ops_slope > 0.0",
        "severity_score": "0.5",
    })
    # The row tries to claim window 999.  The emitted fact must instead carry
    # the completed analyzer window supplied by context.
    rows = pd.DataFrame({
        "__liveflow_window_index": [999.0],
        "reader_posix_read_ops_slope": [0.5],
    }, index=["worker"])
    context = {
        "native_window": 17, "worker_pids": [22],
        "row_pid_by_index": {"worker": 22},
        "window_metrics": {
            "domain_coherent": 1.0, "domain_procs_active": 1.0,
            "window_wall_time_proc_crit": 2.0, "domain_input_time_proc_crit": 1.0,
            "domain_decode_time_proc_crit": 0.0, "domain_comm_time_proc_crit": 0.0,
            "domain_compute_time_proc_crit": 1.0, "residual_time_proc_crit": 0.0,
            "epoch_nunique": 1.0, "step_nunique": 1.0,
            "step_min": 1.0, "step_max": 1.0,
        },
    }
    facts = FactEngine([rule]).evaluate({("proc_name",): rows}, raw_stats={}, context=context)
    assert facts[("proc_name",)][0].window.window == 17


@pytest.mark.parametrize("invalid_control", [
    {"epoch_nunique": 2.0},
    {"step_nunique": 0.0},
    {"step_min": 3.0, "step_max": 2.0},
])
def test_proc_name_worker_slope_fails_closed_for_invalid_control_window(invalid_control):
    rule = build_fact_rule({
        "id": "wisio.worker.control-gate", "priority": 1,
        "source_view": "proc_name", "emit_mode": "window", "worker_only": True,
        "window_reducer": "min", "window_reducer_metric": "read_ops_slope",
        "fact_type": "operation_family_time_disproportion",
        "required_metrics": ["domain_coherent", "read_ops_slope"],
        "when": "domain_coherent == 1.0 and read_ops_slope > 0.0",
        "severity_score": "clip01(1.0 - read_ops_slope)",
    })
    control = {
        "domain_coherent": 1.0, "domain_procs_active": 1.0,
        "window_wall_time_proc_crit": 1.0,
        "domain_input_time_proc_crit": 1.0,
        "domain_decode_time_proc_crit": 0.0, "domain_comm_time_proc_crit": 0.0,
        "domain_compute_time_proc_crit": 0.0, "residual_time_proc_crit": -0.5,
        "epoch_nunique": 1.0, "step_nunique": 1.0,
        "step_min": 2.0, "step_max": 2.0,
    }
    control.update(invalid_control)
    facts = FactEngine([rule]).evaluate(
        {("proc_name",): pd.DataFrame({"read_ops_slope": [0.2, 0.1]}, index=["rank", "worker"])},
        raw_stats={},
        context={"native_window": 8, "worker_pids": [22],
                 "row_pid_by_index": {"rank": 11, "worker": 22},
                 "window_metrics": control},
    )
    assert facts == {}


def test_parked_dlio_worker_rule_merges_only_completed_window_control_vector():
    """The unselected DLIO rule is executable without duplicating control data in proc_name."""
    rule_file = (
        Path(__file__).resolve().parents[1]
        / "python/dftracer/analyzer/configs/fact_rules/dlio-wisio-worker-slope.yaml"
    )
    engine = FactEngine.from_rule_file(
        str(rule_file), strict_time_semantics=True,
        allow_mixed_time_aggregates=False,
    )
    facts = engine.evaluate(
        {("proc_name",): pd.DataFrame({
            "reader_posix_read_ops_slope": [0.2, 0.7, 1.1],
        }, index=["rank", "worker-slow", "worker-fast"])},
        raw_stats={"run_id": "parked-rule-unit"},
        context={
            "native_window": 12,
            "worker_pids": [22, 23],
            "row_pid_by_index": {"rank": 11, "worker-slow": 22, "worker-fast": 23},
            "window_metrics": {
                "domain_coherent": 1.0, "domain_procs_active": 1.0,
                "window_wall_time_proc_crit": 2.0,
                "domain_input_time_proc_crit": 1.0,
                "domain_decode_time_proc_crit": 0.0,
                "domain_comm_time_proc_crit": 0.0,
                "domain_compute_time_proc_crit": 1.0,
                "residual_time_proc_crit": -0.5,
                "epoch_nunique": 1.0, "step_nunique": 2.0,
                "step_min": 4.0, "step_max": 5.0,
            },
        },
    )
    fact = facts[("proc_name",)][0]
    assert fact.window.window == 12
    assert fact.evidence["metrics"]["reader_posix_read_ops_slope"] == pytest.approx(0.7)
    assert fact.evidence["metrics"]["selected_worker_pid"] == 22
    assert fact.evidence["metrics"]["epoch_nunique"] == pytest.approx(1.0)


def test_parked_p3_rank_thread_rule_uses_exact_name_pid_and_minimum_deficit():
    """P3 threads belong to rank processes; they are not DataLoader workers."""
    rule_file = (
        Path(__file__).resolve().parents[1]
        / "python/dftracer/analyzer/configs/fact_rules/p3-tf-rank-thread-slope.yaml"
    )
    engine = FactEngine.from_rule_file(
        str(rule_file), strict_time_semantics=True,
        allow_mixed_time_aggregates=False,
    )
    rows = pd.DataFrame({
        "reader_posix_read_count_frac_total": [0.4, 0.1, 0.3, 0.0, 0.2],
        "reader_posix_read_time_proc_frac_total": [0.5, 0.5, 0.3, 0.2, 0.1],
        "reader_posix_read_ops_slope": [0.8, 0.2, 0.1, 0.0, 2.0],
    }, index=[
        "app#node-a#101#1001",     # eligible, but not the minimum
        "app#node-a#101#1002",     # selected
        "app#node-b#101#1003",     # wrong K host
        "app#node-a#101#1004",     # zero normalized count share
        "app#node-a#101#1005",     # non-deficit
    ])
    control = {
        "domain_coherent": 1.0, "domain_procs_active": 1.0,
        "window_wall_time_proc_crit": 3.0,
        "domain_input_time_proc_crit": 1.0,
        "domain_decode_time_proc_crit": 0.0,
        "domain_comm_time_proc_crit": 0.0,
        "domain_compute_time_proc_crit": 2.0,
        "residual_time_proc_crit": 0.0,
        "epoch_nunique": 1.0, "step_nunique": 2.0,
        "step_min": 4.0, "step_max": 5.0,
    }
    context = {
        "native_window": 41, "native_host": "node-a", "rank_pids": [101],
        "row_pid_by_index": {
            "app#node-a#101#1001": 101,
            "app#node-a#101#1002": 101,
            "app#node-b#101#1003": 101,
            "app#node-a#101#1004": 101,
            "app#node-a#101#1005": 101,
        },
        "window_metrics": control,
    }
    facts = engine.evaluate({("proc_name",): rows}, raw_stats={}, context=context)
    fact = facts[("proc_name",)][0]
    metrics = fact.evidence["metrics"]
    assert fact.window.window == 41
    assert fact.window.view_type == "proc_name"
    assert metrics["selected_proc_name"] == "app#node-a#101#1002"
    assert metrics["selected_rank_pid"] == 101
    assert metrics["selected_thread_id"] == 1002
    assert metrics["selected_mapped_pid"] == 101
    assert metrics["reader_posix_read_ops_slope"] == pytest.approx(0.2)
    assert metrics["reader_posix_read_count_frac_total"] == pytest.approx(0.1)
    assert metrics["reader_posix_read_time_proc_frac_total"] == pytest.approx(0.5)
    assert fact.severity.score == pytest.approx(0.8)

    # Malformed, non-integer, mismapped, non-rank, zero/NaN, and >=1 candidates
    # all suppress instead of being converted into an issue.
    invalid = pd.DataFrame({
        "reader_posix_read_count_frac_total": [0.2] * 6,
        "reader_posix_read_time_proc_frac_total": [0.2] * 6,
        "reader_posix_read_ops_slope": [0.2, 0.2, 0.2, 0.0, float("nan"), 1.2],
    }, index=[
        "wrong#node-a#101#1", "app#node-a#notpid#1",
        "app#node-a#101#3", "app#node-a#101#4",
        "app#node-a#101#5", "app#node-a#101#6",
    ])
    invalid_context = dict(context)
    invalid_context["row_pid_by_index"] = {
        "app#node-a#101#3": 999, "app#node-a#101#4": 101,
        "app#node-a#101#5": 101, "app#node-a#101#6": 101,
    }
    assert engine.evaluate({("proc_name",): invalid}, raw_stats={}, context=invalid_context) == {}

    for invalid_control in (
        {"epoch_nunique": 2.0},
        {"step_nunique": 0.0},
        {"step_min": 6.0, "step_max": 5.0},
    ):
        incomplete_context = dict(context)
        incomplete_context["window_metrics"] = dict(control, **invalid_control)
        assert engine.evaluate({("proc_name",): rows}, raw_stats={}, context=incomplete_context) == {}
    missing_control = dict(context)
    missing_control["window_metrics"] = {
        key: value for key, value in control.items() if key != "step_max"
    }
    assert engine.evaluate({("proc_name",): rows}, raw_stats={}, context=missing_control) == {}


def test_rank_thread_selector_rejects_worker_selector_combination():
    with pytest.raises(FactRuleValidationError, match="worker_only and rank_thread_only"):
        build_fact_rule({
            "id": "invalid.selector.combination", "priority": 1,
            "source_view": "proc_name", "fact_type": "invalid",
            "worker_only": True, "rank_thread_only": True,
        })


def test_fact_engine_skips_rule_with_unresolved_identifier():
    unresolved_rule = build_fact_rule(
        raw_rule={
            "id": "rule.unresolved.identifier",
            "priority": 10,
            "source_view": "epoch",
            "fact_type": "invalid_fact",
            "required_metrics": ["epoch_time_max"],
            "when": "missing_metric > 0",
            "severity_score": "0.5",
        }
    )
    engine = FactEngine([unresolved_rule])
    flat_views = {
        ("epoch",): pd.DataFrame(
            {
                "epoch_time_max": [1.0, 2.0],
            },
            index=[1, 2],
        )
    }

    facts = engine.evaluate(flat_views=flat_views, raw_stats={"run_id": "run-123"})

    assert facts == {}


def test_fact_engine_can_load_packaged_dlio_rules_by_alias():
    engine = FactEngine.from_rule_file(
        "dlio",
        strict_time_semantics=True,
        allow_mixed_time_aggregates=False,
    )

    assert len(engine.rules) >= 5
    assert "epoch" in engine.rules_by_view
    assert "proc_name" in engine.rules_by_view


def test_fact_engine_sets_time_range_and_step_window_metadata():
    time_range_rule = build_fact_rule(
        raw_rule={
            "id": "rule.time.range.window",
            "priority": 10,
            "source_view": "time_range",
            "fact_type": "fetch_interval_pressure",
            "required_metrics": ["fetch_iter_time_frac_parent"],
            "when": "fetch_iter_time_frac_parent > 0.5",
            "severity_score": "0.8",
            "confidence": "0.7",
        }
    )
    step_rule = build_fact_rule(
        raw_rule={
            "id": "rule.step.window",
            "priority": 10,
            "source_view": "step",
            "fact_type": "step_compute_pressure",
            "required_metrics": ["compute_time_frac_parent"],
            "when": "compute_time_frac_parent >= 0.6",
            "severity_score": "0.6",
            "confidence": "0.9",
        }
    )
    engine = FactEngine([time_range_rule, step_rule])
    flat_views = {
        ("time_range",): pd.DataFrame(
            {
                "fetch_iter_time_frac_parent": [0.2, 0.7],
            },
            index=[0, 1],
        ),
        ("step",): pd.DataFrame(
            {
                "compute_time_frac_parent": [0.4, 0.65],
            },
            index=[3, 4],
        ),
    }

    facts = engine.evaluate(
        flat_views=flat_views,
        raw_stats={
            "run_id": "run-xyz",
            "time_granularity": 10.0,
            "time_resolution": 1_000_000.0,
        },
    )

    tr_fact = facts[("time_range",)][0]
    assert tr_fact.window.view_type == "time_range"
    assert tr_fact.window.epoch is None
    assert tr_fact.window.step is None
    assert tr_fact.window.t0_ns == 10_000_000_000
    assert tr_fact.window.t1_ns == 20_000_000_000

    step_fact = facts[("step",)][0]
    assert step_fact.window.view_type == "step"
    assert step_fact.window.step == 4
    assert step_fact.window.t0_ns is None
    assert step_fact.window.t1_ns is None


def test_fact_engine_emits_layer_scoped_facts_with_fillna0_support():
    rule = build_fact_rule(
        raw_rule={
            "id": "rule.reader.operation.imbalance",
            "priority": 10,
            "source_view": "epoch",
            "fact_type": "operation_imbalance",
            "scope_layer": "reader_posix",
            "required_metrics": [
                "reader_posix_read_count_sum",
                "reader_posix_write_count_sum",
            ],
            "derived_metrics": {
                "reader_posix_operation_imbalance_ratio": (
                    "abs(fillna0(reader_posix_read_count_sum) - fillna0(reader_posix_write_count_sum)) "
                    "/ max(min(fillna0(reader_posix_read_count_sum), fillna0(reader_posix_write_count_sum)), 1e-9)"
                ),
            },
            "when": "reader_posix_operation_imbalance_ratio > 0.10",
            "severity_score": "clip01((reader_posix_operation_imbalance_ratio - 0.10) / 0.90)",
            "confidence": "0.8",
            "opportunity_tags": ["read_write_rebalancing"],
        }
    )
    engine = FactEngine([rule])
    flat_views = {
        ("epoch",): pd.DataFrame(
            {
                "reader_posix_read_count_sum": pd.array([10.0], dtype="Float64"),
                "reader_posix_write_count_sum": pd.array([pd.NA], dtype="Float64"),
            },
            index=[3],
        )
    }

    facts = engine.evaluate(flat_views=flat_views, raw_stats={"run_id": "run-123"})

    epoch_facts = facts[("epoch",)]
    assert len(epoch_facts) == 1
    assert epoch_facts[0].scope.layer == "reader_posix"
    assert epoch_facts[0].window.epoch == 3
    assert epoch_facts[0].evidence["metrics"]["reader_posix_operation_imbalance_ratio"] > 1e6

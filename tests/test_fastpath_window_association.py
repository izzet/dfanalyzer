"""Regression contract for unkeyed streaming rows in an otherwise keyed window.

The completed control window is authoritative for membership.  It may supply an
epoch to unkeyed domain rows only when the keyed rows in that same window name
exactly one epoch; it must never infer/broadcast across multiple epochs.
"""

import dataclasses as dc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Match node_agent.bootstrap(): dftracer is an installed regular package while
# the live analyzer path is intentionally served from this worktree.
import dftracer

_SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "python" / "dftracer"
if str(_SOURCE_PACKAGE) not in dftracer.__path__:
    dftracer.__path__.insert(0, str(_SOURCE_PACKAGE))

from dftracer.analyzer.fastpath import FastPathPlan
from dftracer.analyzer.config import AnalyzerPresetConfigDLIOAILogging
from dftracer.analyzer.fact_engine import FactEngine


def _plan():
    plan = FastPathPlan.__new__(FastPathPlan)
    plan.view_types = ["epoch"]
    plan.terms = [
        ("window_wall", 'func_name == "window.stop"', False),
        ("compute", 'func_name == "compute"', False),
        ("comm", 'func_name == "comm"', False),
    ]
    plan._mask_cache = {}
    return plan


def _row(epoch, pid, proc_name, func_name, time):
    return {
        "epoch": epoch, "pid": pid, "proc_name": proc_name,
        "cat": "test", "io_cat": "", "func_name": func_name,
        "time": time, "size": 0,
    }


def test_single_keyed_epoch_attaches_unkeyed_wall_only_to_its_domain_matrix():
    traces = pd.DataFrame([
        _row(7, 100, "rank0-main", "compute", 4.0),
        _row(np.nan, 100, "rank0-main", "window.stop", 10.0),
        # A worker has no wall and must not become a selectable active rank.
        _row(np.nan, 200, "worker", "compute", 50.0),
    ])

    view = _plan().flat_views(traces)[("epoch",)]
    row = view.loc[7]
    assert row["domain_coherent"] == 1.0
    assert row["domain_procs_active"] == 1.0
    assert row["window_wall_time_proc_crit"] == 10.0
    assert row["domain_compute_time_proc_crit"] == 4.0
    assert row["residual_time_proc_crit"] == 6.0
    # Legacy aggregate output remains keyed-only: the worker was not silently
    # injected into the epoch view.
    assert row["compute_time_sum"] == 4.0


def test_multiple_keyed_epochs_leave_unkeyed_wall_unassociated_and_incoherent():
    traces = pd.DataFrame([
        _row(7, 100, "rank0-main", "compute", 4.0),
        _row(8, 100, "rank0-main", "compute", 5.0),
        _row(np.nan, 100, "rank0-main", "window.stop", 10.0),
    ])

    view = _plan().flat_views(traces)[("epoch",)]
    assert view.loc[7, "domain_coherent"] == 0.0
    assert view.loc[8, "domain_coherent"] == 0.0
    assert view.loc[7, "compute_time_sum"] == 4.0
    assert view.loc[8, "compute_time_sum"] == 5.0


def test_dlio_signal_rule_uses_fastpath_but_legacy_wall_gate_stays_closed():
    """The established signal rule is an explicit fast-path diagnostic, not dlio.yaml.

    The conservative association supplies a coherent vector but preserves legacy
    aggregates, so the rule's legacy-wall predicate remains closed.  This guards
    the exact configuration proposed for the next diagnostic smoke without
    implying equivalence to the default rule set.
    """
    preset = AnalyzerPresetConfigDLIOAILogging()
    facts = FactEngine.from_rule_file(
        "dlio-signal",
        strict_time_semantics=True,
        allow_mixed_time_aggregates=False,
    )
    plan = FastPathPlan.build(preset, ["epoch"], list(preset.layer_defs), facts)
    assert plan is not None

    traces = pd.DataFrame([
        _row(7, 100, "rank0-main", "compute", 4.0) | {"cat": "compute"},
        _row(np.nan, 100, "rank0-main", "window.stop", 10.0) | {"cat": "window"},
    ])
    view = plan.flat_views(traces)[("epoch",)]
    row = view.loc[7]
    assert row["domain_epoch_association_unambiguous"] == 1.0
    assert row["domain_unkeyed_rows_included"] == 1.0
    assert pd.isna(row["window_wall_time_proc_max"])
    assert row["compute_time_sum"] == 4.0
    assert row["window_wall_time_proc_crit"] == 10.0

    emitted = facts.evaluate({("epoch",): view}, raw_stats={})
    assert emitted == {}


def test_parked_worker_slope_rule_remains_on_fastpath_with_native_window_gate():
    preset = AnalyzerPresetConfigDLIOAILogging()
    rule_file = (
        Path(__file__).resolve().parents[1]
        / "python/dftracer/analyzer/configs/fact_rules/dlio-wisio-worker-slope.yaml"
    )
    facts = FactEngine.from_rule_file(
        str(rule_file), strict_time_semantics=True,
        allow_mixed_time_aggregates=False,
    )
    plan = FastPathPlan.build(
        preset, ["window", "proc_name"], list(preset.layer_defs), facts
    )
    assert plan is not None
    assert {"epoch_nunique", "step_nunique", "step_min", "step_max"}.issubset(
        plan.produced_columns
    )


def test_p3_rank_thread_rule_keeps_canonical_shares_on_fastpath():
    """P3 needs the canonical share fields as well as the derived slope.

    If this plan declines, the full path is correct but cannot construct the
    per-PID coherent vector.  That must remain an explicit regression, never a
    reason to weaken the detector's coherence gate.
    """
    preset = AnalyzerPresetConfigDLIOAILogging()
    rule_file = (
        Path(__file__).resolve().parents[1]
        / "python/dftracer/analyzer/configs/fact_rules/p3-tf-rank-thread-slope.yaml"
    )
    facts = FactEngine.from_rule_file(
        str(rule_file), strict_time_semantics=True,
        allow_mixed_time_aggregates=False,
    )
    plan = FastPathPlan.build(
        preset, ["window", "proc_name"], list(preset.layer_defs), facts
    )
    assert plan is not None
    assert {
        "reader_posix_read_count_frac_total",
        "reader_posix_read_time_proc_frac_total",
        "reader_posix_read_ops_slope",
    }.issubset(plan.produced_columns | plan.proc_name_only_columns)

    # Those shares exist only on proc_name. A window rule requesting one must
    # decline rather than borrow a sibling proc view's metric.
    facts.rules = [dc.replace(facts.rules[0], source_view="window")]
    assert FastPathPlan.build(
        preset, ["window", "proc_name"], list(preset.layer_defs), facts
    ) is None

    # The derived slope has the same view restriction even when a rule asks
    # for no share inputs.
    slope = "reader_posix_read_ops_slope"
    facts.rules = [dc.replace(
        facts.rules[0], required_metrics=[slope], referenced_columns={slope},
        derived_metrics={}, when=f"{slope} > 0.0", severity_score=slope,
    )]
    assert FastPathPlan.build(
        preset, ["window", "proc_name"], list(preset.layer_defs), facts
    ) is None


def test_p3_fastpath_builds_coherent_window_and_rank_thread_shares():
    preset = AnalyzerPresetConfigDLIOAILogging()
    rule_file = (
        Path(__file__).resolve().parents[1]
        / "python/dftracer/analyzer/configs/fact_rules/p3-tf-rank-thread-slope.yaml"
    )
    facts = FactEngine.from_rule_file(
        str(rule_file), strict_time_semantics=True,
        allow_mixed_time_aggregates=False,
    )
    plan = FastPathPlan.build(
        preset, ["window", "proc_name"], list(preset.layer_defs), facts
    )
    assert plan is not None

    def event(pid, tid, category, name, seconds, io_cat=0):
        return {
            "window": 7, "epoch": 1, "step": 5, "pid": pid,
            "proc_name": f"app#node-a#{pid}#{tid}", "cat": category,
            "func_name": name, "io_cat": io_cat, "time": seconds, "size": 0,
        }

    traces = pd.DataFrame([
        event(100, 100, "window", "window.stop", 10.0),
        event(100, 100, "compute", "compute", 6.0),
        event(100, 101, "posix_reader", "pread", 6.0, io_cat=1),
        event(101, 101, "window", "window.stop", 10.0),
        event(101, 101, "compute", "compute", 5.0),
        event(101, 102, "posix_reader", "pread", 1.0, io_cat=1),
        event(101, 102, "posix_reader", "pread", 1.0, io_cat=1),
    ])
    views = plan.flat_views(traces, control_pids=[100, 101])
    window = views[("window",)].loc[7]
    assert window["domain_coherent"] == 1.0
    assert window["domain_procs_active"] == 2.0
    proc = views[("proc_name",)]
    for column in (
        "reader_posix_read_count_frac_total",
        "reader_posix_read_time_proc_frac_total",
        "reader_posix_read_ops_slope",
    ):
        assert np.isfinite(proc[column].dropna()).all()


def test_fastpath_uses_canonical_wisio_slope_on_proc_name_view():
    plan = FastPathPlan.__new__(FastPathPlan)
    plan.view_types = ["proc_name"]
    plan.terms = [("reader_posix_read", 'func_name == "read"', False)]
    plan._mask_cache = {}
    traces = pd.DataFrame([
        _row(1, 100, "worker-a", "read", 3.0),
        _row(1, 200, "worker-b", "read", 1.0),
        _row(1, 200, "worker-b", "read", 0.0),
        _row(1, 200, "worker-b", "read", 0.0),
    ])
    view = plan.flat_views(traces)[("proc_name",)]
    # worker-a has 1/4 of the calls but 3/4 of the family time: (1/4)/(3/4).
    assert view.loc["worker-a", "reader_posix_read_ops_slope"] == pytest.approx(1.0 / 3.0)
    assert view.loc["worker-b", "reader_posix_read_ops_slope"] == pytest.approx(3.0)


def test_tf_slope_exposure_rule_is_served_by_fastpath_with_stall_coverage():
    """Regression for the 2026-09-25 loop smoke: an unsupported_metrics decline
    on stall_storage_rank_coverage silently demoted the rule to the full path,
    where every window is incoherent and every fact is suppressed."""
    preset = AnalyzerPresetConfigDLIOAILogging()
    rule_file = (
        Path(__file__).resolve().parents[1]
        / "python/dftracer/analyzer/configs/fact_rules/liveflow-tf-input-slope-exposure.yaml"
    )
    facts = FactEngine.from_rule_file(
        str(rule_file), strict_time_semantics=True, allow_mixed_time_aggregates=False)
    assert FastPathPlan.build(
        preset, ["window", "proc_name"], list(preset.layer_defs), facts) is not None


def test_tf_input_and_decode_rule_pair_is_served_by_fastpath():
    """Both families (reader_posix_read and the TF `reader_decode` region) on one fast-path plan."""
    preset = AnalyzerPresetConfigDLIOAILogging()
    rule_file = (
        Path(__file__).resolve().parents[1]
        / "python/dftracer/analyzer/configs/fact_rules/liveflow-tf-input-decode-slope-exposure.yaml"
    )
    facts = FactEngine.from_rule_file(
        str(rule_file), strict_time_semantics=True, allow_mixed_time_aggregates=False)
    assert len(facts.rules) == 2
    plan = FastPathPlan.build(preset, ["window", "proc_name"], list(preset.layer_defs), facts)
    assert plan is not None
    assert "reader_decode_ops_slope" in plan.proc_name_only_columns


def test_tf_skew_rule_is_served_by_fastpath():
    preset = AnalyzerPresetConfigDLIOAILogging()
    rule_file = (
        Path(__file__).resolve().parents[1]
        / "python/dftracer/analyzer/configs/fact_rules/liveflow-tf-input-skew-slope-exposure.yaml"
    )
    facts = FactEngine.from_rule_file(
        str(rule_file), strict_time_semantics=True, allow_mixed_time_aggregates=False)
    assert FastPathPlan.build(
        preset, ["window", "proc_name"], list(preset.layer_defs), facts) is not None

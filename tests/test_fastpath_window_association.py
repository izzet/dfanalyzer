"""Regression contract for unkeyed streaming rows in an otherwise keyed window.

The completed control window is authoritative for membership.  It may supply an
epoch to unkeyed domain rows only when the keyed rows in that same window name
exactly one epoch; it must never infer/broadcast across multiple epochs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

# Match the node agent bootstrap: use this worktree's analyzer code while the
# extension package itself remains installed in the supported venv.
import dftracer

_SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "python" / "dftracer"
if str(_SOURCE_PACKAGE) not in dftracer.__path__:
    dftracer.__path__.insert(0, str(_SOURCE_PACKAGE))

from dftracer.analyzer.analyzer import Analyzer, attach_window_horizon, window_horizon_metrics
from dftracer.analyzer.types import (
    AnalysisFact,
    FactProvenance,
    FactScope,
    FactWindow,
)


def _boundary(**overrides):
    event = {
        "name": "window.stop",
        "epoch_index": 2,
        "epochs_total": 5,
        "window_index": 7,
        "windows_per_epoch": 4,
    }
    event.update(overrides)
    return event


def test_synthetic_window_event_publishes_remaining_horizon_to_fact_metrics():
    metrics = window_horizon_metrics([_boundary()])
    assert metrics == {
        "epoch_index": 2,
        "epochs_total": 5,
        "window_index": 7,
        "windows_per_epoch": 4,
        "epochs_remaining": 3,
        "windows_remaining": 13,
    }

    fact = AnalysisFact(
        fact_type="input_compute_signal",
        window=FactWindow(),
        scope=FactScope(),
        evidence={"metrics": {"active_time": 0.045}},
        provenance=FactProvenance(rule_id="dlio.input_compute_signal.v1", rule_version="1"),
    )
    result = SimpleNamespace(
        flat_views={("epoch",): pd.DataFrame({"active_time": [0.045]})},
        iter_analysis_facts=lambda: iter([(("epoch",), fact)]),
    )

    attach_window_horizon(result, metrics)

    assert fact.evidence["metrics"]["epochs_remaining"] == 3
    assert fact.evidence["metrics"]["windows_remaining"] == 13
    assert result.flat_views[("epoch",)].loc[0, "epochs_remaining"] == 3


def test_incomplete_or_disagreeing_horizon_is_not_invented():
    assert window_horizon_metrics([_boundary(epochs_total=None)]) == {}
    assert window_horizon_metrics([_boundary(), _boundary(epoch_index=3)]) == {}
    # A native control index outside its own declared cadence cannot mean
    # "the run is genuinely over".  It is an invalid source contract and must
    # not become a misleading zero horizon that authorizes a fake decline.
    assert window_horizon_metrics([_boundary(window_index=21)]) == {}


def test_native_window_identity_preserves_workload_epoch_and_step():
    original = {"epoch": 3, "step": 17, "pid": 42}

    native = Analyzer._attach_stream_window(original, 9, native_window=True)
    assert native == {"epoch": 3, "step": 17, "pid": 42, "window": 9}

    # The non-native path is frozen only as a compatibility baseline for the
    # forthcoming parity test; production defaults are not changed here.
    legacy = Analyzer._attach_stream_window(original, 9, native_window=False)
    assert legacy == {"epoch": 9, "step": 9, "pid": 42, "window": 9}


def test_step_horizon_forwards_epoch_only_for_verification():
    step = dict(configured_total_steps=1000, completed_steps=40, configured_window_steps=20,
                planner_history_ready=0)
    metrics = window_horizon_metrics([_boundary(epoch_index=0, **step)])
    assert metrics['verification_epoch_index'] == 0
    assert 'epoch_index' not in metrics  # preserve the planner's existing step-based regime key
    fact = AnalysisFact(fact_type='input_compute_signal', window=FactWindow(), scope=FactScope(),
                        evidence={'metrics': {}},
                        provenance=FactProvenance(rule_id='dlio.input_compute_signal.v1', rule_version='1'))
    result = SimpleNamespace(flat_views={('window',): pd.DataFrame({'active_time': [.1]})},
                             iter_analysis_facts=lambda: iter([(('',), fact)]))
    attach_window_horizon(result, metrics)
    assert fact.evidence['metrics']['verification_epoch_index'] == 0


def test_missing_or_disagreeing_step_epoch_stays_absent():
    step = dict(name='window.stop', configured_total_steps=1000, completed_steps=40,
                configured_window_steps=20)
    assert 'verification_epoch_index' not in window_horizon_metrics([step])
    assert 'verification_epoch_index' not in window_horizon_metrics([
        dict(step, epoch_index=1), dict(step, epoch_index=2)])

from types import SimpleNamespace

import pandas as pd

from dftracer.analyzer.analyzer import attach_window_horizon, window_horizon_metrics
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

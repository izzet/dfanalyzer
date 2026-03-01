import pandas as pd
import pytest

from dftracer.analyzer.types import (
    AnalysisFact,
    AnalyzerResultType,
    FactProvenance,
    FactScope,
    FactSeverity,
    FactWindow,
    RawStats,
)


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _build_fact(view_type: str, entity: str, run_id: str = "run-abc") -> AnalysisFact:
    fact = AnalysisFact(
        fact_type="test_fact",
        window=FactWindow(run_id=run_id, view_type=view_type, epoch=1 if view_type == "epoch" else None),
        scope=FactScope(entity=entity, rank_set="all"),
        evidence={"metrics": {"x": 1.0}},
        severity=FactSeverity(score=0.5, label="medium"),
        confidence=0.8,
        opportunity_tags=["tag-a"],
        provenance=FactProvenance(
            rule_id="rule.test",
            rule_version="1.0.0",
            view_key=[view_type],
        ),
    )
    fact.finalize_id()
    return fact


def test_analyzer_result_to_fact_envelope_contract():
    result = AnalyzerResultType(
        _hlms={},
        _main_views={},
        _metric_boundaries={},
        checkpoint_dir=".",
        flat_views={("epoch",): pd.DataFrame({"epoch_time_max": [1.0]}, index=[1])},
        layers=["epoch", "fetch_data"],
        raw_stats=RawStats(
            job_time=10,
            time_granularity=10,
            time_resolution=1_000_000,
            total_event_count=100,
            unique_file_count=2,
            unique_host_count=1,
            unique_process_count=4,
        ),
        view_types=["epoch", "time_range"],
        views={},
        analysis_facts={
            ("epoch",): [_build_fact("epoch", "1")],
            ("time_range",): [_build_fact("time_range", "0")],
        },
    )

    envelope = result.to_fact_envelope()
    payload = envelope.to_dict()

    assert payload["schema_version"] == "analyzer.fact-envelope.v1"
    assert payload["context"]["run_id"] == "run-abc"
    assert payload["context"]["view_types"] == ["epoch", "time_range"]
    assert payload["context"]["time_granularity"] == 10.0
    assert payload["context"]["total_event_count"] == 100
    assert payload["fact_count_by_view"]["epoch"] == 1
    assert payload["fact_count_by_view"]["time_range"] == 1
    assert payload["context"]["window_type_counts"]["epoch"] == 1
    assert payload["context"]["window_type_counts"]["time_range"] == 1
    assert len(payload["facts"]) == 2

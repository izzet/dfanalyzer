"""Tests for view_types matching, emit_mode (detail/aggregate), and the
aggregate reduction in the fact engine."""
import pandas as pd

from dftracer.analyzer.fact_engine import FactEngine
from dftracer.analyzer.fact_rules import build_fact_rule


def _rule(emit_mode=None, view_types=None):
    raw = {
        "id": "test.metadata_dominance.v1",
        "fact_type": "metadata_dominance",
        "scope_layer": "reader_posix",
        "required_metrics": ["reader_posix_metadata_time_frac_parent"],
        "when": "reader_posix_metadata_time_frac_parent > 0.5",
        "severity_score": "reader_posix_metadata_time_frac_parent",
        "opportunity_tags": ["metadata_reduction"],
    }
    if emit_mode is not None:
        raw["emit_mode"] = emit_mode
    if view_types is not None:
        raw["view_types"] = view_types
    return build_fact_rule(raw, strict_time_semantics=False, allow_mixed_time_aggregates=True)


def _file_view():
    # 5 files; 3 are over the 0.5 threshold (0.9, 0.8, 0.7)
    return pd.DataFrame(
        {"reader_posix_metadata_time_frac_parent": [0.9, 0.8, 0.7, 0.3, 0.1]},
        index=["/d/f0.npz", "/d/f1.npz", "/d/f2.npz", "/d/f3.npz", "/d/f4.npz"],
    )


_RAW_STATS = {"run_id": "r1", "time_granularity": 1.0, "time_resolution": 1.0}


def _flatten(facts):
    return [f for fs in facts.values() for f in fs]


def test_aggregate_mode_emits_one_rolled_up_fact():
    engine = FactEngine([_rule(emit_mode="aggregate", view_types=["file_name"])])
    flat = _flatten(engine.evaluate({("file_name",): _file_view()}, _RAW_STATS))

    assert len(flat) == 1
    fact = flat[0]
    assert fact.fact_type == "metadata_dominance"
    assert fact.window.view_type == "file_name"
    assert fact.scope.entity is None  # whole-view aggregate
    m = fact.evidence["metrics"]
    assert m["affected_count"] == 3 and m["total_count"] == 5
    assert abs(m["affected_fraction"] - 0.6) < 1e-9
    assert abs(fact.severity.score - 0.9) < 1e-9  # max matched severity
    assert [t["entity"] for t in fact.evidence["top_k"]] == ["/d/f0.npz", "/d/f1.npz", "/d/f2.npz"]


def test_detail_mode_emits_per_entity_facts():
    engine = FactEngine([_rule(emit_mode="detail", view_types=["file_name"])])
    flat = _flatten(engine.evaluate({("file_name",): _file_view()}, _RAW_STATS))
    assert len(flat) == 3
    assert sorted(f.scope.entity for f in flat) == ["/d/f0.npz", "/d/f1.npz", "/d/f2.npz"]


def test_default_emit_mode_used_when_rule_unset():
    engine = FactEngine([_rule(view_types=["file_name"])], default_emit_mode="aggregate")
    flat = _flatten(engine.evaluate({("file_name",): _file_view()}, _RAW_STATS))
    assert len(flat) == 1 and flat[0].scope.entity is None


def test_rule_emit_mode_overrides_default():
    engine = FactEngine([_rule(emit_mode="detail", view_types=["file_name"])],
                        default_emit_mode="aggregate")
    flat = _flatten(engine.evaluate({("file_name",): _file_view()}, _RAW_STATS))
    assert len(flat) == 3  # rule's detail wins over the aggregate default


def test_view_types_restricts_views():
    engine = FactEngine([_rule(emit_mode="aggregate", view_types=["file_name"])])
    assert engine.evaluate({("epoch",): _file_view()}, _RAW_STATS) == {}


def test_empty_view_types_applies_to_all_views():
    engine = FactEngine([_rule(emit_mode="aggregate")])  # no view_types -> all views
    flat = _flatten(engine.evaluate({("proc_name",): _file_view()}, _RAW_STATS))
    assert len(flat) == 1 and flat[0].window.view_type == "proc_name"


def test_no_matches_emits_nothing():
    engine = FactEngine([_rule(emit_mode="aggregate", view_types=["file_name"])])
    below = pd.DataFrame(
        {"reader_posix_metadata_time_frac_parent": [0.1, 0.2]}, index=["/a", "/b"]
    )
    assert engine.evaluate({("file_name",): below}, _RAW_STATS) == {}

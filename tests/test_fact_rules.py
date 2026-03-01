import pytest

from dftracer.analyzer.fact_rules import FactRuleValidationError, build_fact_rule


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def test_build_fact_rule_rejects_mixed_time_aggregates_by_default():
    raw_rule = {
        "id": "mixed.time.agg",
        "fact_type": "mixed_time_agg",
        "required_metrics": ["epoch_time_sum", "epoch_time_max"],
        "when": "epoch_time_sum > epoch_time_max",
        "severity_score": "0.5",
    }

    with pytest.raises(FactRuleValidationError, match="mixes time aggregates"):
        build_fact_rule(
            raw_rule=raw_rule,
            strict_time_semantics=True,
            allow_mixed_time_aggregates=False,
        )


def test_build_fact_rule_allows_mixed_time_aggregates_when_configured():
    raw_rule = {
        "id": "mixed.time.agg.allowed",
        "fact_type": "mixed_time_agg",
        "required_metrics": ["epoch_time_sum", "epoch_time_max"],
        "when": "epoch_time_sum > epoch_time_max",
        "severity_score": "0.5",
    }

    rule = build_fact_rule(
        raw_rule=raw_rule,
        strict_time_semantics=True,
        allow_mixed_time_aggregates=True,
    )

    assert rule.allow_mixed is True
    assert rule.uses_time_aggregates == {"sum", "max"}


def test_build_fact_rule_rejects_invalid_emit_mode():
    raw_rule = {
        "id": "invalid.emit.mode",
        "fact_type": "invalid_emit_mode",
        "required_metrics": ["epoch_time_max"],
        "when": "epoch_time_max > 0",
        "severity_score": "0.3",
        "emit_mode": "aggregate",
    }

    with pytest.raises(FactRuleValidationError, match="invalid emit_mode"):
        build_fact_rule(raw_rule=raw_rule)

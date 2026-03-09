import dataclasses as dc
import re
from typing import Any, Dict, Iterable, List, Optional, Set


IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
TIME_SUM_SUFFIX = "_time_sum"
TIME_MAX_SUFFIX = "_time_max"


class FactRuleValidationError(ValueError):
    pass


@dc.dataclass(frozen=True)
class FactRule:
    id: str
    priority: int
    source_view: str
    fact_type: str
    scope_layer: Optional[str]
    required_metrics: List[str]
    derived_metrics: Dict[str, str]
    when: str
    severity_score: str
    confidence: Optional[str]
    opportunity_tags: List[str]
    emit_mode: str = "per_row"
    rule_version: str = "1.0.0"
    allow_mixed: bool = False
    referenced_columns: Set[str] = dc.field(default_factory=set)
    uses_time_aggregates: Set[str] = dc.field(default_factory=set)

    def projected_columns(self) -> List[str]:
        return sorted(set(self.required_metrics).union(self.referenced_columns))


def extract_identifiers(expressions: Iterable[str]) -> Set[str]:
    identifiers: Set[str] = set()
    for expr in expressions:
        if not expr:
            continue
        identifiers.update(IDENTIFIER_PATTERN.findall(expr))
    return identifiers


def detect_time_aggregates(columns: Iterable[str]) -> Set[str]:
    aggs = set()
    for column in columns:
        if column.endswith(TIME_SUM_SUFFIX):
            aggs.add("sum")
        elif column.endswith(TIME_MAX_SUFFIX):
            aggs.add("max")
    return aggs


def build_fact_rule(
    raw_rule: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
    *,
    strict_time_semantics: bool = True,
    allow_mixed_time_aggregates: bool = False,
) -> FactRule:
    defaults = defaults or {}

    rule_id = raw_rule["id"]
    source_view = raw_rule.get("source_view", defaults.get("source_view", "epoch"))
    scope_layer = raw_rule.get("scope_layer")
    required_metrics = list(raw_rule.get("required_metrics", []))
    derived_metrics = dict(raw_rule.get("derived_metrics", {}))
    when = raw_rule.get("when", "False")
    severity_score = raw_rule.get("severity_score", "0.0")
    confidence = raw_rule.get("confidence", defaults.get("confidence"))
    opportunity_tags = list(raw_rule.get("opportunity_tags", []))
    emit_mode = str(raw_rule.get("emit_mode", defaults.get("emit_mode", "per_row")))
    allow_mixed = bool(raw_rule.get("allow_mixed", False))
    if emit_mode not in {"per_row", "window"}:
        raise FactRuleValidationError(
            f"Rule '{rule_id}' has invalid emit_mode '{emit_mode}'. Expected one of ['per_row', 'window']."
        )

    expressions = [when, severity_score]
    if confidence:
        expressions.append(confidence)
    expressions.extend(derived_metrics.values())
    referenced_columns = extract_identifiers(expressions)
    referenced_columns.update(required_metrics)
    time_aggs = detect_time_aggregates(referenced_columns)

    allow_mixed_effective = allow_mixed or allow_mixed_time_aggregates
    if strict_time_semantics and len(time_aggs) > 1 and not allow_mixed_effective:
        raise FactRuleValidationError(
            (
                f"Rule '{rule_id}' mixes time aggregates {sorted(time_aggs)}. "
                "Use a single aggregate basis or set allow_mixed=true."
            )
        )

    return FactRule(
        id=rule_id,
        priority=int(raw_rule.get("priority", 0)),
        source_view=str(source_view),
        fact_type=str(raw_rule["fact_type"]),
        scope_layer=str(scope_layer) if scope_layer is not None else None,
        required_metrics=required_metrics,
        derived_metrics=derived_metrics,
        when=str(when),
        severity_score=str(severity_score),
        confidence=str(confidence) if confidence is not None else None,
        opportunity_tags=opportunity_tags,
        emit_mode=emit_mode,
        rule_version=str(raw_rule.get("rule_version", defaults.get("rule_version", "1.0.0"))),
        allow_mixed=allow_mixed_effective,
        referenced_columns=referenced_columns,
        uses_time_aggregates=time_aggs,
    )

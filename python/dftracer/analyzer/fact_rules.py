import dataclasses as dc
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
TIME_SUM_SUFFIX = "_time_sum"
TIME_MAX_SUFFIX = "_time_max"


class FactRuleValidationError(ValueError):
    pass


@dc.dataclass(frozen=True)
class FactRule:
    id: str
    priority: int
    view_types: Tuple[str, ...]  # which views this rule applies to; empty => all computed views
    fact_type: str
    scope_layer: Optional[str]
    required_metrics: List[str]
    derived_metrics: Dict[str, str]
    when: str
    severity_score: str
    confidence: Optional[str]
    opportunity_tags: List[str]
    suppresses_tags: List[str] = dc.field(default_factory=list)
    emit_mode: Optional[str] = None  # 'detail' | 'aggregate'; None => inherit facts.emit_mode
    rule_version: str = "1.0.0"
    allow_mixed: bool = False
    referenced_columns: Set[str] = dc.field(default_factory=set)
    uses_time_aggregates: Set[str] = dc.field(default_factory=set)

    def projected_columns(self) -> List[str]:
        return sorted(set(self.required_metrics).union(self.referenced_columns))

    def matches_view(self, view_type: str) -> bool:
        """Empty view_types => the rule applies to every computed view."""
        return (not self.view_types) or (view_type in self.view_types)


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
    # view_types: which views the rule applies to; empty => all computed views.
    # `source_view` (single string) is accepted as a transitional alias.
    raw_views = raw_rule.get(
        "view_types",
        raw_rule.get("source_view", defaults.get("view_types", defaults.get("source_view"))),
    )
    if raw_views is None:
        view_types: Tuple[str, ...] = ()
    elif isinstance(raw_views, str):
        view_types = (raw_views,)
    else:
        view_types = tuple(raw_views)
    scope_layer = raw_rule.get("scope_layer")
    required_metrics = list(raw_rule.get("required_metrics", []))
    derived_metrics = dict(raw_rule.get("derived_metrics", {}))
    when = raw_rule.get("when", "False")
    severity_score = raw_rule.get("severity_score", "0.0")
    confidence = raw_rule.get("confidence", defaults.get("confidence"))
    opportunity_tags = list(raw_rule.get("opportunity_tags", []))
    emit_mode_raw = raw_rule.get("emit_mode", defaults.get("emit_mode"))
    emit_mode = str(emit_mode_raw) if emit_mode_raw is not None else None
    allow_mixed = bool(raw_rule.get("allow_mixed", False))
    if emit_mode is not None and emit_mode not in {"detail", "aggregate"}:
        raise FactRuleValidationError(
            f"Rule '{rule_id}' has invalid emit_mode '{emit_mode}'. Expected one of ['detail', 'aggregate']."
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

    suppresses_tags = list(raw_rule.get("suppresses_tags", []))

    return FactRule(
        id=rule_id,
        priority=int(raw_rule.get("priority", 0)),
        view_types=view_types,
        fact_type=str(raw_rule["fact_type"]),
        scope_layer=str(scope_layer) if scope_layer is not None else None,
        required_metrics=required_metrics,
        derived_metrics=derived_metrics,
        when=str(when),
        severity_score=str(severity_score),
        confidence=str(confidence) if confidence is not None else None,
        opportunity_tags=opportunity_tags,
        suppresses_tags=suppresses_tags,
        emit_mode=emit_mode,
        rule_version=str(raw_rule.get("rule_version", defaults.get("rule_version", "1.0.0"))),
        allow_mixed=allow_mixed_effective,
        referenced_columns=referenced_columns,
        uses_time_aggregates=time_aggs,
    )

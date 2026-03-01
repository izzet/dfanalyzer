import dataclasses as dc
import importlib.util
import math
import os
from pathlib import Path
import pandas as pd
import structlog
from typing import Any, Dict, List, Mapping, Optional
from omegaconf import OmegaConf

from .fact_rules import FactRule, FactRuleValidationError, build_fact_rule
from .types import AnalysisFact, FactProvenance, FactScope, FactSeverity, FactWindow, ViewKey


logger = structlog.get_logger()
HAS_NUMEXPR = importlib.util.find_spec("numexpr") is not None

RESERVED_IDENTIFIERS = {
    "True",
    "False",
    "None",
    "and",
    "or",
    "not",
    "max",
    "min",
    "abs",
    "clip01",
    "math",
}


def _as_dict(raw_stats: Any) -> Dict[str, Any]:
    if raw_stats is None:
        return {}
    if isinstance(raw_stats, Mapping):
        return dict(raw_stats)
    if dc.is_dataclass(raw_stats):
        return dc.asdict(raw_stats)
    return {}


def _to_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _to_series(df: pd.DataFrame, value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series([value] * len(df), index=df.index)


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _severity_label(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.60:
        return "high"
    if score >= 0.30:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def _clip01(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value.clip(lower=0.0, upper=1.0)
    try:
        return min(1.0, max(0.0, float(value)))
    except Exception:
        return value


def _eval_python_expr(df: pd.DataFrame, expression: str) -> pd.Series:
    local_env: Dict[str, Any] = {column: df[column] for column in df.columns}
    local_env.update(
        {
            "max": max,
            "min": min,
            "abs": abs,
            "clip01": _clip01,
            "math": math,
        }
    )
    result = eval(expression, {"__builtins__": {}}, local_env)
    return _to_series(df, result)


def _eval_expr(df: pd.DataFrame, expression: str) -> pd.Series:
    # Use numexpr first when available for speed; then pandas python engine;
    # finally restricted eval for aggregate-style expressions.
    if HAS_NUMEXPR:
        try:
            return _to_series(df, df.eval(expression, engine="numexpr"))
        except Exception:
            pass
    try:
        return _to_series(df, df.eval(expression, engine="python"))
    except Exception:
        return _eval_python_expr(df, expression)


class FactEngine:
    def __init__(self, rules: List[FactRule]):
        self.rules = sorted(rules, key=lambda rule: rule.priority, reverse=True)
        self.rules_by_view: Dict[str, List[FactRule]] = {}
        for rule in self.rules:
            self.rules_by_view.setdefault(rule.source_view, []).append(rule)

    @classmethod
    def from_rule_file(
        cls,
        rule_file: str,
        *,
        strict_time_semantics: bool,
        allow_mixed_time_aggregates: bool,
    ) -> "FactEngine":
        if not rule_file:
            logger.info("facts.rules.none")
            return cls([])
        resolved_rule_file = cls._resolve_rule_file(rule_file)
        if not os.path.exists(resolved_rule_file):
            raise FactRuleValidationError(f"Fact rule file does not exist: {resolved_rule_file}")

        config = OmegaConf.load(resolved_rule_file)
        config_obj = OmegaConf.to_object(config)
        defaults = dict(config_obj.get("defaults", {}))
        raw_rules = list(config_obj.get("rules", []))

        compiled_rules: List[FactRule] = []
        for raw_rule in raw_rules:
            try:
                compiled_rules.append(
                    build_fact_rule(
                        raw_rule=raw_rule,
                        defaults=defaults,
                        strict_time_semantics=strict_time_semantics,
                        allow_mixed_time_aggregates=allow_mixed_time_aggregates,
                    )
                )
            except FactRuleValidationError as error:
                logger.error("facts.rule.invalid", rule_id=raw_rule.get("id"), error=str(error))
                raise

        logger.info("facts.rules.loaded", rule_file=resolved_rule_file, count=len(compiled_rules))
        return cls(compiled_rules)

    @staticmethod
    def _resolve_rule_file(rule_file: str) -> str:
        path = Path(rule_file)
        candidates: List[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(Path.cwd() / path)
            if path.suffix == "":
                candidates.append(Path.cwd() / f"{rule_file}.yaml")
                candidates.append(Path.cwd() / f"{rule_file}.yml")
            package_rules = Path(__file__).resolve().parent / "configs" / "fact_rules"
            candidates.append(package_rules / path)
            if path.suffix == "":
                candidates.append(package_rules / f"{rule_file}.yaml")
                candidates.append(package_rules / f"{rule_file}.yml")

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0]) if candidates else rule_file

    def evaluate(
        self,
        flat_views: Dict[ViewKey, pd.DataFrame],
        raw_stats: Any,
    ) -> Dict[ViewKey, List[AnalysisFact]]:
        if not self.rules:
            return {}

        raw_stats_dict = _as_dict(raw_stats)
        all_facts: Dict[ViewKey, List[AnalysisFact]] = {}
        for view_key, view_df in flat_views.items():
            source_view = view_key[-1]
            view_rules = self.rules_by_view.get(source_view, [])
            if not view_rules:
                continue
            facts = self._evaluate_view_rules(
                rule_df=view_df,
                rules=view_rules,
                raw_stats=raw_stats_dict,
                view_key=view_key,
            )
            if facts:
                all_facts[view_key] = facts
        return all_facts

    def _evaluate_view_rules(
        self,
        rule_df: pd.DataFrame,
        rules: List[FactRule],
        raw_stats: Dict[str, Any],
        view_key: ViewKey,
    ) -> List[AnalysisFact]:
        run_id = raw_stats.get("run_id")
        view_type = view_key[-1]
        time_granularity = _to_float_or_none(raw_stats.get("time_granularity"))
        time_resolution = _to_float_or_none(raw_stats.get("time_resolution"))
        facts: List[AnalysisFact] = []

        for rule in rules:
            projected_columns = [column for column in rule.projected_columns() if column in rule_df.columns]
            missing_required = [column for column in rule.required_metrics if column not in projected_columns]
            if missing_required:
                logger.debug("facts.rule.skip_missing_required", rule_id=rule.id, missing=missing_required)
                continue

            working_df = rule_df.loc[:, projected_columns].copy()

            for metric_name, expression in rule.derived_metrics.items():
                working_df[metric_name] = _eval_expr(working_df, expression)

            unresolved_identifiers = [
                name
                for name in rule.referenced_columns
                if name not in working_df.columns and name not in RESERVED_IDENTIFIERS
            ]
            if unresolved_identifiers:
                logger.debug(
                    "facts.rule.skip_unresolved_identifiers",
                    rule_id=rule.id,
                    unresolved=sorted(unresolved_identifiers),
                )
                continue

            condition_series = _eval_expr(working_df, rule.when)
            condition_mask = condition_series.fillna(False).astype(bool)
            if not condition_mask.any():
                continue

            severity_series = _eval_expr(working_df, rule.severity_score).clip(lower=0.0, upper=1.0)
            confidence_series = None
            if rule.confidence:
                confidence_series = _eval_expr(working_df, rule.confidence).clip(lower=0.0, upper=1.0)

            matched_indices = list(working_df.index[condition_mask])
            if rule.emit_mode == "window" and matched_indices:
                # Emit a single finding for the whole window/view using the first matched row as anchor.
                matched_indices = [matched_indices[0]]

            for row_index in matched_indices:
                row = working_df.loc[row_index]
                severity_score = _to_float_or_none(_to_scalar(severity_series.loc[row_index]))
                severity_score = 0.0 if severity_score is None else severity_score
                confidence = 1.0
                if confidence_series is not None:
                    confidence = _to_float_or_none(_to_scalar(confidence_series.loc[row_index]))
                evidence_metrics: Dict[str, Any] = {}
                for metric in rule.required_metrics:
                    evidence_metrics[metric] = _to_scalar(row.get(metric))
                for metric_name in rule.derived_metrics:
                    evidence_metrics[metric_name] = _to_scalar(row.get(metric_name))

                epoch = None
                step = None
                t0_ns = None
                t1_ns = None
                if view_type == "epoch":
                    epoch = _to_int_or_none(row_index)
                elif "epoch" in row.index and row.get("epoch") is not None and not pd.isna(row.get("epoch")):
                    epoch = int(row.get("epoch"))
                if view_type == "step":
                    step = _to_int_or_none(row_index)
                elif "step" in row.index and row.get("step") is not None and not pd.isna(row.get("step")):
                    step = _to_int_or_none(row.get("step"))
                if view_type == "time_range":
                    time_bucket = _to_int_or_none(row_index)
                    if time_bucket is None and "time_range" in row.index:
                        time_bucket = _to_int_or_none(row.get("time_range"))
                    if (
                        time_bucket is not None
                        and time_granularity is not None
                        and time_resolution is not None
                    ):
                        interval_us = time_granularity * time_resolution
                        if interval_us > 0:
                            t0_ns = int(time_bucket * interval_us * 1_000)
                            t1_ns = int((time_bucket + 1) * interval_us * 1_000)

                fact = AnalysisFact(
                    fact_type=rule.fact_type,
                    window=FactWindow(
                        run_id=run_id,
                        view_type=view_type,
                        epoch=epoch,
                        step=step,
                        t0_ns=t0_ns,
                        t1_ns=t1_ns,
                        trigger="rule_eval",
                    ),
                    scope=FactScope(entity="window" if rule.emit_mode == "window" else str(row_index), rank_set="all"),
                    evidence={"metrics": evidence_metrics},
                    severity=FactSeverity(score=severity_score, label=_severity_label(severity_score)),
                    confidence=confidence,
                    opportunity_tags=rule.opportunity_tags.copy(),
                    provenance=FactProvenance(
                        rule_id=rule.id,
                        rule_version=rule.rule_version,
                        view_key=list(view_key),
                    ),
                )
                fact.finalize_id()
                facts.append(fact)
        return facts

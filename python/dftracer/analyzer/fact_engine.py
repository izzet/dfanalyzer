import dataclasses as dc
import importlib.util
import math
import os
import re
from pathlib import Path
import pandas as pd
import structlog
from typing import Any, Dict, List, Mapping, Optional
from omegaconf import OmegaConf

from .liveflow_v1 import STEP_EVIDENCE_FIELDS
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
    "fillna0",
    "math",
}

COHERENT_WINDOW_METRICS = (
    "domain_coherent", "domain_procs_active", "window_wall_time_proc_crit",
    "domain_input_time_proc_crit", "domain_decode_time_proc_crit",
    "domain_comm_time_proc_crit", "domain_compute_time_proc_crit",
    "residual_time_proc_crit",
)

# The online control window is the authority for safety and domain context.  A
# worker-only proc_name rule may receive only this fixed vector; arbitrary
# analyzer context must never become rule input.
CONTROL_WINDOW_METRICS = COHERENT_WINDOW_METRICS + (
    "epoch_nunique", "step_nunique", "step_min", "step_max",
)

# Optional, analyzer-owned: the measured cross-rank input-skew attribution (SKEW-ATTRIBUTION-2026-09-25.md)
# and the typical rank's vector.  Injected into a rank-thread rule only when the window frame carries them
# and they are finite; a rule that requires one fails closed when it is absent.
SKEW_WINDOW_METRICS = (
    "domain_skew_coverage", "domain_input_skew_attributed_time", "domain_input_skew_excess_time",
    "domain_comm_barrier_wait_time", "domain_skew_late_proc_input_time",
    "domain_skew_late_proc_comm_time", "domain_comm_residual_after_skew_time",
    "domain_input_time_proc_typ", "domain_decode_time_proc_typ", "domain_comm_time_proc_typ",
    "domain_compute_time_proc_typ", "window_wall_time_proc_typ",
)

# Window-level exposure metrics the PLANNER reads (clbo_planner.domains_from_metrics): the step-
# granular input sum-of-max (per node and job), and the exposed/total comm window medians.  The
# analyzer computes them on the ("window",) view (metrics.attach_input_step_max_sum_metrics,
# attach_input_job_step_max_sum_metrics, attach_comm_exposed_median_metrics,
# attach_comm_total_median_metrics) but a finding's key_metrics are exactly the fact's evidence
# metrics, which held only the rule's own required/derived metrics -- so on every real run the
# planner logged input_job_sum_of_max=null, comm_*_median=null and silently fell back to the
# crit-reference tier these metrics exist to replace (v4.5.10.1 1N run, 2026-09-28).  Forwarded
# ONLY when present and finite; a rule never requires them, so no rule's fail-closed behaviour
# changes and no threshold is involved.
LOOP_EXPOSURE_WINDOW_METRICS = (
    "domain_input_step_max_sum_time", "domain_input_step_nunique",
    "domain_input_job_step_max_sum_time",
    "domain_comm_pure_exposure_time_v2", "domain_comm_exposure_step_nunique",
    "domain_comm_total_median_time_v3", "domain_comm_total_step_nunique",
)

# LiveFlow-v1 job-level layer provenance (liveflow_v1.JOB_VALUE_KEYS): per-step seconds, forwarded when present.
LF_V1_JOB_METRICS = (
    "lf_v1_job_available", "lf_v1_job_level", "lf_v1_job_critical_input", "lf_v1_job_own_input",
    "lf_v1_job_peer_skew", "lf_v1_job_comm_wait_before", "lf_v1_job_attributed",
)

# Per-stage service of the critical rank, seconds per step (liveflow_v1.SERVICE_VALUE_KEYS): the Little's-law sizing input.
LF_V1_SERVICE_METRICS = tuple(k for f in ("storage", "decode", "delivery")
                              for k in ("lf_v1_svc_" + f, "lf_v1_svc_" + f + "_critical"))

COMPLETED_WINDOW_PROVENANCE_METRICS = (
    "stall_storage_rank_coverage", "stall_storage_rank_expected_count",
    "stall_storage_rank_observed_count", "stall_storage_raw_event_count",
)

COMPLETED_WINDOW_WORKER_ROSTER_PROVENANCE_METRICS = (
    "worker_roster_coverage", "worker_roster_expected_count",
    "worker_roster_observed_count",
)

_RANK_THREAD_NAME = re.compile(r"^app#([^#]+)#([0-9]+)#([0-9]+)$")
_RANK_THREAD_FAMILY_METRICS = (
    "reader_posix_read_count_frac_total",
    "reader_posix_read_time_proc_frac_total",
    "reader_posix_read_ops_slope",
)


def _rank_thread_family_metrics(rule) -> tuple:
    """The canonical WiSIO count/time/slope triple of the rule's operation family.

    The family is named by the rule's reducer metric (`<family>_ops_slope`); a rule without one
    keeps the original reader_posix_read family.  Families are never mixed within one rule.
    """
    metric = getattr(rule, "window_reducer_metric", None) or ""
    if metric.endswith("_ops_slope"):
        family = metric[: -len("_ops_slope")]
        return ("%s_count_frac_total" % family, "%s_time_proc_frac_total" % family, metric)
    return _RANK_THREAD_FAMILY_METRICS


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


def _fillna0(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value.fillna(0.0)
    if value is None:
        return 0.0
    try:
        return 0.0 if pd.isna(value) else value
    except Exception:
        return value


def _to_float_series(x):
    """Convert to float64 Series/array, replacing pd.NA with np.nan."""
    if isinstance(x, pd.Series):
        return x.astype("Float64").to_numpy(dtype="float64", na_value=float("nan"))
    return x


def _na_safe_max(*args):
    """NA-safe max that works on pandas Series with nullable dtypes."""
    import numpy as np

    result = args[0]
    for arg in args[1:]:
        if isinstance(result, pd.Series) or isinstance(arg, pd.Series):
            idx = result.index if isinstance(result, pd.Series) else arg.index
            result = pd.Series(
                np.fmax(_to_float_series(result), _to_float_series(arg)),
                index=idx,
            )
        else:
            result = max(result, arg)
    return result


def _na_safe_min(*args):
    """NA-safe min that works on pandas Series with nullable dtypes."""
    import numpy as np

    result = args[0]
    for arg in args[1:]:
        if isinstance(result, pd.Series) or isinstance(arg, pd.Series):
            idx = result.index if isinstance(result, pd.Series) else arg.index
            result = pd.Series(
                np.fmin(_to_float_series(result), _to_float_series(arg)),
                index=idx,
            )
        else:
            result = min(result, arg)
    return result


def _eval_python_expr(df: pd.DataFrame, expression: str) -> pd.Series:
    local_env: Dict[str, Any] = {column: df[column] for column in df.columns}
    local_env.update(
        {
            "max": _na_safe_max,
            "min": _na_safe_min,
            "abs": abs,
            "clip01": _clip01,
            "fillna0": _fillna0,
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

    def rebind_source_view(self, *, from_view: str, to_view: str) -> int:
        """Explicitly retarget selected rules without changing their semantics.

        This exists for the native control-window migration only.  Callers must
        opt in; the historical rule file and its default epoch dispatch remain
        untouched.  Frozen rules are replaced so dispatch and provenance stay
        internally consistent.
        """
        rebound = [
            dc.replace(rule, source_view=to_view) if rule.source_view == from_view else rule
            for rule in self.rules
        ]
        count = sum(before is not after for before, after in zip(self.rules, rebound))
        if count:
            self.rules = rebound
            self.rules_by_view = {}
            for rule in self.rules:
                self.rules_by_view.setdefault(rule.source_view, []).append(rule)
            logger.info("facts.rules.source_view_rebound", from_view=from_view, to_view=to_view, count=count)
        return count

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
        engine = cls(compiled_rules)
        if Path(resolved_rule_file).stem == "liveflow-v1":
            from .liveflow_v1 import declaration
            engine.liveflow_declaration = declaration()
        return engine

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
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[ViewKey, List[AnalysisFact]]:
        # Always-on (no --debug dependency): the 2026-09-27 v4.5.6 short rerun showed
        # analysis_facts=0 for all 40 windows with zero facts.rule.* lines even with DEBUG
        # confirmed active process-wide (patch 12) -- not even the unconditional first-action
        # facts.rule.skip inside _evaluate_view_rules fired once. FactEngine construction was
        # independently verified correct (2 rules, rules_by_view={'proc_name': [...]}), so the
        # remaining open question is whether the runtime flat_views dict handed to evaluate()
        # ever actually contains a ("proc_name",) key for real windows. This line answers that
        # directly and cheaply, every call, regardless of self.rules or view-key match outcome.
        logger.info(
            "facts.evaluate.view_keys",
            view_keys=list(flat_views.keys()),
            view_row_counts={str(k): len(v) for k, v in flat_views.items()},
            rules_by_view_keys=list(self.rules_by_view.keys()),
        )
        if hasattr(self, "liveflow_declaration"):
            from .liveflow_v1 import prepare_views
            flat_views = prepare_views(flat_views, self.liveflow_declaration, context)
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
                context=context or {},
            )
            if facts:
                all_facts[view_key] = facts
        if all_facts:
            logger.info(
                "facts.evaluate.done",
                total_facts=sum(len(v) for v in all_facts.values()),
                view_keys=list(all_facts.keys()),
            )
        return all_facts

    def _evaluate_view_rules(
        self,
        rule_df: pd.DataFrame,
        rules: List[FactRule],
        raw_stats: Dict[str, Any],
        view_key: ViewKey,
        context: Dict[str, Any],
    ) -> List[AnalysisFact]:
        run_id = raw_stats.get("run_id")
        view_type = view_key[-1]
        time_granularity = _to_float_or_none(raw_stats.get("time_granularity"))
        time_resolution = _to_float_or_none(raw_stats.get("time_resolution"))
        facts: List[AnalysisFact] = []

        for rule in rules:
            projected_columns = [column for column in rule.projected_columns() if column in rule_df.columns]
            if hasattr(self, "liveflow_declaration"):
                optional = ("windows_remaining", "epochs_remaining", "remaining_windows",
                            "remaining_epochs", "remaining_steps", "configured_total_steps",
                            "completed_steps", "configured_window_steps", "planner_history_ready",
                            "lf_v1_per_step", "reader_posix_read_ops_slope",
                            "reader_decode_ops_slope", "comm_ops_slope",
                            "step_wall_p90_us", *STEP_EVIDENCE_FIELDS, *LF_V1_JOB_METRICS, *LF_V1_SERVICE_METRICS)
                # Applied-vector snapshot: `physical_<knob>` for each knob the workload declared.
                optional += tuple(c for c in rule_df.columns
                                  if isinstance(c, str) and c.startswith("physical_"))
                projected_columns += [name for name in optional
                                      if name in rule_df.columns and name not in projected_columns]
            working_df = rule_df.loc[:, projected_columns].copy()

            rank_thread_provenance = {}
            if rule.worker_only or rule.rank_thread_only:
                # A proc_name view is diagnostic by default.  Online
                # worker-membership comes only from completed-window PID
                # provenance, never from parsing a display name.
                if view_type != "proc_name":
                    logger.debug("facts.rule.skip", rule_id=rule.id,
                                 reason="worker_only_non_proc_name_view")
                    continue
                row_pid_by_index = context.get("row_pid_by_index", {})
                if not row_pid_by_index:
                    logger.debug("facts.rule.skip", rule_id=rule.id,
                                 reason="selector_provenance_missing")
                    continue
                eligible = []
                if rule.worker_only:
                    worker_pids = {int(pid) for pid in context.get("worker_pids", ())}
                    if not worker_pids:
                        logger.debug("facts.rule.skip", rule_id=rule.id,
                                     reason="worker_provenance_missing")
                        continue
                    for index in working_df.index:
                        try:
                            eligible.append(int(row_pid_by_index.get(index)) in worker_pids)
                        except (TypeError, ValueError):
                            eligible.append(False)
                else:
                    native_host = context.get("native_host")
                    try:
                        rank_pids = {int(pid) for pid in context.get("rank_pids", ())}
                    except (TypeError, ValueError):
                        rank_pids = set()
                    if not isinstance(native_host, str) or not native_host or not rank_pids:
                        logger.debug("facts.rule.skip", rule_id=rule.id,
                                     reason="rank_thread_provenance_missing")
                        continue
                    for index in working_df.index:
                        match = _RANK_THREAD_NAME.match(str(index))
                        if match is None or match.group(1) != native_host:
                            eligible.append(False)
                            continue
                        parsed_pid, thread_id = int(match.group(2)), int(match.group(3))
                        try:
                            mapped_pid = int(row_pid_by_index.get(index))
                        except (TypeError, ValueError):
                            eligible.append(False)
                            continue
                        ok = parsed_pid == mapped_pid and parsed_pid in rank_pids
                        eligible.append(ok)
                        if ok:
                            rank_thread_provenance[index] = {
                                "selected_proc_name": str(index),
                                "selected_rank_pid": parsed_pid,
                                "selected_thread_id": thread_id,
                                "selected_mapped_pid": mapped_pid,
                            }
                working_df = working_df.loc[eligible]
                if working_df.empty:
                    logger.debug("facts.rule.skip", rule_id=rule.id,
                                 reason="no_selector_eligible_rows")
                    continue
                if rule.rank_thread_only:
                    family_metrics = _rank_thread_family_metrics(rule)
                    missing_family = [name for name in family_metrics
                                      if name not in working_df.columns]
                    if missing_family:
                        logger.debug("facts.rule.skip", rule_id=rule.id,
                                     reason="rank_thread_family_metrics_missing",
                                     missing=missing_family)
                        continue
                    numeric = working_df.loc[:, list(family_metrics)].apply(
                        pd.to_numeric, errors="coerce")
                    finite_positive = numeric.map(
                        lambda value: pd.notna(value) and math.isfinite(float(value)) and float(value) > 0.0)
                    working_df = working_df.loc[finite_positive.all(axis=1)]
                    if working_df.empty:
                        logger.debug("facts.rule.skip", rule_id=rule.id,
                                     reason="rank_thread_family_metrics_invalid")
                        continue
                # The worker row supplies only operation-family evidence.  The
                # coherent control vector is the one-row completed-window
                # aggregate, supplied by the analyzer context rather than
                # selected from an arbitrary worker row.
                native_window = _to_int_or_none(context.get("native_window"))
                window_metrics = context.get("window_metrics", {})
                if native_window is None or not isinstance(window_metrics, Mapping):
                    logger.debug("facts.rule.skip", rule_id=rule.id,
                                 reason="window_control_vector_missing")
                    continue
                try:
                    control = {
                        metric: float(window_metrics[metric])
                        for metric in CONTROL_WINDOW_METRICS
                    }
                except (KeyError, TypeError, ValueError):
                    logger.debug("facts.rule.skip", rule_id=rule.id,
                                 reason="window_control_vector_invalid")
                    continue
                if (not all(math.isfinite(value) for value in control.values())
                        or control["domain_coherent"] != 1.0
                        or control["domain_procs_active"] <= 0
                        or control["window_wall_time_proc_crit"] <= 0
                        or any(control[name] < 0 for name in (
                            "domain_input_time_proc_crit", "domain_decode_time_proc_crit",
                            "domain_comm_time_proc_crit", "domain_compute_time_proc_crit",
                        ))
                        or control["epoch_nunique"] != 1.0
                        or control["step_nunique"] < 1.0
                        or control["step_min"] > control["step_max"]):
                    logger.debug("facts.rule.skip", rule_id=rule.id,
                                 reason="window_control_vector_invalid")
                    continue
                # Merge after PID membership filtering and before validation.
                # The proc_name row remains the sole source of operation-family
                # slope; this vector supplies only the completed-window gate.
                # The native identity is also analyzer-owned context.  It is
                # deliberately overwritten rather than accepted from a row so
                # a test fixture may select a native window without allowing a
                # proc-name artifact to forge its control provenance.
                for metric, value in control.items():
                    working_df[metric] = value
                if rule.rank_thread_only:
                    for metric in SKEW_WINDOW_METRICS + LOOP_EXPOSURE_WINDOW_METRICS:
                        try:
                            value = float(window_metrics[metric])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if math.isfinite(value):
                            working_df[metric] = value
                # A proc_name row may contain an identically named column.
                # Never let it stand in for analyzer-owned raw-event evidence.
                working_df.drop(columns=list(COMPLETED_WINDOW_PROVENANCE_METRICS),
                                errors="ignore", inplace=True)
                working_df.drop(columns=list(COMPLETED_WINDOW_WORKER_ROSTER_PROVENANCE_METRICS),
                                errors="ignore", inplace=True)
                # Optional raw-event provenance is supplied exclusively by
                # Analyzer's completed-window context. Future rules that
                # require it fail closed if this context is unavailable;
                # existing rules do not consume it.
                provenance = context.get("completed_window_provenance", {})
                if isinstance(provenance, Mapping):
                    for metric in COMPLETED_WINDOW_PROVENANCE_METRICS:
                        try:
                            value = float(provenance[metric])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if math.isfinite(value):
                            working_df[metric] = value
                # The worker roster is an opt-in completed-window identity
                # proof. It is never inferred from a proc_name row and is
                # intentionally unavailable to rank-thread or default rules.
                if rule.worker_only:
                    roster_provenance = context.get("completed_window_worker_roster_provenance", {})
                    if isinstance(roster_provenance, Mapping):
                        for metric in COMPLETED_WINDOW_WORKER_ROSTER_PROVENANCE_METRICS:
                            try:
                                value = float(roster_provenance[metric])
                            except (KeyError, TypeError, ValueError):
                                continue
                            if math.isfinite(value):
                                working_df[metric] = value
                working_df["__liveflow_window_index"] = float(native_window)

            missing_required = [column for column in rule.required_metrics if column not in working_df.columns]
            if missing_required:
                logger.debug("facts.rule.skip", rule_id=rule.id, reason="missing_metrics", missing=missing_required)
                continue

            try:
                for metric_name, expression in rule.derived_metrics.items():
                    working_df[metric_name] = _eval_expr(working_df, expression)
            except Exception:
                logger.warning("facts.rule.derived_metric_error", rule_id=rule.id, exc_info=True)
                continue

            # Log derived metric values for debugging rule evaluation.
            #
            # REQUIRED metrics are logged alongside the derived ones. Without them a window's
            # evaluation is unfalsifiable from the logs: you can see that a rule fired but not on
            # what. That gap hid a frozen objective -- ten consecutive windows reporting
            # byte-identical fetch/compute times while carrying between 16 and 156,492 events --
            # which is invisible in `when_result` and fatal to any planner that learns.
            if rule.derived_metrics or rule.required_metrics:
                try:
                    derived_sample = {
                        m: _to_scalar(working_df[m].iloc[0]) if m in working_df.columns else None
                        for m in list(rule.derived_metrics or ()) + list(rule.required_metrics or ())
                    }
                except Exception:
                    derived_sample = {}
                logger.debug(
                    "facts.rule.evaluate",
                    rule_id=rule.id,
                    derived_metrics=derived_sample,
                    when_result="pending",
                )

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
                try:
                    derived_vals = {
                        m: _to_scalar(working_df[m].iloc[0]) if m in working_df.columns else None
                        for m in rule.derived_metrics
                    }
                except Exception:
                    derived_vals = {}
                logger.debug(
                    "facts.rule.evaluate",
                    rule_id=rule.id,
                    derived_metrics=derived_vals,
                    when_result=False,
                    when=rule.when,
                    rows=len(working_df),
                )
                continue

            severity_series = _eval_expr(working_df, rule.severity_score).clip(lower=0.0, upper=1.0)
            confidence_series = None
            if rule.confidence:
                confidence_series = _eval_expr(working_df, rule.confidence).clip(lower=0.0, upper=1.0)

            matched_indices = list(working_df.index[condition_mask])
            if rule.emit_mode == "window" and matched_indices:
                if rule.window_reducer == "first":
                    matched_indices = [matched_indices[0]]
                else:
                    metric = rule.window_reducer_metric
                    assert metric is not None
                    values = pd.to_numeric(working_df.loc[matched_indices, metric], errors="coerce").dropna()
                    if values.empty:
                        logger.debug("facts.rule.skip", rule_id=rule.id,
                                     reason="window_reducer_metric_missing", metric=metric)
                        continue
                    selected = values.idxmin() if rule.window_reducer == "min" else values.idxmax()
                    matched_indices = [selected]

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
                if hasattr(self, "liveflow_declaration"):
                    for name in ("windows_remaining", "epochs_remaining", "remaining_windows",
                                 "remaining_epochs", "remaining_steps", "configured_total_steps",
                                 "completed_steps", "configured_window_steps", "planner_history_ready",
                                 "lf_v1_per_step", "reader_posix_read_ops_slope",
                                 "reader_decode_ops_slope", "comm_ops_slope", "step_wall_p90_us",
                                 *STEP_EVIDENCE_FIELDS, *LF_V1_JOB_METRICS, *LF_V1_SERVICE_METRICS,
                                 *[c for c in row.index if isinstance(c, str) and c.startswith("physical_")]):
                        if name in row.index and name not in evidence_metrics:
                            value = _to_scalar(row.get(name))
                            if value is not None:
                                evidence_metrics[name] = value
                for metric_name in LOOP_EXPOSURE_WINDOW_METRICS:
                    if metric_name in evidence_metrics or metric_name not in row.index:
                        continue
                    forwarded = _to_scalar(row.get(metric_name))
                    if forwarded is not None:
                        evidence_metrics[metric_name] = forwarded
                if rule.worker_only:
                    evidence_metrics["selected_worker_pid"] = context["row_pid_by_index"].get(row_index)
                elif rule.rank_thread_only:
                    evidence_metrics.update(rank_thread_provenance[row_index])

                window = None
                epoch = None
                step = None
                t0_ns = None
                t1_ns = None
                if view_type == "window":
                    window = _to_int_or_none(row_index)
                elif "window" in row.index and row.get("window") is not None and not pd.isna(row.get("window")):
                    window = _to_int_or_none(row.get("window"))
                if view_type == "epoch":
                    epoch = _to_int_or_none(row_index)
                elif "epoch" in row.index and row.get("epoch") is not None and not pd.isna(row.get("epoch")):
                    epoch = int(row.get("epoch"))
                if view_type == "step":
                    step = _to_int_or_none(row_index)
                elif "step" in row.index and row.get("step") is not None and not pd.isna(row.get("step")):
                    step = _to_int_or_none(row.get("step"))
                native_window = context.get("native_window")
                if native_window is not None:
                    window = _to_int_or_none(native_window)
                    if window is None:
                        logger.debug("facts.rule.skip", rule_id=rule.id,
                                     reason="invalid_native_window")
                        continue
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

                scope_entity = (
                    rule.scope_entity
                    if rule.scope_entity is not None
                    else ("window" if rule.emit_mode == "window" else str(row_index))
                )
                scope_node = None
                if view_type in {"source_node", "host_hash"}:
                    scope_node = str(row_index)
                    if rule.emit_mode != "window":
                        scope_entity = view_type

                fact = AnalysisFact(
                    fact_type=rule.fact_type,
                    window=FactWindow(
                        run_id=run_id,
                        view_type=view_type,
                        window=window,
                        epoch=epoch,
                        step=step,
                        t0_ns=t0_ns,
                        t1_ns=t1_ns,
                        trigger="rule_eval",
                    ),
                    scope=FactScope(
                        layer=rule.scope_layer,
                        entity=scope_entity,
                        rank_set="all",
                        node=scope_node,
                    ),
                    evidence={"metrics": evidence_metrics},
                    severity=FactSeverity(score=severity_score, label=_severity_label(severity_score)),
                    confidence=confidence,
                    opportunity_tags=rule.opportunity_tags.copy(),
                    suppresses_tags=rule.suppresses_tags.copy() if rule.suppresses_tags else [],
                    provenance=FactProvenance(
                        rule_id=rule.id,
                        rule_version=rule.rule_version,
                        view_key=list(view_key),
                    ),
                )
                fact.finalize_id()
                logger.debug(
                    "facts.rule.fired",
                    rule_id=rule.id,
                    fact_type=fact.fact_type,
                    severity_score=severity_score,
                    opportunity_tags=fact.opportunity_tags,
                    key_metrics=evidence_metrics,
                )
                facts.append(fact)
        return facts

"""A threshold-free-signal fast path for the live per-window analysis.

WHY THIS EXISTS
---------------
At 32 nodes with 8 ranks per node and a two-step window, the full analysis path spent a measured
**8.7 s per 82 ms window** (`T3-0-SHM-LOOP-CORONA.md` section 4j), i.e. 106x slower than real time.
The agent therefore saw one window in 350, the ring lanes overflowed behind it (37.6% of trace and
98.6% of control records dropped), and the windows it did analyse were assembled from partial
records.  The run's own log attributes that cost, over n=4174 windows, as:

    Compute views                 mean 3.592 s   (67%)
    Process views                 mean 1.566 s   (29%)
    Compute high-level metrics    mean 0.190 s   ( 4%)

and the cost is **independent of the window's size**: a real window of this cell carries on the
order of 10^2 events, and the same 6 s is spent on 70 events as on 70,000.  It is fixed per-window
overhead -- one pandas `groupby().agg()` over a ~40-entry aggregation dict per layer, per view,
per window, across 16 layers, plus the derived-metric expansion, the cross-layer metric block and
the eager `memory_usage(deep=True)` inside the debug logging.

WHAT THE THRESHOLD-FREE RULE SETS ACTUALLY NEED
-----------------------------------------------
`configs/fact_rules/*-signal.yaml` emit ONE per-window finding carrying raw per-domain times and
let the planner decide.  Between them they read only:

    <layer>_time_proc_max      per-layer max over processes of the per-process time sum
    <layer>_time_sum           per-layer total time
    <layer>_count_sum          per-layer call count
    residual_time_proc_max     and the domain_* decomposition around it
    window_wall_time_proc_max  the objective

Every one of those is a sum or a max over the window's rows.  None of them needs the main view,
the per-layer view cross product, the quantile block, the call-statistics block, the cross-layer
exclusive-time block, the frac-of-boundary metrics or the flat-view merge.  This module computes
exactly that set directly from the traces frame with numpy, and hands the result to the SAME fact
engine, so the fact envelope the planner receives is byte-identical for the rules it can serve.

SAFETY: IT REFUSES RATHER THAN GUESSES
---------------------------------------
`FastPathPlan.build` returns None whenever a loaded rule references anything this module does not
produce -- a gated rule set, a frac-of-boundary metric, an exclusive `o_*` time, a quantile.  The
caller then runs the full path.  A rule set that the fast path cannot serve is therefore SLOW, never
WRONG, and the gated rule sets keep working unchanged as ablation controls.
"""

from __future__ import annotations

import hashlib
import json
import re
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import structlog

from .constants import (
    COL_CATEGORY,
    COL_COUNT,
    COL_FUNC_NAME,
    COL_IO_CAT,
    COL_PROC_NAME,
    COL_SIZE,
    COL_TIME,
)
from .metrics import (
    DOMAIN_VECTOR_COLUMNS,
    attach_window_integrity_metrics,
    attach_input_step_max_sum_metrics,
    attach_input_job_step_max_sum_metrics,
    attach_comm_exposed_median_metrics,
    attach_comm_total_median_metrics,
    coherent_domain_vectors,
    set_view_metrics,
    set_window_decomposition_metrics,
)

logger = structlog.get_logger()

# The layers whose per-(group, process) matrices the coherent decomposition needs. Everything else
# is still reduced straight to a sum and a per-layer max, so this costs one extra reshape per
# layer in this list per window and nothing else.
_DOMAIN_LAYERS = (
    "window_wall", "comm", "compute", "device_transfer", "stall_storage", "stall_decode",
    "fetch_iter", "reader_decode",
)

# Columns a layer or derived-metric condition may use and still be evaluated on the DEDUPLICATED
# key table rather than row by row.  These are the low-cardinality identity columns; a condition
# touching anything else (a size, an offset, a timestamp) falls back to a per-row evaluation,
# which is still correct, just not constant time.
_KEY_COLUMNS = (COL_CATEGORY, COL_FUNC_NAME, COL_IO_CAT)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


_STRING_LITERAL = re.compile(r"""(['"])(?:\\.|(?!\1).)*\1""")


def completed_window_stall_storage_provenance(
    traces: pd.DataFrame, expected_rank_pids: Iterable[int], *, pid_column: str = "pid",
) -> Dict[str, object]:
    """Raw-event coverage for a completed control window's rank-owned stalls.

    Event presence is intentionally independent of duration: an emitted
    zero-duration ``stall.storage`` record proves instrumentation coverage.
    Missing schema, no expected ranks, absent events, or a missing rank fail
    closed. This is completed-window context, never a flat-view metric.
    """
    try:
        expected = {int(pid) for pid in expected_rank_pids}
    except (TypeError, ValueError):
        expected = set()
    required = {COL_CATEGORY, COL_FUNC_NAME, pid_column}
    observed = set()
    event_count = 0
    if expected and required.issubset(traces.columns):
        mask = ((traces[COL_CATEGORY] == "stall")
                & (traces[COL_FUNC_NAME] == "stall.storage"))
        for raw_pid in traces.loc[mask, pid_column]:
            try:
                pid = int(raw_pid)
            except (TypeError, ValueError):
                continue
            if pid in expected:
                observed.add(pid)
                event_count += 1
    covered = bool(expected) and observed == expected
    return {
        "stall_storage_rank_coverage": float(covered),
        "stall_storage_rank_expected_count": float(len(expected)),
        "stall_storage_rank_observed_count": float(len(observed)),
        "stall_storage_raw_event_count": float(event_count),
        "stall_storage_expected_rank_pids": sorted(expected),
        "stall_storage_observed_rank_pids": sorted(observed),
    }


_DLIO_WORKER_ROSTER_SCHEMA = "dlio-worker-posix-roster.v1"
_STORMER_WORKER_ROSTER_SCHEMA = "stormer-worker-posix-roster.v1"
# Schema dispatch is by EXPLICIT allow-list.  The two producers write the same field set and the
# same `<schema>.<digest>.rank-R.worker-W.json` naming; anything else fails closed.
_WORKER_ROSTER_SCHEMAS = frozenset((_DLIO_WORKER_ROSTER_SCHEMA, _STORMER_WORKER_ROSTER_SCHEMA))
_DLIO_WORKER_ROSTER_FIELDS = frozenset((
    "schema_version", "run_prefix", "host", "rank", "worker_id", "pid",
    "expected_workers",
))


def completed_window_worker_roster_provenance(
    traces: pd.DataFrame,
    row_pid_by_index: Dict[object, int],
    *,
    roster_dir: Optional[str],
    roster_prefix: Optional[str],
    expected_workers: Optional[int],
    expected_host: str,
    expected_ranks: int,
    expected_rank_pids: Iterable[int],
    pid_column: str = "pid",
    roster_schema: str = _DLIO_WORKER_ROSTER_SCHEMA,
) -> Dict[str, object]:
    """Fail-closed v1 worker roster proof (DLIO default, or StorMer) for one completed local window.

    `roster_schema` selects which producer's records are accepted; the default preserves the DLIO
    behaviour exactly.  An unknown schema, or records of a different schema, fail closed.

    The roster is an opt-in identity whitelist, not a replacement for control
    membership or rank-side raw stall coverage.  It is read only after the
    completed window frame is available, so a late filesystem record simply
    suppresses this window.  Every whitelisted PID must also own an
    unambiguous same-window reader-POSIX row.
    """
    result = {
        "worker_roster_coverage": 0.0,
        "worker_roster_expected_count": 0.0,
        "worker_roster_observed_count": 0.0,
        "worker_roster_pids": [],
    }
    if roster_schema not in _WORKER_ROSTER_SCHEMAS:
        return result
    if (not isinstance(roster_dir, str) or not roster_dir
            or not isinstance(roster_prefix, str) or not roster_prefix
            or type(expected_workers) is not int or expected_workers < 1
            or not isinstance(expected_host, str) or not expected_host
            or type(expected_ranks) is not int or expected_ranks < 1):
        return result
    directory = Path(roster_dir)
    if not directory.is_absolute() or not directory.is_dir():
        return result

    digest = hashlib.sha256(roster_prefix.encode("utf-8")).hexdigest()[:16]
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return result
        if not isinstance(record, dict) or set(record) != _DLIO_WORKER_ROSTER_FIELDS:
            return result
        valid = (
            record.get("schema_version") == roster_schema
            and record.get("run_prefix") == roster_prefix
            and isinstance(record.get("host"), str) and bool(record["host"])
            and type(record.get("rank")) is int and record["rank"] >= 0
            and type(record.get("worker_id")) is int
            and 0 <= record["worker_id"] < expected_workers
            and type(record.get("pid")) is int and record["pid"] > 0
            and record.get("expected_workers") == expected_workers
        )
        expected_name = (
            f"{roster_schema}.{digest}.rank-{record.get('rank')}"
            f".worker-{record.get('worker_id')}.json"
        )
        if not valid or path.name != expected_name:
            return result
        records.append(record)

    local = [record for record in records if record["host"] == expected_host]
    if len(local) != expected_ranks * expected_workers:
        return result
    rank_ids = {record["rank"] for record in local}
    if len(rank_ids) != expected_ranks:
        return result
    if any(
        {record["worker_id"] for record in local if record["rank"] == rank}
        != set(range(expected_workers))
        for rank in rank_ids
    ):
        return result
    roster_pids = {record["pid"] for record in local}
    if len(roster_pids) != len(local):
        return result

    result["worker_roster_expected_count"] = float(len(roster_pids))
    if rank_ids != set(range(expected_ranks)):
        return result

    def _exact_pid(value: object) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            try:
                integer = int(value)
            except (OverflowError, ValueError):
                return None
            return integer if value == integer else None
        return None

    rank_pids = {_exact_pid(pid) for pid in expected_rank_pids}
    rank_pids.discard(None)
    if len(rank_pids) != expected_ranks or roster_pids & rank_pids:
        return result
    required = {COL_CATEGORY, COL_PROC_NAME, pid_column}
    if not required.issubset(traces.columns):
        return result
    categories = traces[COL_CATEGORY].astype("string")
    reader_posix = (
        categories.str.contains("posix", case=False, regex=False, na=False)
        & categories.str.contains("_reader", case=False, regex=False, na=False)
    )
    observed = set()
    for proc_name, raw_pid in traces.loc[reader_posix, [COL_PROC_NAME, pid_column]].dropna().itertuples(
        index=False, name=None
    ):
        pid = _exact_pid(raw_pid)
        if pid is None:
            continue
        if pid in roster_pids and row_pid_by_index.get(proc_name) == pid:
            observed.add(pid)
    result["worker_roster_observed_count"] = float(len(observed))
    result["worker_roster_pids"] = sorted(roster_pids)
    result["worker_roster_coverage"] = float(observed == roster_pids)
    return result


def _condition_columns(condition: str, available: Iterable[str]) -> Set[str]:
    """Columns a condition reads.

    String literals are stripped first. Without that, `func_name.str.startswith("epoch")` reads as
    touching the `epoch` column, which is not a key column, which disabled the mask cache for
    EVERY term and left the fast path paying 190 ms a window in pandas expression parsing.
    """
    available = set(available)
    stripped = _STRING_LITERAL.sub(" ", condition or "")
    return {name for name in _IDENTIFIER.findall(stripped) if name in available}


def _eval_mask(frame: pd.DataFrame, condition: str, term: str) -> np.ndarray:
    """Boolean mask for one condition, falling back to `query` for expressions `eval` refuses."""
    try:
        result = frame.eval(condition)
    except Exception:
        try:
            result = frame.index.isin(frame.query(condition).index)
        except Exception:
            logger.warning("analyzer.fastpath.condition_failed", term=term, condition=condition)
            return np.zeros(len(frame), dtype=bool)
    return np.asarray(pd.Series(result).fillna(False).astype(bool))


def preserve_measured_stall_zeros(view: pd.DataFrame) -> pd.DataFrame:
    """Publish exact-zero stall times only in groups with observed stall records.

    Legacy aggregation encodes zero time as null; positive family event counts
    retain its presence information. Do not fill absent families or ratios.
    This follows liveflow_v1._measured's existing zero/presence contract.
    """
    for family in ("stall_storage", "stall_decode"):
        count_col = family + "_count_sum"
        if count_col not in view:
            continue
        counts = pd.to_numeric(view[count_col], errors="coerce")
        observed = counts.notna() & np.isfinite(counts) & (counts > 0)
        for suffix in ("time_sum", "time_proc_max", "time_proc_mean", "time_proc_min"):
            col = family + "_" + suffix
            if col in view:
                view[col] = view[col].mask(observed & view[col].isna(), 0.0)
    return view


class FastPathPlan:
    """The set of per-layer sums a rule file needs, resolved once against a preset.

    Built once per analyzer (the preset and the rule file do not change mid-run) and reused for
    every window, so the rule inspection and the condition parsing are not per-window costs.
    """

    def __init__(self, preset, view_types: List[str], layers: List[str]):
        self.preset = preset
        self.view_types = list(view_types)
        self.layers = list(layers)
        self.time_boundary_layer = getattr(preset, "time_boundary_layer", None)

        size_layers = {name.lower() for name in (preset.size_layers or [])}
        derived_metrics = preset.derived_metrics or {}
        size_derived = preset.size_derived_metrics or {}

        # (column_prefix, condition, wants_size).  The prefix is the flat-view column prefix, so
        # `reader_posix` -> reader_posix_time_proc_max and `posix` + derived `read` ->
        # posix_read_time_proc_max, exactly as the full path names them.
        self.terms: List[Tuple[str, str, bool]] = []
        for layer, layer_condition in preset.layer_defs.items():
            key = layer.lower()
            wants_size = key in size_layers
            self.terms.append((key, layer_condition or "", wants_size))
            layer_size_derived = set(size_derived.get(key, []) or [])
            for metric, metric_condition in (derived_metrics.get(key, {}) or {}).items():
                combined = metric_condition or ""
                if layer_condition:
                    combined = "(%s) and (%s)" % (layer_condition, combined) if combined else layer_condition
                self.terms.append(
                    ("%s_%s" % (key, metric), combined, wants_size and metric in layer_size_derived)
                )

        self.produced_columns: Set[str] = set()
        # These are materialized only by ``set_view_metrics`` on the proc_name
        # view.  Keep them distinct from universally available aggregates so
        # a window/epoch rule cannot be accepted merely because a sibling
        # proc_name view happens to be requested.
        self.proc_name_only_columns: Set[str] = set()
        for prefix, _condition, wants_size in self.terms:
            self.produced_columns.add("%s_time_sum" % prefix)
            self.produced_columns.add("%s_time_proc_max" % prefix)
            self.produced_columns.add("%s_count_sum" % prefix)
            # ``set_view_metrics`` below emits these canonical WiSIO shares
            # for the proc_name view before it emits ``*_ops_slope``.  They
            # are first-class rule inputs, not optional diagnostics: omitting
            # them here incorrectly rejects an otherwise exact fast-path plan
            # and silently falls back to the legacy aggregate-only path,
            # whose control vector is deliberately incoherent.
            self.proc_name_only_columns.add("%s_count_frac_total" % prefix)
            self.proc_name_only_columns.add("%s_time_proc_frac_total" % prefix)
            # `set_view_metrics()` below produces the canonical WiSIO slope
            # for a process view from precisely these count/time sums.
            self.proc_name_only_columns.add("%s_ops_slope" % prefix)
            if wants_size:
                self.produced_columns.add("%s_size_sum" % prefix)
        # set_window_decomposition_metrics adds these on top, for both time metrics.
        for metric in ("time_sum", "time_proc_max"):
            for name in ("domain_input", "domain_decode", "domain_comm", "domain_compute",
                         "domain_device", "residual"):
                self.produced_columns.add("%s_%s" % (name, metric))
        self.produced_columns.add("domain_exposure_present")
        self.produced_columns.update(DOMAIN_VECTOR_COLUMNS)
        # `attach_window_integrity_metrics()` adds these only to the native
        # window view.  Worker-only rules receive that one-row vector as a
        # fail-closed safety gate; advertise the existing fields so such a
        # rule is not needlessly demoted to the full online path.
        self.produced_columns.update({
            "epoch_nunique", "step_nunique", "step_min", "step_max",
        })
        self._mask_cache: Dict[tuple, Dict[str, np.ndarray]] = {}

    # -- gate ------------------------------------------------------------------------------
    @classmethod
    def build(cls, preset, view_types, layers, fact_engine) -> Optional["FastPathPlan"]:
        """Return a plan only when it can serve EVERY loaded rule, else None."""
        if fact_engine is None or not getattr(fact_engine, "rules", None):
            return None
        plan = cls(preset, view_types, layers)
        from .fact_engine import (
            COMPLETED_WINDOW_PROVENANCE_METRICS,
            COMPLETED_WINDOW_WORKER_ROSTER_PROVENANCE_METRICS,
            RESERVED_IDENTIFIERS,
        )

        for rule in fact_engine.rules:
            if rule.source_view not in plan.view_types and rule.source_view != COL_PROC_NAME:
                # The rule wants a view this analyzer is not asked to build; the full path would
                # not serve it either, but say so rather than assume.
                logger.info("analyzer.fastpath.declined", rule_id=rule.id,
                            reason="source_view_not_built", source_view=rule.source_view)
                return None
            available = set(plan.produced_columns)
            if rule.source_view == COL_PROC_NAME:
                available.update(plan.proc_name_only_columns)
                if rule.rank_thread_only:
                    from .fact_engine import SKEW_WINDOW_METRICS
                    available.update(SKEW_WINDOW_METRICS)
                if rule.worker_only or rule.rank_thread_only:
                    # These are never flat-view columns. FactEngine injects
                    # them only from completed-window analyzer context, so a
                    # missing raw-event coverage still fails closed at evaluation.
                    available.update(COMPLETED_WINDOW_PROVENANCE_METRICS)
                if rule.worker_only:
                    # The opt-in worker roster is never available to rank-thread rules.
                    available.update(COMPLETED_WINDOW_WORKER_ROSTER_PROVENANCE_METRICS)
            else:
                available.difference_update(plan.proc_name_only_columns)
            unknown = (
                set(rule.referenced_columns)
                - available
                - set(rule.derived_metrics or ())
                - set(RESERVED_IDENTIFIERS)
            )
            if unknown:
                logger.info("analyzer.fastpath.declined", rule_id=rule.id,
                            reason="unsupported_metrics", metrics=sorted(unknown))
                return None
        logger.info(
            "analyzer.fastpath.enabled",
            rules=[rule.id for rule in fact_engine.rules],
            terms=len(plan.terms),
            views=plan.view_types,
            fact_engine_id=id(fact_engine),
        )
        return plan

    # -- per-window computation ------------------------------------------------------------
    def _term_masks(self, traces: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Evaluate every layer / derived-metric condition once for this window.

        Two properties make this constant time rather than proportional to the window:

        1. Conditions only look at the identity columns, whose distinct combinations number in the
           tens however large the window is, so they are evaluated on the DEDUPLICATED key table
           and scattered back through the factorisation codes.
        2. Those distinct combinations barely change between consecutive windows, so the answers
           are cached against the key table. A pandas `eval` costs ~4 ms because it rebuilds the
           frame's column resolvers every call; 46 of them per window is 190 ms on its own, which
           would miss an 82 ms budget by itself. Cached, a window pays a dict lookup and a take.
        """
        available = set(traces.columns)
        key_cols = [col for col in _KEY_COLUMNS if col in available]
        codes, uniques = pd.MultiIndex.from_frame(traces[key_cols]).factorize()
        key_df = uniques.to_frame(index=False)
        key_df.columns = key_cols
        key_set = set(key_cols)
        n_rows = len(traces)

        cache_key = tuple(map(tuple, key_df.astype(str).itertuples(index=False, name=None)))
        cached = self._mask_cache.get(cache_key)
        if cached is None:
            cached = {}
            for prefix, condition, _wants_size in self.terms:
                if not condition:
                    cached[prefix] = np.ones(len(key_df), dtype=bool)
                    continue
                if not _condition_columns(condition, available).issubset(key_set):
                    continue          # needs a per-row column; evaluated below, never cached
                cached[prefix] = _eval_mask(key_df, condition, prefix)
            if len(self._mask_cache) >= 64:
                self._mask_cache.clear()
            self._mask_cache[cache_key] = cached

        masks: Dict[str, np.ndarray] = {}
        for prefix, condition, _wants_size in self.terms:
            values = cached.get(prefix)
            if values is not None:
                masks[prefix] = values[codes]
            else:
                masks[prefix] = _eval_mask(traces, condition, prefix)
        return masks

    def flat_views(
        self,
        traces: pd.DataFrame,
        control_pids: Optional[List[int]] = None,
    ) -> Dict[Tuple[str, ...], pd.DataFrame]:
        views: Dict[Tuple[str, ...], pd.DataFrame] = {}
        if traces is None or traces.empty:
            return views
        # Legacy flat aggregates retain the full-path semantics: pandas groupby drops a row whose
        # key is NA in ANY configured view.  Do not change those aggregates or their slopes here.
        #
        # The domain matrix has a stricter, separate association contract.  This method is called
        # once for a completed control window, so that window is authoritative for membership.  An
        # unkeyed row may join the *epoch domain matrix only* when all keyed rows in this completed
        # window name exactly one epoch.  It is never copied into normal aggregates and is never
        # guessed/broadcast over multiple epochs.  That admits the unannotated window.stop needed
        # to establish an active rank while preserving the fail-closed outcome for ambiguity.
        domain_traces = traces
        domain_unkeyed_rows_included = 0
        domain_epoch_association_unambiguous = 0
        if "epoch" in self.view_types and "epoch" in traces.columns:
            known_epochs = pd.unique(traces["epoch"].dropna())
            unkeyed_epoch = traces["epoch"].isna()
            if len(known_epochs) == 1 and unkeyed_epoch.any():
                domain_traces = traces.copy()
                domain_traces.loc[unkeyed_epoch, "epoch"] = known_epochs[0]
                domain_unkeyed_rows_included = int(unkeyed_epoch.sum())
                domain_epoch_association_unambiguous = 1
        keyed = np.ones(len(traces), dtype=bool)
        for key_view in self.view_types:
            if key_view in traces.columns:
                keyed &= ~pd.isna(traces[key_view]).to_numpy()
        if not keyed.all():
            traces = traces.loc[keyed]
            if traces.empty:
                return views

        masks = self._term_masks(traces)
        time_values = pd.to_numeric(traces[COL_TIME], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if COL_SIZE in traces.columns:
            size_values = pd.to_numeric(traces[COL_SIZE], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            size_values = np.zeros(len(traces), dtype=float)
        domain_masks = self._term_masks(domain_traces)
        proc_codes, proc_uniques = pd.factorize(traces[COL_PROC_NAME])
        n_proc = max(len(proc_uniques), 1)
        # THE COHERENT VECTORS DECOMPOSE A PROCESS, AND `proc_name` IS PER THREAD.
        # `_set_proc_names` builds `app#<host>#<pid>#<tid>`, so two threads of one rank are two
        # different "processes" here. That is right for the per-layer `time_proc_max` columns,
        # which have always meant per-thread and are left exactly as they were. It is WRONG for
        # the domain matrices, because a domain that a rank pays on a helper thread then lands on
        # a row whose `window_wall` is zero, and `coherent_domain_vectors` drops that row as a
        # process that did not run the window.
        #
        # Measured on `stgi_capacity_gate1` (StorMer, Corona, 1 node, 8 ranks), window 8 of 24:
        # every rank emits `window.stop`, `stall.storage` and `compute` on its main thread and
        # every one of its `comm.all_reduce` regions on a DDP helper thread, 28.6 s of a 34.8 s
        # window on rank 0. Grouped per thread the comm domain is zero in EVERY window of EVERY
        # StorMer run and that time is absorbed into compute, in both `crit` and `typ`. DLIO emits
        # its barrier on the main thread and is unaffected, which is why the DLIO half of
        # STATUS-ATTRIBUTION-FIX-2026-09-08 validated on this path while the StorMer half
        # validated through `make_domain_fixture.py`, which groups by PID.
        #
        # So the domain matrices are grouped by pid, which is what "per process" means and what
        # the fixture behind `test_domain_attribution.py` has always used. Falls back to
        # `proc_name` where there is no pid column, so nothing that lacks one changes behaviour.
        for view_type in self.view_types:
            if view_type not in traces.columns:
                continue
            group_codes, group_uniques = pd.factorize(traces[view_type], sort=True)
            if len(group_uniques) == 0:
                continue
            valid = group_codes >= 0
            n_groups = len(group_uniques)
            is_process_based = view_type == COL_PROC_NAME
            if is_process_based:
                pair_codes = group_codes
                n_pairs = n_groups
            else:
                pair_codes = group_codes * n_proc + proc_codes
                n_pairs = n_groups * n_proc
            # One preallocated float64 block for the whole view. Assembling the frame column by
            # column and converting at the end cost more than every bincount put together: a
            # 171-column `astype` is ~40 ms, which is half an 82 ms window budget on its own.
            names: List[str] = []
            blocks: List[np.ndarray] = []
            # (n_groups, n_proc) time matrices for the domain layers, kept so the decomposition can
            # be taken from ONE process rather than assembled out of per-layer maxima that come
            # from different ones. See metrics.coherent_domain_vectors.
            domain_mats: Dict[str, np.ndarray] = {}

            # Only the epoch domain matrix may use the completed window's associated unkeyed
            # rows. Other views continue to use their ordinary keyed source exactly as before.
            matrix_traces = domain_traces if view_type == "epoch" else traces
            matrix_masks = domain_masks if view_type == "epoch" else masks
            if "pid" in matrix_traces.columns:
                matrix_dproc_codes, matrix_dproc_uniques = pd.factorize(matrix_traces["pid"])
            else:
                matrix_dproc_codes, matrix_dproc_uniques = pd.factorize(matrix_traces[COL_PROC_NAME])
            matrix_n_dproc = max(len(matrix_dproc_uniques), 1)
            # Pandas stores categorical codes in the narrowest signed integer.
            # With 34 groups and four processes, the final pair identifier is
            # 33 * 4 = 132: valid in the output matrix but not representable in
            # int8.  Widen before arithmetic; the codes are indexes, not a
            # compact persisted data field.
            matrix_dproc_codes = np.asarray(matrix_dproc_codes, dtype=np.int64)
            matrix_group_codes = np.asarray(
                pd.Categorical(matrix_traces[view_type], categories=group_uniques).codes,
                dtype=np.int64,
            )
            matrix_valid = matrix_group_codes >= 0
            matrix_pair_codes = matrix_group_codes * matrix_n_dproc + matrix_dproc_codes
            matrix_n_pairs = n_groups * matrix_n_dproc
            matrix_time_values = pd.to_numeric(
                matrix_traces[COL_TIME], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)

            def _emit(name: str, values: np.ndarray) -> None:
                names.append(name)
                # The full path runs `.replace(0, pd.NA)` on the main view and on each view, so a
                # layer that contributed nothing reads as absent rather than as a measured zero.
                blocks.append(np.where(values == 0, np.nan, values))

            for prefix, _condition, wants_size in self.terms:
                mask = masks[prefix] & valid
                if prefix in _DOMAIN_LAYERS and not is_process_based:
                    matrix_mask = matrix_masks[prefix] & matrix_valid
                    if matrix_mask.any():
                        domain_mats[prefix] = np.bincount(
                            matrix_pair_codes[matrix_mask], weights=matrix_time_values[matrix_mask],
                            minlength=matrix_n_pairs,
                        ).reshape(n_groups, matrix_n_dproc)
                    else:
                        domain_mats[prefix] = np.zeros((n_groups, matrix_n_dproc))
                if not mask.any():
                    empty = np.zeros(n_groups)
                    _emit("%s_time_sum" % prefix, empty)
                    _emit("%s_time_proc_max" % prefix, empty)
                    _emit("%s_count_sum" % prefix, empty)
                    if wants_size:
                        _emit("%s_size_sum" % prefix, empty)
                    continue
                per_pair = np.bincount(pair_codes[mask], weights=time_values[mask], minlength=n_pairs)
                if is_process_based:
                    time_sum = per_pair
                    time_proc_max = per_pair
                else:
                    per_pair = per_pair.reshape(n_groups, n_proc)
                    time_sum = per_pair.sum(axis=1)
                    time_proc_max = per_pair.max(axis=1)
                _emit("%s_time_sum" % prefix, time_sum)
                _emit("%s_time_proc_max" % prefix, time_proc_max)
                _emit("%s_count_sum" % prefix,
                      np.bincount(group_codes[mask], minlength=n_groups).astype(float))
                if wants_size:
                    _emit("%s_size_sum" % prefix,
                          np.bincount(group_codes[mask], weights=size_values[mask], minlength=n_groups))

            index = pd.Index(group_uniques, name=view_type)
            view = pd.DataFrame(np.column_stack(blocks), index=index, columns=names)
            if getattr(getattr(self, "preset", None), "name", None) == "liveflow-v1":
                view = preserve_measured_stall_zeros(view)
            coherent = None
            if not is_process_based and "window_wall" in domain_mats:
                coherent = coherent_domain_vectors(domain_mats)
            view = set_window_decomposition_metrics(view, is_process_based, coherent=coherent)
            if is_process_based:
                # Reuse the analyzer's canonical normalized-count/normalized-
                # time slope implementation.  This is a small vectorized
                # operation over the already-reduced proc_name frame, not the
                # full view/layer expansion that this fast path avoids.
                view = set_view_metrics(
                    view, metric_boundaries={}, is_view_process_based=True
                )
            if view_type == "epoch" and not is_process_based:
                view["domain_epoch_association_unambiguous"] = domain_epoch_association_unambiguous
                view["domain_unkeyed_rows_included"] = domain_unkeyed_rows_included
            views[(view_type,)] = view.sort_index(axis=1)
        views = attach_window_integrity_metrics(views, traces, control_pids=control_pids)
        views = attach_input_step_max_sum_metrics(views, traces)
        views = attach_input_job_step_max_sum_metrics(views, traces)
        views = attach_comm_exposed_median_metrics(views, traces)
        return attach_comm_total_median_metrics(views, traces)

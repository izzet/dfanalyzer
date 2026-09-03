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

import re
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
from .metrics import set_window_decomposition_metrics

logger = structlog.get_logger()

# Columns a layer or derived-metric condition may use and still be evaluated on the DEDUPLICATED
# key table rather than row by row.  These are the low-cardinality identity columns; a condition
# touching anything else (a size, an offset, a timestamp) falls back to a per-row evaluation,
# which is still correct, just not constant time.
_KEY_COLUMNS = (COL_CATEGORY, COL_FUNC_NAME, COL_IO_CAT)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


_STRING_LITERAL = re.compile(r"""(['"])(?:\\.|(?!\1).)*\1""")


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
        for prefix, _condition, wants_size in self.terms:
            self.produced_columns.add("%s_time_sum" % prefix)
            self.produced_columns.add("%s_time_proc_max" % prefix)
            self.produced_columns.add("%s_count_sum" % prefix)
            if wants_size:
                self.produced_columns.add("%s_size_sum" % prefix)
        # set_window_decomposition_metrics adds these on top, for both time metrics.
        for metric in ("time_sum", "time_proc_max"):
            for name in ("domain_input", "domain_decode", "domain_comm", "domain_compute", "residual"):
                self.produced_columns.add("%s_%s" % (name, metric))
        self.produced_columns.add("domain_exposure_present")
        self._mask_cache: Dict[tuple, Dict[str, np.ndarray]] = {}

    # -- gate ------------------------------------------------------------------------------
    @classmethod
    def build(cls, preset, view_types, layers, fact_engine) -> Optional["FastPathPlan"]:
        """Return a plan only when it can serve EVERY loaded rule, else None."""
        if fact_engine is None or not getattr(fact_engine, "rules", None):
            return None
        plan = cls(preset, view_types, layers)
        from .fact_engine import RESERVED_IDENTIFIERS

        for rule in fact_engine.rules:
            if rule.source_view not in plan.view_types and rule.source_view != COL_PROC_NAME:
                # The rule wants a view this analyzer is not asked to build; the full path would
                # not serve it either, but say so rather than assume.
                logger.info("analyzer.fastpath.declined", rule_id=rule.id,
                            reason="source_view_not_built", source_view=rule.source_view)
                return None
            unknown = (
                set(rule.referenced_columns)
                - plan.produced_columns
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

    def flat_views(self, traces: pd.DataFrame) -> Dict[Tuple[str, ...], pd.DataFrame]:
        views: Dict[Tuple[str, ...], pd.DataFrame] = {}
        if traces is None or traces.empty:
            return views
        # FIDELITY, NOT A CHOICE. The full path groups the HLM by every view type at once, and
        # pandas groupby drops a row whose key is NA in ANY of them, so a row carrying no `epoch`
        # -- every POSIX read from a DataLoader worker, and every window.stop, neither of which is
        # annotated with one -- is absent from EVERY view, not only the epoch view. Dropping the
        # same rows here keeps this a throughput change and nothing else. It also means
        # `window_wall_time_proc_max` and `reader_posix_time_proc_max` come out NA on both paths
        # for the DLIO preset; that gap is real, predates this module, and is written up in
        # liveflow_shm/ANALYZER-FASTPATH.md rather than quietly fixed here.
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
        proc_codes, proc_uniques = pd.factorize(traces[COL_PROC_NAME])
        n_proc = max(len(proc_uniques), 1)

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

            def _emit(name: str, values: np.ndarray) -> None:
                names.append(name)
                # The full path runs `.replace(0, pd.NA)` on the main view and on each view, so a
                # layer that contributed nothing reads as absent rather than as a measured zero.
                blocks.append(np.where(values == 0, np.nan, values))

            for prefix, _condition, wants_size in self.terms:
                mask = masks[prefix] & valid
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
            view = set_window_decomposition_metrics(view, is_process_based)
            views[(view_type,)] = view.sort_index(axis=1)
        return views

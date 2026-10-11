import statistics

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .types import MetricBoundaries, Score


INTENSITY_MIN = 1 / 1024
INTENSITY_MAX = 1 / 1024**3
INTENSITY_BINS = np.geomspace(INTENSITY_MAX, INTENSITY_MIN, num=5)
PERCENTAGE_BINS = [0, 0.25, 0.5, 0.75, 0.9]
SCORE_NAMES = [
    Score.TRIVIAL.value,
    Score.LOW.value,
    Score.MEDIUM.value,
    Score.HIGH.value,
    Score.CRITICAL.value,
]
SCORE_BINS = [1, 2, 3, 4, 5]
SLOPE_BINS = [
    np.tan(np.deg2rad(15)),  # ~0.27
    np.tan(np.deg2rad(30)),  # ~0.58
    np.tan(np.deg2rad(45)),  # 1.0
    np.tan(np.deg2rad(60)),  # ~1.73
    np.tan(np.deg2rad(75)),  # ~3.73
]

def _find_metric_pairs(metrics: list[str], metric_type1: str, metric_type2: str):
    """
    Find pairs of metrics with a common prefix, one ending with metric_type1 and one with metric_type2.
    Example:
        metrics = ["foo_count_per", "foo_time_per", "bar_count_per", "bar_time_per"]
        _find_metric_pairs(metrics, "count_per", "time_per")
        -> [("foo_count_per", "foo_time_per"), ("bar_count_per", "bar_time_per")]
    """
    map1 = {
        metric_name[: -len(metric_type1)]: metric_name for metric_name in metrics if metric_name.endswith(metric_type1)
    }
    map2 = {
        metric_name[: -len(metric_type2)]: metric_name for metric_name in metrics if metric_name.endswith(metric_type2)
    }

    common_prefixes = set(map1.keys()).intersection(map2.keys())
    return [(map1[prefix], map2[prefix]) for prefix in sorted(common_prefixes)]


def find_layer_time_metrics(metrics: list, layer: str, time_metric: str):
    return [m for m in metrics if m.startswith(layer) and m.endswith(time_metric)]


def set_main_metrics(df: pd.DataFrame):
    df = df.copy()

    count_cols = [col for col in df.columns if col.endswith('count')]
    size_cols = [col for col in df.columns if col.endswith('size')]

    new_metrics: List[str] = []

    for size_col in size_cols:
        bw_col = size_col.replace('size', 'bw')
        count_col = size_col.replace('size', 'count')
        intensity_col = size_col.replace('size', 'intensity')
        time_col = size_col.replace('size', 'time')
        has_size = df[size_col] > 0
        safe_size = pd.to_numeric(df[size_col], errors="coerce").replace(0, np.nan)
        safe_time = pd.to_numeric(df[time_col], errors="coerce").replace(0, np.nan)
        safe_count = pd.to_numeric(df[count_col], errors="coerce")
        df[size_col] = df[size_col].where(has_size, pd.NA)
        df[bw_col] = (safe_size / safe_time).where(has_size, pd.NA)
        df[intensity_col] = (safe_count / safe_size).where(has_size, pd.NA)
        new_metrics.extend([bw_col, intensity_col, time_col])

    for count_col in count_cols:
        ops_col = count_col.replace('count', 'ops')
        time_col = count_col.replace('count', 'time')
        safe_time = pd.to_numeric(df[time_col], errors="coerce").replace(0, np.nan)
        df[ops_col] = pd.to_numeric(df[count_col], errors="coerce") / safe_time
        new_metrics.append(ops_col)

    df[new_metrics] = df[new_metrics].replace([np.inf, -np.inf], pd.NA).astype('Float64')

    return df.sort_index(axis=1)


def _fraction_of_total(values: pd.Series) -> pd.Series:
    """Normalize numeric measurements; an undefined normalizer stays missing.

    Full-path aggregation can leave object columns containing measured zero and
    pd.NA. Convert before arithmetic so Python float division cannot raise.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype("Float64")
    total = numeric.sum(min_count=1)
    if pd.isna(total) or not np.isfinite(total) or total <= 0:
        return pd.Series(pd.NA, index=values.index, dtype="Float64")
    return numeric / total


def set_view_metrics(
    df: pd.DataFrame,
    metric_boundaries: MetricBoundaries,
    is_view_process_based: bool,
) -> pd.DataFrame:
    df = df.copy()

    count_metric = 'count_sum'
    size_proc_metric = 'size_sum' if is_view_process_based else 'size_proc_max'
    size_call_metric = 'size_sum'
    time_proc_metric = 'time_sum' if is_view_process_based else 'time_proc_max'
    time_call_metric = 'time_sum'

    view_metrics = list(set(df.columns.tolist()))
    new_metrics: Dict[str, pd.Series] = {}

    for metric in view_metrics:
        if metric.endswith(count_metric):
            count_col = metric
            count_frac_total_col = metric.replace(count_metric, 'count_frac_total')
            new_metrics[count_frac_total_col] = _fraction_of_total(df[count_col])
        elif metric.endswith(size_proc_metric):
            size_col = metric
            size_proc_frac_total_col = metric.replace(size_proc_metric, 'size_proc_frac_total')
            new_metrics[size_proc_frac_total_col] = _fraction_of_total(df[size_col])
        elif metric.endswith(time_proc_metric):
            time_col = metric
            time_proc_frac_total_col = metric.replace(time_proc_metric, 'time_proc_frac_total')
            new_metrics[time_proc_frac_total_col] = _fraction_of_total(df[time_col])

    # Compute call_frac_total metrics (always from *_sum, regardless of view type)
    for metric in view_metrics:
        if metric.endswith(size_call_metric):
            size_call_col = metric
            size_call_frac_total_col = metric.replace(size_call_metric, 'size_call_frac_total')
            new_metrics[size_call_frac_total_col] = _fraction_of_total(df[size_call_col])
        elif metric.endswith(time_call_metric):
            time_call_col = metric
            time_call_frac_total_col = metric.replace(time_call_metric, 'time_call_frac_total')
            new_metrics[time_call_frac_total_col] = _fraction_of_total(df[time_call_col])

    count_time_proc_frac_metric_pairs = _find_metric_pairs(list(new_metrics.keys()), 'count_frac_total', 'time_proc_frac_total')
    for count_frac_total_col, time_proc_frac_total_col in count_time_proc_frac_metric_pairs:
        ops_percentile_col = count_frac_total_col.replace('count_frac_total', 'ops_percentile')
        ops_slope_col = count_frac_total_col.replace('count_frac_total', 'ops_slope')
        # WiSIO orientation: operation-share / time-share.  The geometric
        # 45-degree reference is tan(45)=1; values below it have a
        # disproportionate time share within this operation family.
        time_share = new_metrics[time_proc_frac_total_col]
        # Measured zero time is valid evidence, but its slope is undefined.
        # Mask the denominator before division, retaining NA through ranking
        # and fact evaluation (zero-normalization slopes suppress findings).
        ops_slope = new_metrics[count_frac_total_col] / time_share.where(time_share > 0)
        ops_slope = ops_slope.replace([np.inf, -np.inf], pd.NA)
        new_metrics[ops_percentile_col] = ops_slope.rank(pct=True).where(ops_slope.notna())
        new_metrics[ops_slope_col] = ops_slope

    if new_metrics:
        new_metrics_df = pd.DataFrame(new_metrics, index=df.index)
        new_metrics_df = new_metrics_df.replace([np.inf, -np.inf], pd.NA).astype('Float64')
        overlapping_cols = [col for col in new_metrics_df.columns if col in df.columns]
        if overlapping_cols:
            df = df.drop(columns=overlapping_cols)
        df = pd.concat([df, new_metrics_df], axis=1)

    return df.sort_index(axis=1)


def _thread_ids(traces: pd.DataFrame):
    """Per-row thread id: the `tid` column when present, else field 4 of `app#host#pid#tid`."""
    if "tid" in traces.columns:
        return pd.to_numeric(traces["tid"], errors="coerce")
    if "proc_name" in traces.columns:
        parts = traces["proc_name"].astype(str).str.split("#", expand=True)
        if parts.shape[1] >= 4:
            return pd.to_numeric(parts[3], errors="coerce")
    return None


def authoritative_control_rows(traces: pd.DataFrame, control_pids) -> pd.Series:
    """Boolean mask of the events that define a control window's epoch/step membership.

    The control rank's own (main-thread) events are authoritative.  Threads that live inside the
    same rank process (tf.data pool threads, helper threads) share its PID but their per-thread
    DFTracer epoch stamp can be STALE: on the Corona TF cell every `TFReader._parse_image` event of
    epochs 2 and 3 carried `epoch=1` while the rank's own `fetch.*` events carried 2 or 3, so every
    window looked like it spanned two epochs and the single-epoch gate silently suppressed every
    fact after epoch 1.  For each control PID the main thread (tid == pid) rows are used when the
    window has any; a PID with no main-thread rows keeps all its rows (no silent loss of a control
    rank), and without thread identity the filter is a no-op.  The single-epoch gate itself is not
    changed: two epochs on the authoritative rows still give `epoch_nunique == 2`.
    """
    numeric_pid = pd.to_numeric(traces["pid"], errors="coerce")
    control = {int(pid) for pid in (control_pids or []) if pd.notna(pid)}
    in_control = numeric_pid.isin(control)
    tids = _thread_ids(traces)
    if tids is None:
        return in_control
    main = in_control & (tids == numeric_pid)
    have_main = set(numeric_pid[main].dropna().astype(int).unique())
    keep_all = in_control & ~numeric_pid.fillna(-1).astype(int).isin(have_main)
    return main | keep_all


def attach_window_integrity_metrics(
    flat_views: Dict,
    traces: pd.DataFrame,
    control_pids: Optional[List[int]] = None,
) -> Dict:
    """Attach control-side epoch/step coverage to the native window view.

    The gate is defined by completed-window rank/control PIDs only.  Async
    worker rows remain available to the analyzer and debug ledger, but cannot
    redefine the control epoch or fabricate a workload step.  Missing control
    PID provenance deliberately leaves every gate field missing.
    """
    view = flat_views.get(("window",))
    if view is None:
        return flat_views
    names = (
        "epoch_nunique",
        "step_nunique",
        "step_min",
        "step_max",
    )
    result = view.copy()
    for name in names:
        result[name] = pd.NA
    needed = {"window", "pid", "epoch", "step"}
    numeric_control_pids = {
        int(pid) for pid in (control_pids or []) if pd.notna(pid)
    }
    if (traces is None or not needed.issubset(traces.columns)
            or not numeric_control_pids):
        flat_views[("window",)] = result
        return flat_views
    rows = traces.loc[
        authoritative_control_rows(traces, numeric_control_pids)
        & traces["epoch"].notna()
        & traces["step"].notna(),
        ["window", "epoch", "step"],
    ].copy()
    if rows.empty:
        flat_views[("window",)] = result
        return flat_views
    rows["epoch"] = pd.to_numeric(rows["epoch"], errors="coerce")
    rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
    rows = rows.dropna()
    stats = rows.groupby("window").agg(
        epoch_nunique=("epoch", "nunique"),
        step_nunique=("step", "nunique"),
        step_min=("step", "min"),
        step_max=("step", "max"),
    )
    for name in names:
        result[name] = result.index.to_series().map(stats[name]).astype("Float64")
    flat_views[("window",)] = result
    return flat_views


FETCH_ITER_SELECTOR = 'cat == "dataloader" & func_name == "fetch.iter"'


def attach_input_step_max_sum_metrics(
    flat_views: Dict,
    traces: pd.DataFrame,
) -> Dict:
    """Declared step-granular view for critical-path input exposure (E1, STORMER-LOOP-DESIGN sec 11/12/14).

    Family: fetch_iter (`FETCH_ITER_SELECTOR`, the same selector the accepted rank-side input rule
    uses).  View: a new ``("window", "step")`` grouping key, alongside the existing ``("window",)``
    and ``("proc_name", "window")`` views -- nothing else in the fact path changes.  Named
    count/time aggregation: ``fetch_iter_step_time_sum`` (per window, step, pid) and
    ``fetch_iter_step_count_sum``.

    Reduction to the window level, reported on the existing ``("window",)`` view as NEW columns
    that never overwrite the accepted `A = min(E, W)` skew-attribution fields:
      - `domain_input_step_max_sum_time`  = sum over steps in the window of (max over pid of that
        step's `fetch_iter_step_time_sum`) -- sum-of-max, sec 12.  Structural (argmax + sum), no
        threshold constant.
      - `domain_input_step_max_sum_count` = the same reduction over `fetch_iter_step_count_sum`,
        reported for provenance (should be step_nunique * 1 in the normal single-call-per-step case;
        a value above that says a step made more than one `fetch.iter` call and the exposure sum
        may double count within a step -- report, do not silently correct).
      - `domain_input_step_nunique`       = number of distinct steps this window's fetch_iter rows
        actually cover, so a caller can tell a real reduction from an artifact of a mostly-empty
        window (a window with `step_nunique <= 1` cannot support a sum-of-max claim and must be
        treated as missing, never as zero exposure).

    Only ADDS columns to an existing window view; does not create one. Missing/absent `step` on
    fetch_iter rows (pre-tag data, or an app that never tags this family) leaves every new column
    NA for that window -- consistent with the project rule that missing normalization/attribution
    must suppress a finding, never read as zero.
    """
    view = flat_views.get(("window",))
    if view is None:
        return flat_views
    names = (
        "domain_input_step_max_sum_time",
        "domain_input_step_max_sum_count",
        "domain_input_step_nunique",
    )
    result = view.copy()
    for name in names:
        result[name] = pd.NA
    needed = {"window", "pid", "step", "cat", "func_name", "time"}
    if traces is None or not needed.issubset(traces.columns):
        flat_views[("window",)] = result
        return flat_views
    try:
        family_mask = traces.eval(FETCH_ITER_SELECTOR)
    except Exception:
        flat_views[("window",)] = result
        return flat_views
    rows = traces.loc[
        family_mask & traces["window"].notna() & traces["step"].notna() & traces["pid"].notna(),
        ["window", "step", "pid", "time"],
    ].copy()
    if rows.empty:
        flat_views[("window",)] = result
        return flat_views
    rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
    rows["time"] = pd.to_numeric(rows["time"], errors="coerce").fillna(0.0)
    rows = rows.dropna(subset=["window", "step", "pid"])
    # Declared view: (window, step, pid) -> named count/time aggregation.
    per_pid_step = rows.groupby(["window", "step", "pid"]).agg(
        fetch_iter_step_time_sum=("time", "sum"),
        fetch_iter_step_count_sum=("time", "size"),
    )
    flat_views[("window", "step")] = per_pid_step
    # Reduction: max over pid within (window, step) -- the critical rank's own fetch wait that step.
    per_step_max = per_pid_step.groupby(["window", "step"]).agg(
        step_max_time=("fetch_iter_step_time_sum", "max"),
        step_max_count=("fetch_iter_step_count_sum", "max"),
    )
    # Reduction: sum over steps within window -- sum-of-max, sec 12.
    stats = per_step_max.groupby("window").agg(
        domain_input_step_max_sum_time=("step_max_time", "sum"),
        domain_input_step_max_sum_count=("step_max_count", "sum"),
        domain_input_step_nunique=("step_max_time", "size"),
    )
    for name in names:
        result[name] = result.index.to_series().map(stats[name])
    flat_views[("window",)] = result
    return flat_views


INPUT_EXPOSURE_JOB_SELECTOR = 'cat == "input_exposure_job" & func_name == "input_step_max_sum_job"'
# The job MEAN of every rank's own window fetch wait (DLIO job-level layer, dlio_benchmark
# utils/job_input.py).  critical (above) minus this mean is the peer-skew input a typical rank paid at
# the barrier.  Same family, same per-rank redundancy, same reduction (max over the identical values).
INPUT_OWN_JOB_SELECTOR = 'cat == "input_exposure_job" & func_name == "input_own_mean_job"'


def attach_input_job_step_max_sum_metrics(
    flat_views: Dict,
    traces: pd.DataFrame,
) -> Dict:
    """Job-wide (cross-node) sum-of-max input exposure (STORMER-LOOP-DESIGN sec 16, judge batch 5).

    `attach_input_step_max_sum_metrics` above is PER NODE: each node_agent's analyzer only sees its
    own local ranks' `fetch_iter` rows, so its max-over-pid is a max over that node's ranks, not the
    job's true `max over ALL ranks per step`. Sec 16 closes this on the RANK side instead: every
    rank runs an `Iallreduce(MAX)` over its own per-step fetch-wait vector for the window
    (`Trainer._sync_input_exposure_window`, opt-in `STORMER_INPUT_EXPOSURE_JOB_SYNC=1`), so every
    rank ends up with the IDENTICAL job-wide sum-of-max and writes it as ONE event, per rank, per
    window, to its own node's SHM lane (family `input_exposure_job`, this selector). That means the
    value is already correctly reduced before it reaches this function -- no per-pid max is needed
    here, only a per-window read of the (redundant, identical-per-rank) family time.

    Attaches, on the existing ``("window",)`` view, as NEW columns that never overwrite the
    per-node reference (`domain_input_step_max_sum_time`) or the accepted `A = min(E, W)` fields:
      - `domain_input_job_step_max_sum_time` = this family's `time` for the window (`.max()` over
        the window's redundant per-rank rows for idempotence -- NOT `.sum()`, which would multiply
        the single job-wide value by however many local ranks wrote it).
      - `domain_input_job_step_count`        = the `step_count` int-arg the rank-side reduction
        attached (the job-wide valid-step count for that window), for provenance.

    Absent when the emitting flag was never turned on (the common case today) -- every column stays
    NA, consistent with the project rule that missing attribution suppresses, never reads as zero.
    """
    view = flat_views.get(("window",))
    if view is None:
        return flat_views
    names = ("domain_input_job_step_max_sum_time", "domain_input_job_step_count",
             "domain_input_job_own_mean_time")
    result = view.copy()
    for name in names:
        result[name] = pd.NA
    needed = {"window", "cat", "func_name", "time"}
    if traces is None or not needed.issubset(traces.columns):
        flat_views[("window",)] = result
        return flat_views
    try:
        family_mask = traces.eval(INPUT_EXPOSURE_JOB_SELECTOR)
    except Exception:
        flat_views[("window",)] = result
        return flat_views
    try:
        own_mask = traces.eval(INPUT_OWN_JOB_SELECTOR)
        own_rows = traces.loc[own_mask & traces["window"].notna(), ["window", "time"]].copy()
        if not own_rows.empty:
            own_rows["time"] = pd.to_numeric(own_rows["time"], errors="coerce")
            own_stats = own_rows.dropna(subset=["window"]).groupby("window")["time"].max()
            result["domain_input_job_own_mean_time"] = result.index.to_series().map(own_stats)
    except Exception:
        pass
    rows = traces.loc[family_mask & traces["window"].notna(), ["window", "time"]].copy()
    step_count_col = None
    if "step_count" in traces.columns:
        step_count_col = pd.to_numeric(
            traces.loc[family_mask & traces["window"].notna(), "step_count"], errors="coerce")
    if rows.empty:
        flat_views[("window",)] = result
        return flat_views
    rows["time"] = pd.to_numeric(rows["time"], errors="coerce")
    rows = rows.dropna(subset=["window"])
    stats = rows.groupby("window")["time"].max().rename("domain_input_job_step_max_sum_time")
    result["domain_input_job_step_max_sum_time"] = (
        result.index.to_series().map(stats))
    if step_count_col is not None:
        rows2 = rows.assign(step_count=step_count_col.reindex(rows.index))
        sc = rows2.groupby("window")["step_count"].max().rename("domain_input_job_step_count")
        result["domain_input_job_step_count"] = result.index.to_series().map(sc)
    flat_views[("window",)] = result
    return flat_views


# CONFIRMED against domain2's landed emission (tree-v4.4.1, train.py `_emit_exposed_comm`):
# `inst.log_event(name="comm.exposed", cat="comm", ..., int_args={"step": step})`, one call per rank
# per step, no-op below a sub-microsecond floor (a fully-hidden step emits nothing, not a zero-duration
# event) -- `exposed_comm_ms = max(0, last_all_reduce_end - backward_end)` via
# `comm_hook.compute_exposed_comm_ms` (STORMER-LOOP-DESIGN sec 17 v2). `func_name == "comm.exposed"`
# keeps this family disjoint from the existing `comm`/`comm_all_reduce` selectors (`cat == "comm"` with
# no func_name filter, and `func_name == "all_reduce"` respectively) despite sharing `cat == "comm"`.
COMM_EXPOSED_SELECTOR = 'cat == "comm" & func_name == "comm.exposed"'
COMM_EXPOSED_TRIM_FRACTION = 0.10   # the "10%" in "10%-trimmed mean" -- a declared, disclosed choice,
                                     # not hidden inside the reduction; median is the primary estimator.


def _trimmed_mean(values, frac):
    """Symmetric trimmed mean: drop the lowest and highest `frac` fraction each side, then average
    what remains. `frac` must be stated wherever this is cited (sec 17.1); it is not constant-free
    the way a plain median/max/argmax is, and is reported as an explicit alternate, never silently
    substituted for the median."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    k = int(n * frac)
    trimmed = vals[k: n - k] if n - 2 * k > 0 else vals
    return sum(trimmed) / len(trimmed)


def attach_comm_exposed_median_metrics(
    flat_views: Dict,
    traces: pd.DataFrame,
) -> Dict:
    """Window comm exposure from domain2's per-step exposed-comm event (STORMER-LOOP-DESIGN sec 17 v2,
    orchestrator decision 2026-09-27, full input+comm loop).

    Domain2's B1 agreement check found raw per-step exposed-comm sums are spike-dominated (a
    non-blocking CUDA-event timer absorbs cross-rank arrival skew at stall steps, up to 14x a
    blocking-probe reference that measures a DIFFERENT quantity and must not be used to calibrate
    this one). The window fact is therefore the MEDIAN (primary) or a declared 10%-trimmed mean
    (alternate, `COMM_EXPOSED_TRIM_FRACTION`) of the window's per-step values, times the number of
    steps in the window -- both are structural population statistics (a percentile and a symmetric
    trim), not thresholds on the DATA itself; only the trim fraction is a disclosed numeric choice.

    Family: `COMM_EXPOSED_SELECTOR` (confirmed against domain2's landed emission, tree-v4.4.1). View: a new
    `("window", "step")` grouping key (parallel to `attach_input_step_max_sum_metrics`), named
    count/time aggregation `comm_exposed_step_time_sum`/`comm_exposed_step_count_sum` per
    `(window, step, pid)`. Reduction: per step, MAX over pid (the job-wide CRITICAL exposed comm that
    step -- never averaged away, same "critical path" philosophy as every other reduction in this
    file); per window, MEDIAN (and 10%-trimmed mean) over the window's per-step maxima, multiplied by
    the count of steps that went into it.

    New columns on the existing `("window",)` view, never overwriting the older `domain_comm_time_
    proc_crit`:
      - `domain_comm_pure_exposure_time_v2` = median(job-wide per-step exposed comm) * step_nunique.
      - `domain_comm_exposure_trimmed10_time_v2` = the declared 10%-trimmed-mean alternate, same
        step count multiplier -- reported, never silently substituted for the median above.
      - `domain_comm_exposure_step_nunique` = the step count used, so a caller can tell a real
        reduction from a near-empty window (mirrors `domain_input_step_nunique`'s own suppression rule
        -- a `step_nunique <= 1` window must be treated as missing, not zero exposure).

    ZERO-FILL FIX (2026-09-27, orchestrator): `_emit_exposed_comm` is a no-op below a sub-microsecond
    floor, so a fully-hidden step emits NO row at all, not a `time == 0` row. Left alone, the per-step
    reduction below would only ever see steps that had at least one rank's row, silently DROPPING the
    true zero-exposure steps from the population and biasing the median upward. The authoritative step
    population for a window is instead taken from the SAME step-tagged `fetch_iter` family sec 12/16
    already use (every step that produced a fetch wait, tagged the same way, in the same run): a step
    present there but absent from `comm.exposed` rows is fully hidden and contributes exactly 0.0, not
    a missing value. If a window has NO `fetch_iter` step tags at all, there is no authoritative
    population to zero-fill against, and this fact stays NA for that window -- suppression, never a
    fabricated population, per the project's missing-attribution rule.
    """
    return _comm_median_metrics(
        flat_views, traces, selector=COMM_EXPOSED_SELECTOR,
        time_col="domain_comm_pure_exposure_time_v2",
        trimmed_col="domain_comm_exposure_trimmed10_time_v2",
        nunique_col="domain_comm_exposure_step_nunique",
        pid_step_time_col="comm_exposed_step_time_sum",
        pid_step_count_col="comm_exposed_step_count_sum",
    )


# sec 17 v3 (orchestrator decision 2026-09-27): domain2's exposed-comm validation found exposed comm
# tiny and HIGHER at cap400 (1.8ms cap25/W16 vs 7.6ms cap400/W16) while TOTAL comm was 173 vs 81ms --
# Step 0c's cap400 gain looks like comm/compute INTERFERENCE, not an exposed-tail effect. `comm.total`
# (train.py's `_emit_total_comm`, event-probe mode only) is the sum of every bucket's own span that
# step, an UPPER BOUND on what a cap change can be worth. Selected via
# `DFOPTIMIZER_COMM_EXPOSURE_SOURCE=total` (clbo_planner.py); JOINT_VERIFY's measured wall improvement
# is the arbiter that decides whether a move made on this bound is kept (cbandit_planner.py already
# does this, unmodified -- see its docstring for the "proposal bound, wall as arbiter" framing).
COMM_TOTAL_SELECTOR = 'cat == "comm" & func_name == "comm.total"'


def attach_comm_total_median_metrics(
    flat_views: Dict,
    traces: pd.DataFrame,
) -> Dict:
    """Window TOTAL comm (sec 17 v3) -- same reduction as `attach_comm_exposed_median_metrics`
    (per-step max-over-pid, then window median/trimmed-mean over steps, zero-filled from the
    `fetch_iter` step population), applied to the `comm.total` family instead of `comm.exposed`.

    New columns on the existing ``("window",)`` view, never overwriting `domain_comm_pure_exposure_
    time_v2` (sec 17 v2) or the older `domain_comm_time_proc_crit`:
      - `domain_comm_total_median_time_v3` = median(job-wide per-step TOTAL comm) * step_nunique.
      - `domain_comm_total_trimmed10_time_v3` = the declared 10%-trimmed-mean alternate.
      - `domain_comm_total_step_nunique` = the step count used (same suppression rule as sec 17 v2:
        `<= 1` must be treated as missing, not zero).
    """
    return _comm_median_metrics(
        flat_views, traces, selector=COMM_TOTAL_SELECTOR,
        time_col="domain_comm_total_median_time_v3",
        trimmed_col="domain_comm_total_trimmed10_time_v3",
        nunique_col="domain_comm_total_step_nunique",
        pid_step_time_col="comm_total_step_time_sum",
        pid_step_count_col="comm_total_step_count_sum",
    )


def _comm_median_metrics(
    flat_views: Dict,
    traces: pd.DataFrame,
    *,
    selector: str,
    time_col: str,
    trimmed_col: str,
    nunique_col: str,
    pid_step_time_col: str,
    pid_step_count_col: str,
) -> Dict:
    """Shared reduction for sec 17 v2 (`comm.exposed`) and v3 (`comm.total`): per-step MAX over pid,
    then window MEDIAN (primary) and a declared 10%-trimmed-mean alternate over steps, multiplied by
    the step count, zero-filled against the `fetch_iter` step population (a step present there but
    absent from `selector`'s rows is fully hidden and contributes 0.0, never silently dropped nor
    fabricated as NA when no window can be built -- see the ZERO-FILL FIX note on the v2 caller)."""
    view = flat_views.get(("window",))
    if view is None:
        return flat_views
    names = (time_col, trimmed_col, nunique_col)
    result = view.copy()
    for name in names:
        result[name] = pd.NA
    needed = {"window", "pid", "step", "cat", "func_name", "time"}
    if traces is None or not needed.issubset(traces.columns):
        flat_views[("window",)] = result
        return flat_views
    fetch_rows = traces.loc[
        traces.eval(FETCH_ITER_SELECTOR) & traces["window"].notna() & traces["step"].notna(),
        ["window", "step"],
    ].copy()
    if fetch_rows.empty:
        flat_views[("window",)] = result
        return flat_views
    fetch_rows["step"] = pd.to_numeric(fetch_rows["step"], errors="coerce")
    fetch_rows = fetch_rows.dropna(subset=["window", "step"])
    fetch_steps_by_window = fetch_rows.groupby("window")["step"].apply(lambda s: set(s.unique()))
    try:
        family_mask = traces.eval(selector)
    except Exception:
        family_mask = pd.Series(False, index=traces.index)
    rows = traces.loc[
        family_mask & traces["window"].notna() & traces["step"].notna() & traces["pid"].notna(),
        ["window", "step", "pid", "time"],
    ].copy()
    if not rows.empty:
        rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
        rows["time"] = pd.to_numeric(rows["time"], errors="coerce").fillna(0.0)
        rows = rows.dropna(subset=["window", "step", "pid"])
    per_pid_step = rows.groupby(["window", "step", "pid"]).agg(
        **{pid_step_time_col: ("time", "sum"), pid_step_count_col: ("time", "size")}
    ) if not rows.empty else pd.DataFrame(columns=[pid_step_time_col])
    # Note: deliberately does NOT touch flat_views[("window", "step")] -- sec 14's own declared view
    # under that same key (fetch_iter) is a different family and must not be overwritten by this one.
    per_step_max = (
        per_pid_step.groupby(["window", "step"])[pid_step_time_col].max()
        if not per_pid_step.empty else pd.Series(dtype=float)
    )

    def _reduce(vals):
        n = len(vals)
        if not n:
            return None, None, None
        med = statistics.median(vals) * n
        trimmed = (_trimmed_mean(vals, COMM_EXPOSED_TRIM_FRACTION) or 0.0) * n
        return med, trimmed, n

    out = {name: {} for name in names}
    for window, steps in fetch_steps_by_window.items():
        vals = [per_step_max.get((window, s), 0.0) for s in steps]
        med, trimmed, n = _reduce(vals)
        out[time_col][window] = med
        out[trimmed_col][window] = trimmed
        out[nunique_col][window] = n
    for name in names:
        result[name] = result.index.to_series().map(out[name])
    flat_views[("window",)] = result
    return flat_views


def set_cross_layer_metrics(
    df: pd.DataFrame,
    layers: List[str],
    layer_deps: Dict[str, Optional[str]],
    async_layers: List[str],
    derived_metrics: Dict[str, Dict[str, str]],
    is_view_process_based: bool,
    time_boundary_layer: str,
) -> pd.DataFrame:
    time_proc_metric = 'time_sum' if is_view_process_based else 'time_proc_max'
    time_call_metric = 'time_sum'
    compute_time_proc_metric = f"compute_{time_proc_metric}"
    compute_time_call_metric = f"compute_{time_call_metric}"
    time_proc_boundary_metric = f"{time_boundary_layer}_{time_proc_metric}"
    time_call_boundary_metric = f"{time_boundary_layer}_{time_call_metric}"

    # Collect new columns and assign them in batch to avoid fragmentation warnings
    x_layer_metrics: Dict[str, pd.Series] = {}

    # Set relational time metrics for layers
    for layer in layers:
        layer_time_proc = df[f"{layer}_{time_proc_metric}"]
        layer_time_call = df[f"{layer}_{time_call_metric}"]

        # Proc: frac_boundary
        time_proc_frac_boundary_col = f"{layer}_time_proc_frac_{time_boundary_layer}"
        x_layer_metrics[time_proc_frac_boundary_col] = layer_time_proc / df[time_proc_boundary_metric]
        # Call: frac_boundary
        time_call_frac_boundary_col = f"{layer}_time_call_frac_{time_boundary_layer}"
        x_layer_metrics[time_call_frac_boundary_col] = layer_time_call / df[time_call_boundary_metric]

        child_layers = [child for child, parent in layer_deps.items() if parent == layer]
        if not child_layers:
            continue

        # Proc: overhead
        o_time_proc_col = f"o_{layer}_{time_proc_metric}"
        o_time_proc_frac_boundary_col = f"o_{layer}_time_proc_frac_{time_boundary_layer}"
        o_time_proc_frac_self_col = f"o_{layer}_time_proc_frac_self"
        o_time_proc_frac_total_col = o_time_proc_col.replace(time_proc_metric, 'time_proc_frac_total')

        child_time_proc_sum = sum(df[f"{child}_{time_proc_metric}"].fillna(0) for child in child_layers)
        o_time_proc = np.maximum(layer_time_proc - child_time_proc_sum, 0)
        o_time_proc_total = o_time_proc.sum()

        o_time_proc_series = pd.array(o_time_proc, dtype='Float64')
        x_layer_metrics[o_time_proc_col] = o_time_proc_series
        x_layer_metrics[o_time_proc_frac_boundary_col] = o_time_proc_series / df[time_proc_boundary_metric]
        x_layer_metrics[o_time_proc_frac_self_col] = o_time_proc_series / layer_time_proc
        x_layer_metrics[o_time_proc_frac_total_col] = pd.NA
        if o_time_proc_total > 0:
            x_layer_metrics[o_time_proc_frac_total_col] = o_time_proc_series / o_time_proc_total

        # Call: overhead
        o_time_call_col = f"o_{layer}_{time_call_metric}"
        o_time_call_frac_boundary_col = f"o_{layer}_time_call_frac_{time_boundary_layer}"
        o_time_call_frac_self_col = f"o_{layer}_time_call_frac_self"
        o_time_call_frac_total_col = f"o_{layer}_time_call_frac_total"

        child_time_call_sum = sum(df[f"{child}_{time_call_metric}"].fillna(0) for child in child_layers)
        o_time_call = np.maximum(layer_time_call - child_time_call_sum, 0)
        o_time_call_total = o_time_call.sum()

        o_time_call_series = pd.array(o_time_call, dtype='Float64')
        x_layer_metrics[o_time_call_col] = o_time_call_series
        x_layer_metrics[o_time_call_frac_boundary_col] = o_time_call_series / df[time_call_boundary_metric]
        x_layer_metrics[o_time_call_frac_self_col] = o_time_call_series / layer_time_call
        x_layer_metrics[o_time_call_frac_total_col] = pd.NA
        if o_time_call_total > 0:
            x_layer_metrics[o_time_call_frac_total_col] = o_time_call_series / o_time_call_total

        # Proc + Call: frac_parent
        layer_has_time_proc = layer_time_proc.sum() > 0
        layer_has_time_call = layer_time_call.sum() > 0
        for child_layer in child_layers:
            time_proc_frac_parent_col = f"{child_layer}_time_proc_frac_parent"
            x_layer_metrics[time_proc_frac_parent_col] = pd.NA
            if layer_has_time_proc:
                x_layer_metrics[time_proc_frac_parent_col] = df[f"{child_layer}_{time_proc_metric}"] / layer_time_proc

            time_call_frac_parent_col = f"{child_layer}_time_call_frac_parent"
            x_layer_metrics[time_call_frac_parent_col] = pd.NA
            if layer_has_time_call:
                x_layer_metrics[time_call_frac_parent_col] = df[f"{child_layer}_{time_call_metric}"] / layer_time_call

    # Set relational time metrics for derived metrics
    for layer in derived_metrics:
        for dm in derived_metrics[layer]:
            dm_col = f"{layer}_{dm}"
            dm_time_proc_col = f"{dm_col}_{time_proc_metric}"
            dm_time_call_col = f"{dm_col}_{time_call_metric}"

            # Proc
            if dm_time_proc_col in df.columns:
                dm_time_proc = df[dm_time_proc_col]
                dm_time_proc_total = dm_time_proc.sum()

                x_layer_metrics[f"{dm_col}_time_proc_frac_{time_boundary_layer}"] = dm_time_proc / df[time_proc_boundary_metric]
                x_layer_metrics[f"{dm_col}_time_proc_frac_parent"] = dm_time_proc / df[f"{layer}_{time_proc_metric}"]
                x_layer_metrics[f"{dm_col}_time_proc_frac_total"] = pd.NA
                if dm_time_proc_total > 0:
                    x_layer_metrics[f"{dm_col}_time_proc_frac_total"] = dm_time_proc / dm_time_proc_total

            # Call
            if dm_time_call_col in df.columns:
                dm_time_call = df[dm_time_call_col]
                dm_time_call_total = dm_time_call.sum()

                x_layer_metrics[f"{dm_col}_time_call_frac_{time_boundary_layer}"] = dm_time_call / df[time_call_boundary_metric]
                x_layer_metrics[f"{dm_col}_time_call_frac_parent"] = dm_time_call / df[f"{layer}_{time_call_metric}"]
                x_layer_metrics[f"{dm_col}_time_call_frac_total"] = pd.NA
                if dm_time_call_total > 0:
                    x_layer_metrics[f"{dm_col}_time_call_frac_total"] = dm_time_call / dm_time_call_total

    # Set unoverlapped times if there is compute time
    if compute_time_proc_metric in df.columns:
        compute_time_proc = df[compute_time_proc_metric].fillna(0).astype('Float64')
        compute_time_call = df[compute_time_call_metric].fillna(0).astype('Float64')
        # Set unoverlapped time metrics
        for async_layer in async_layers:
            # Proc: unoverlapped
            time_proc_col = f"{async_layer}_{time_proc_metric}"
            u_time_proc_col = f"u_{time_proc_col}"

            layer_time_proc = df[time_proc_col]
            u_time_proc = (layer_time_proc - compute_time_proc).clip(lower=0).astype('Float64')
            u_time_proc_total = u_time_proc.sum()

            u_time_proc_series = pd.array(u_time_proc, dtype='Float64')
            x_layer_metrics[u_time_proc_col] = u_time_proc_series
            x_layer_metrics[f"u_{async_layer}_time_proc_frac_self"] = u_time_proc_series / layer_time_proc
            x_layer_metrics[f"u_{async_layer}_time_proc_frac_{time_boundary_layer}"] = u_time_proc_series / df[time_proc_boundary_metric]
            x_layer_metrics[f"u_{async_layer}_time_proc_frac_total"] = pd.NA
            if u_time_proc_total > 0:
                x_layer_metrics[f"u_{async_layer}_time_proc_frac_total"] = u_time_proc_series / u_time_proc_total

            parent_layer = layer_deps.get(async_layer)
            if parent_layer:
                x_layer_metrics[f"u_{async_layer}_time_proc_frac_parent"] = u_time_proc_series / df[f"{parent_layer}_{time_proc_metric}"]

            # Call: unoverlapped
            time_call_col = f"{async_layer}_{time_call_metric}"
            u_time_call_col = f"u_{time_call_col}"

            layer_time_call = df[time_call_col]
            u_time_call = (layer_time_call - compute_time_call).clip(lower=0).astype('Float64')
            u_time_call_total = u_time_call.sum()

            u_time_call_series = pd.array(u_time_call, dtype='Float64')
            x_layer_metrics[u_time_call_col] = u_time_call_series
            x_layer_metrics[f"u_{async_layer}_time_call_frac_self"] = u_time_call_series / layer_time_call
            x_layer_metrics[f"u_{async_layer}_time_call_frac_{time_boundary_layer}"] = u_time_call_series / df[time_call_boundary_metric]
            x_layer_metrics[f"u_{async_layer}_time_call_frac_total"] = pd.NA
            if u_time_call_total > 0:
                x_layer_metrics[f"u_{async_layer}_time_call_frac_total"] = u_time_call_series / u_time_call_total

            if parent_layer:
                x_layer_metrics[f"u_{async_layer}_time_call_frac_parent"] = u_time_call_series / df[f"{parent_layer}_{time_call_metric}"]

    if x_layer_metrics:
        x_layer_metrics_df = pd.DataFrame(x_layer_metrics, index=df.index)
        x_layer_metrics_df = x_layer_metrics_df.replace([np.inf, -np.inf], pd.NA).astype('Float64')
        overlapping_cols = [col for col in x_layer_metrics_df.columns if col in df.columns]
        if overlapping_cols:
            df = df.drop(columns=overlapping_cols)
        df = pd.concat([df.copy(), x_layer_metrics_df], axis=1)

    df = set_window_decomposition_metrics(df, is_view_process_based)

    return df.sort_index(axis=1)


# --- rank-coherent domain decomposition ---------------------------------------------------------
#
# The four domains PARTITION the window. Per process, per window, on `stgi_capacity_gate1`
# (StorMer, Corona, 1 node, 8 ranks, the shipped default), every one of the eight ranks closes
#
#     stall_storage + comm + (compute - comm)  =  window_wall - 0.16 s
#
# in 99.4 to 99.6 percent of the wall, in every interior window of the run. `comm` is NESTED inside
# `compute` (the DDP all_reduce fires during backward) which is why compute is reported net of it,
# and `stall_decode` is nested inside the same consumer-side exposure total as `stall_storage`, so
# the two split one quantity rather than adding to it. A share of the window is therefore the right
# statistic and the planner's use of it is not the bug.
#
# WHAT WAS THE BUG. `<layer>_time_proc_max` is the maximum over processes taken SEPARATELY for
# every layer. In a lockstep collective a rank starved of input waits in `stall` while the ranks
# waiting for it at the barrier wait in `comm`, so the input maximum and the comm maximum come from
# OPPOSITE ranks by construction. The four numbers are then not a decomposition of any process's
# timeline, and their sum is not bounded by the window: on the same run they sum to between 96 and
# 139 percent of it while every individual rank closes to 99.5. The error is largest exactly where
# the rank skew, which is the phenomenon, is largest.
#
# The fix is to decompose PER PROCESS first and then take a vector WHOLE from one real process.
# Two are published, because they answer different questions and both are needed:
#
#   crit  the process with the largest input+decode exposure, i.e. the critical path through the
#         data pipeline. Every other rank's comm is inflated by this rank's stall, so this is the
#         vector an input knob must be priced against, and it is what the planner reads.
#   typ   the process at the median of the same ordering. This is the typical rank, and it is the
#         one comparable to a workload instrument that prints from rank 0 only -- StorMer's
#         COMM_TIMING and DLIO's STEP_DIRECT/FETCH_DIRECT both do -- so it is what
#         `liveflow_shm/tests/test_domain_attribution.py` validates against ground truth.
#
# Both are taken from a single process, so both close on the wall by construction and their
# residual is a measurement of what the annotations did not name rather than an aggregation error.

# CROSS-RANK INPUT-SKEW ATTRIBUTION (2026-09-25; liveflow_shm/SKEW-ATTRIBUTION-2026-09-25.md).
# A bulk-synchronous step makes the ranks that arrive EARLY wait for the LATE one inside the
# collective, so that wait is measured as comm on the early ranks although the late rank's own
# input stall caused it.  `coherent_domain_vectors` therefore also reports a measured accounting
# identity per window (all seconds, one value per window):
#   late rank L            = the active rank that waited LEAST in the collective (smallest comm);
#                            it arrived last.  (Not `crit`: crit is the largest input exposure.)
#   peers                  = every other active rank
#   E  excess_time         = max(0, input(L) - median_peers(input))   input = stall.storage +
#                            stall.decode, or fetch_iter - decode when the stall instrument is absent
#   W  wait_time           = max(0, median_peers(comm) - comm(L))     comm = comm.all_reduce/barrier
#   A  attributed_time     = min(E, W)     the part of the peers' comm wait that L's own excess
#                            input can account for; never more than either measured quantity
#   comm_residual_time     = median_peers(comm) - A      what stays communication
#   coverage               = 1 only if the window has >= 2 active ranks, every active rank measured
#                            a positive comm time (an absent comm record is not a zero), an input
#                            instrument is present, and, when the caller knows it, the active-rank
#                            count equals the expected rank count.  Otherwise every value is 0 and
#                            coverage is 0: attribution is SUPPRESSED, never invented.
# There is no constant: E and W are differences of measurements taken in the same window, and A is
# their minimum.
SKEW_COLUMNS = (
    "domain_skew_coverage",
    "domain_skew_late_proc_input_time", "domain_skew_late_proc_comm_time",
    "domain_skew_peer_median_input_time", "domain_skew_peer_median_comm_time",
    "domain_input_skew_excess_time", "domain_comm_barrier_wait_time",
    "domain_input_skew_attributed_time", "domain_comm_residual_after_skew_time",
)


DOMAIN_VECTOR_SUFFIXES = ("crit", "typ")
DOMAIN_VECTOR_COLUMNS = tuple(
    "%s_time_proc_%s" % (name, suffix)
    for suffix in DOMAIN_VECTOR_SUFFIXES
    for name in ("domain_input", "domain_decode", "domain_comm", "domain_compute",
                 "domain_device", "residual", "window_wall")
) + ("domain_coherent", "domain_input_spread_time_proc", "domain_procs_active",
     # ORDER STATISTICS OF THE PER-PROCESS INPUT EXPOSURE, in seconds, over the processes that ran
     # the window. The scalar spread cannot tell TWO starved ranks from a genuinely input-bound
     # run, and that distinction decided whether the 2026-09-08 StorMer candidate was a workload
     # property or our own oversubscription: on `stgi_capacity_gate1` the same two of eight ranks
     # carry 5 to 23 s per window while the other six sit at 0.3 s, so `p1` and `p2` are large and
     # `med` is not. Published because the judge verdict of 2026-09-08 requires the sorted per-rank
     # vector wherever an input domain is claimed, and because `p1 - p2` is the planner's payoff
     # bound when it relieves the critical rank.
     "domain_input_time_proc_p1", "domain_input_time_proc_p2",
     "domain_input_time_proc_med", "domain_input_time_proc_min") + SKEW_COLUMNS


def skew_attribution(
    active: np.ndarray, input_t: np.ndarray, comm_t: np.ndarray,
    *, input_instrument: bool, comm_instrument: bool, expected_procs: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Per-window measured input-skew accounting; see the note on SKEW_COLUMNS.

    `active`, `input_t`, `comm_t` are (n_groups, n_proc).  Pure and deterministic.
    """
    n_groups = active.shape[0]
    out = {name: np.zeros(n_groups) for name in SKEW_COLUMNS}
    if not (input_instrument and comm_instrument):
        return out
    for g in range(n_groups):
        idx = np.flatnonzero(active[g])
        if idx.size < 2:
            continue
        if expected_procs is not None and idx.size != int(expected_procs):
            continue
        comm = comm_t[g][idx]
        if not np.all(comm > 0.0):
            continue                       # a rank with no comm record: cannot order arrivals
        inp = input_t[g][idx]
        # Late rank = smallest comm; exact ties resolved toward the SMALLEST input (conservative:
        # a tie can never manufacture attribution).
        tied = np.flatnonzero(comm == comm.min())
        late = tied[np.argmin(inp[tied])]
        peers = np.delete(np.arange(idx.size), late)
        peer_in = float(np.median(inp[peers]))
        peer_comm = float(np.median(comm[peers]))
        excess = max(0.0, float(inp[late]) - peer_in)
        wait = max(0.0, peer_comm - float(comm[late]))
        attributed = min(excess, wait)
        out["domain_skew_coverage"][g] = 1.0
        out["domain_skew_late_proc_input_time"][g] = float(inp[late])
        out["domain_skew_late_proc_comm_time"][g] = float(comm[late])
        out["domain_skew_peer_median_input_time"][g] = peer_in
        out["domain_skew_peer_median_comm_time"][g] = peer_comm
        out["domain_input_skew_excess_time"][g] = excess
        out["domain_comm_barrier_wait_time"][g] = wait
        out["domain_input_skew_attributed_time"][g] = attributed
        out["domain_comm_residual_after_skew_time"][g] = peer_comm - attributed
    return out


def coherent_domain_vectors(mats: Dict[str, np.ndarray],
                            expected_procs: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Per-window domain vectors taken whole from one process each.

    `mats` maps a layer prefix to a (n_groups, n_proc) array of that layer's summed time for that
    (window group, process) pair. `window_wall`, `comm` and `compute` are required; the rest
    default to zero. Exposure-key PRESENCE selects the input definition, exactly as
    `set_window_decomposition_metrics` does, so a measured zero exposure stays distinct from an
    absent instrument.
    """
    wall = mats["window_wall"]
    n_groups, n_proc = wall.shape
    zeros = np.zeros_like(wall)

    def m(name):
        return mats.get(name, zeros)

    exposure_present = "stall_storage" in mats
    if exposure_present:
        input_t = m("stall_storage")
        decode_t = m("stall_decode")
    else:
        decode_t = m("reader_decode")
        input_t = np.clip(m("fetch_iter") - decode_t, 0.0, None)
    comm_t = m("comm")
    device_t = m("device_transfer")
    # HOST TO DEVICE STAGING. It is annotated, it is on the critical path between the fetch and the
    # step, and until 2026-09-08 it belonged to no domain, so it sat in the residual: on
    # `stgi_capacity_gate1` the residual equals it to within 0.04 s in every interior window.
    # It is folded into compute rather than made a fifth domain because the workload's own
    # instrument counts it inside `step` (COMM_TIMING's `other`), so this is what makes the two
    # commensurable, and because neither compute nor a device domain owns a registered knob today,
    # so no decision changes. It is published separately as `domain_device_*` so a run where it
    # grows is visible rather than hidden inside compute -- and if the GLANCED-IO transfer-size
    # knob is ever registered, that is the knob which owns it and it must become its own domain.
    compute_t = np.clip(m("compute") - comm_t, 0.0, None) + device_t

    # A process with no window wall did not run this window. DataLoader worker processes are in
    # the same frame and would otherwise be selectable as a row of zeros, which would report a
    # perfectly empty window and a residual equal to the whole wall.
    active = wall > 0
    order = input_t + decode_t

    crit = np.zeros(n_groups, dtype=int)
    typ = np.zeros(n_groups, dtype=int)
    n_active = np.zeros(n_groups, dtype=float)
    for g in range(n_groups):
        idx = np.flatnonzero(active[g])
        if idx.size == 0:
            # There is no rank-local window to price.  Do not select an arbitrary
            # zero-wall row merely to make a vector-shaped result: downstream must
            # see this as explicitly incoherent and decline the window.
            continue
        n_active[g] = float(idx.size)
        ranked = idx[np.argsort(order[g][idx], kind="stable")]
        crit[g] = ranked[-1]
        typ[g] = ranked[(ranked.size - 1) // 2]

    out = {}
    rows = np.arange(n_groups)
    for suffix, sel in (("crit", crit), ("typ", typ)):
        i, d, c, k, w = (input_t[rows, sel], decode_t[rows, sel], comm_t[rows, sel],
                         compute_t[rows, sel], wall[rows, sel])
        out["domain_input_time_proc_%s" % suffix] = i
        out["domain_decode_time_proc_%s" % suffix] = d
        out["domain_comm_time_proc_%s" % suffix] = c
        out["domain_compute_time_proc_%s" % suffix] = k
        out["domain_device_time_proc_%s" % suffix] = device_t[rows, sel]
        out["window_wall_time_proc_%s" % suffix] = w
        out["residual_time_proc_%s" % suffix] = w - (i + d + c + k)
    spread = np.zeros(n_groups)
    for g in range(n_groups):
        vals = order[g][np.flatnonzero(active[g])]
        if vals.size:
            spread[g] = vals.max() - vals.min()
    out["domain_input_spread_time_proc"] = spread
    out["domain_procs_active"] = n_active
    # The sorted per-process input exposure, reported as four order statistics. See the note on
    # DOMAIN_VECTOR_COLUMNS. `p1` is the critical rank's own exposure and equals crit's
    # input+decode; `med` is the typical rank's and equals typ's; `p2` is what makes the shape of
    # the distribution legible, because `p1` large with `med` small and `p2` large is a small set
    # of starved ranks, while `p1` close to `med` is a workload whose ranks all wait.
    p1 = np.zeros(n_groups)
    p2 = np.zeros(n_groups)
    med = np.zeros(n_groups)
    lo = np.zeros(n_groups)
    for g in range(n_groups):
        vals = np.sort(order[g][np.flatnonzero(active[g])])
        if vals.size == 0:
            continue
        p1[g] = vals[-1]
        p2[g] = vals[-2] if vals.size > 1 else vals[-1]
        med[g] = vals[(vals.size - 1) // 2]
        lo[g] = vals[0]
    out["domain_input_time_proc_p1"] = p1
    out["domain_input_time_proc_p2"] = p2
    out["domain_input_time_proc_med"] = med
    out["domain_input_time_proc_min"] = lo
    # This is provenance, not a confidence score.  A zero says the analyzer
    # received process matrices but no process ran this window; the optimizer
    # must fail closed rather than reinterpret those same inputs as legacy
    # aggregate metrics.
    out["domain_coherent"] = (n_active > 0).astype(float)
    out.update(skew_attribution(
        active, input_t + decode_t, comm_t,
        input_instrument=exposure_present or "fetch_iter" in mats,
        comm_instrument="comm" in mats, expected_procs=expected_procs))
    return out


def set_window_decomposition_metrics(
    df: pd.DataFrame,
    is_view_process_based: bool,
    coherent: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Publish a signed per-window closure residual when wall is available.

    ``window_wall`` comes from the duration-bearing ``window.stop`` event and
    is independent of the domain annotations.  A positive residual therefore
    exposes time which input, decode, communication, and compute did not name.
    A negative residual exposes overlap or double counting.  Keeping the sign
    is important because clipping would hide the latter failure mode.

    Exposure metrics, when present, are selected by key presence rather than
    truthiness.  A real zero exposure is therefore different from an absent
    instrument.  The planner records the same distinction when it applies its
    fallback domain mapping.
    """
    time_metric = "time_sum" if is_view_process_based else "time_proc_max"
    wall_col = f"window_wall_{time_metric}"
    if wall_col not in df.columns:
        return df

    def values(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(0.0, index=df.index, dtype="Float64")
        return pd.to_numeric(df[name], errors="coerce").fillna(0).astype("Float64")

    fetch = values(f"fetch_iter_{time_metric}")
    comm = values(f"comm_{time_metric}")
    compute_total = values(f"compute_{time_metric}")
    storage_exposure_col = f"stall_storage_{time_metric}"
    decode_exposure_col = f"stall_decode_{time_metric}"
    # DLIO has an aggregate consumer-side storage exposure but no separately measurable decode
    # stage. Storage-key presence is the contract. A present zero remains distinct from absence.
    exposure_present = storage_exposure_col in df.columns

    if exposure_present:
        input_time = values(storage_exposure_col)
        decode_time = values(decode_exposure_col)
    else:
        decode_time = values(f"reader_decode_{time_metric}")
        input_time = (fetch - decode_time).clip(lower=0)

    device_time = values(f"device_transfer_{time_metric}")
    compute_time = (compute_total - comm).clip(lower=0) + device_time
    wall = values(wall_col)
    residual = wall - (input_time + decode_time + comm + compute_time)

    columns = {
        f"domain_input_{time_metric}": input_time,
        f"domain_decode_{time_metric}": decode_time,
        f"domain_comm_{time_metric}": comm,
        f"domain_compute_{time_metric}": compute_time,
        f"domain_device_{time_metric}": device_time,
        f"residual_{time_metric}": residual,
        "domain_exposure_present": float(exposure_present),
    }

    # THE COHERENT VECTORS. See the comment above `coherent_domain_vectors`. The columns above stay
    # exactly as they were, because other rules read them and because a metric whose meaning
    # changes silently is the defect this project keeps catching; the crit/typ family is new and
    # named for what it is.
    if coherent is not None:
        for name, arr in coherent.items():
            columns[name] = pd.Series(np.asarray(arr, dtype=float), index=df.index)
    elif is_view_process_based:
        # One row per process already: the row IS a coherent vector, so crit and typ are itself.
        for suffix in DOMAIN_VECTOR_SUFFIXES:
            columns[f"domain_input_time_proc_{suffix}"] = input_time
            columns[f"domain_decode_time_proc_{suffix}"] = decode_time
            columns[f"domain_comm_time_proc_{suffix}"] = comm
            columns[f"domain_compute_time_proc_{suffix}"] = compute_time
            columns[f"domain_device_time_proc_{suffix}"] = device_time
            columns[f"window_wall_time_proc_{suffix}"] = wall
            columns[f"residual_time_proc_{suffix}"] = residual
        columns["domain_input_spread_time_proc"] = 0.0
        columns["domain_procs_active"] = 1.0
        columns["domain_coherent"] = 1.0
        for name in ("p1", "p2", "med", "min"):
            columns["domain_input_time_proc_%s" % name] = input_time + decode_time
        for name in SKEW_COLUMNS:          # no per-process matrices: attribution suppressed (0, coverage 0)
            columns[name] = 0.0
    else:
        # No per-process matrices reached us (the full analysis path, which builds each view from
        # already-aggregated per-layer maxima). Publish the columns so the wire contract is the
        # same on both paths, and mark them INCOHERENT so the planner declines to price them and
        # falls back to the metrics it has always used. Filling them with the per-domain maxima
        # would hand a decomposition that does not close on the wall to a planner that has no way
        # to tell.
        for suffix in DOMAIN_VECTOR_SUFFIXES:
            columns[f"domain_input_time_proc_{suffix}"] = input_time
            columns[f"domain_decode_time_proc_{suffix}"] = decode_time
            columns[f"domain_comm_time_proc_{suffix}"] = comm
            columns[f"domain_compute_time_proc_{suffix}"] = compute_time
            columns[f"domain_device_time_proc_{suffix}"] = device_time
            columns[f"window_wall_time_proc_{suffix}"] = wall
            columns[f"residual_time_proc_{suffix}"] = residual
        columns["domain_input_spread_time_proc"] = 0.0
        columns["domain_procs_active"] = 0.0
        columns["domain_coherent"] = 0.0
        for name in ("p1", "p2", "med", "min"):
            columns["domain_input_time_proc_%s" % name] = input_time + decode_time
        for name in SKEW_COLUMNS:          # no per-process matrices: attribution suppressed (0, coverage 0)
            columns[name] = 0.0

    additions = pd.DataFrame(columns, index=df.index).astype("Float64")
    return pd.concat([df.drop(columns=[c for c in additions if c in df.columns]), additions], axis=1)


def set_quantile_metrics(df: pd.DataFrame):
    quantile_metrics = [col for col in df.columns if col.endswith('_stats') and '_q' in col]

    if not quantile_metrics:
        return df

    new_cols: Dict[str, pd.Series] = {}

    for stats_col in quantile_metrics:
        stats = df[stats_col]

        if stats.empty:
            continue

        col_base = stats_col.replace('_stats', '')
        mean_col = f"{col_base}_mean"
        std_col = f"{col_base}_std"
        count_col = f"{col_base}_count"

        mean_series = pd.to_numeric(stats.str[0], errors='coerce').astype('Float64')
        std_series = pd.to_numeric(stats.str[1], errors='coerce').astype('Float64')
        count_series = pd.to_numeric(stats.str[2], errors='coerce').astype('Int64')

        new_cols[mean_col] = mean_series
        new_cols[std_col] = std_series
        new_cols[count_col] = count_series

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        df = df.drop(columns=quantile_metrics)

    return df.sort_index(axis=1)

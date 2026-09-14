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
            count_sum = df[count_col].sum()
            if count_sum > 0:
                new_metrics[count_frac_total_col] = df[count_col] / count_sum
            else:
                new_metrics[count_frac_total_col] = pd.Series(pd.NA, index=df.index, dtype='Float64')
        elif metric.endswith(size_proc_metric):
            size_col = metric
            size_proc_frac_total_col = metric.replace(size_proc_metric, 'size_proc_frac_total')
            size_total = df[size_col].sum()
            if size_total > 0:
                new_metrics[size_proc_frac_total_col] = df[size_col] / size_total
            else:
                new_metrics[size_proc_frac_total_col] = pd.Series(pd.NA, index=df.index, dtype='Float64')
        elif metric.endswith(time_proc_metric):
            time_col = metric
            time_proc_frac_total_col = metric.replace(time_proc_metric, 'time_proc_frac_total')
            time_total = df[time_col].sum()
            if time_total > 0:
                new_metrics[time_proc_frac_total_col] = df[time_col] / time_total
            else:
                new_metrics[time_proc_frac_total_col] = pd.Series(pd.NA, index=df.index, dtype='Float64')

    # Compute call_frac_total metrics (always from *_sum, regardless of view type)
    for metric in view_metrics:
        if metric.endswith(size_call_metric):
            size_call_col = metric
            size_call_frac_total_col = metric.replace(size_call_metric, 'size_call_frac_total')
            size_call_total = df[size_call_col].sum()
            if size_call_total > 0:
                new_metrics[size_call_frac_total_col] = df[size_call_col] / size_call_total
            else:
                new_metrics[size_call_frac_total_col] = pd.Series(pd.NA, index=df.index, dtype='Float64')
        elif metric.endswith(time_call_metric):
            time_call_col = metric
            time_call_frac_total_col = metric.replace(time_call_metric, 'time_call_frac_total')
            time_call_total = df[time_call_col].sum()
            if time_call_total > 0:
                new_metrics[time_call_frac_total_col] = df[time_call_col] / time_call_total
            else:
                new_metrics[time_call_frac_total_col] = pd.Series(pd.NA, index=df.index, dtype='Float64')

    count_time_proc_frac_metric_pairs = _find_metric_pairs(list(new_metrics.keys()), 'count_frac_total', 'time_proc_frac_total')
    for count_frac_total_col, time_proc_frac_total_col in count_time_proc_frac_metric_pairs:
        ops_percentile_col = count_frac_total_col.replace('count_frac_total', 'ops_percentile')
        ops_slope_col = count_frac_total_col.replace('count_frac_total', 'ops_slope')
        # WiSIO orientation: operation-share / time-share.  The geometric
        # 45-degree reference is tan(45)=1; values below it have a
        # disproportionate time share within this operation family.
        ops_slope = new_metrics[count_frac_total_col] / new_metrics[time_proc_frac_total_col]
        ops_slope = ops_slope.replace([np.inf, -np.inf], pd.NA)
        new_metrics[ops_percentile_col] = ops_slope.rank(pct=True)
        new_metrics[ops_slope_col] = ops_slope

    if new_metrics:
        new_metrics_df = pd.DataFrame(new_metrics, index=df.index)
        new_metrics_df = new_metrics_df.replace([np.inf, -np.inf], pd.NA).astype('Float64')
        overlapping_cols = [col for col in new_metrics_df.columns if col in df.columns]
        if overlapping_cols:
            df = df.drop(columns=overlapping_cols)
        df = pd.concat([df, new_metrics_df], axis=1)

    return df.sort_index(axis=1)


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
        for dm in derived_metrics.get(layer.lower(), {}):
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
     "domain_input_time_proc_med", "domain_input_time_proc_min")


def coherent_domain_vectors(mats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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

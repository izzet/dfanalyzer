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
        ops_slope = new_metrics[time_proc_frac_total_col] / new_metrics[count_frac_total_col]
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

    return df.sort_index(axis=1)


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

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
        df[size_col] = df[size_col].where(df[size_col] > 0, pd.NA)
        df[bw_col] = (df[size_col] / df[time_col]).where(df[size_col] > 0, pd.NA)
        df[intensity_col] = (df[count_col] / df[size_col]).where(df[size_col] > 0, pd.NA)
        new_metrics.extend([bw_col, intensity_col, time_col])

    for count_col in count_cols:
        ops_col = count_col.replace('count', 'ops')
        time_col = count_col.replace('count', 'time')
        df[ops_col] = df[count_col] / df[time_col]
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
    size_metric = 'size_sum'
    time_metric = 'time_sum' if is_view_process_based else 'time_max'

    view_metrics = list(set(df.columns.tolist()))
    new_metrics: List[str] = []

    for metric in view_metrics:
        if metric.endswith(count_metric):
            count_col = metric
            count_frac_total_col = metric.replace(count_metric, 'count_frac_total')
            count_sum = df[count_col].sum()
            df[count_frac_total_col] = df[count_col] / count_sum if count_sum > 0 else pd.NA
            new_metrics.append(count_frac_total_col)
        elif metric.endswith(size_metric):
            size_col = metric
            size_frac_total_col = metric.replace(size_metric, 'size_frac_total')
            size_sum = df[size_col].sum()
            df[size_frac_total_col] = df[size_col] / size_sum if size_sum > 0 else pd.NA
            new_metrics.append(size_frac_total_col)
        elif metric.endswith(time_metric):
            time_col = metric
            time_frac_total_col = metric.replace(time_metric, 'time_frac_total')
            time_sum = df[time_col].sum()
            df[time_frac_total_col] = df[time_col] / time_sum if time_sum > 0 else pd.NA
            new_metrics.append(time_frac_total_col)

    count_time_frac_metric_pairs = _find_metric_pairs(new_metrics, 'count_frac_total', 'time_frac_total')
    for count_frac_total_col, time_frac_total_col in count_time_frac_metric_pairs:
        ops_percentile_col = count_frac_total_col.replace('count_frac_total', 'ops_percentile')
        ops_slope_col = count_frac_total_col.replace('count_frac_total', 'ops_slope')
        ops_slope = df[time_frac_total_col] / df[count_frac_total_col]
        ops_slope = ops_slope.replace([np.inf, -np.inf], pd.NA)
        df[ops_percentile_col] = ops_slope.rank(pct=True)
        df[ops_slope_col] = ops_slope
        new_metrics.append(ops_percentile_col)
        new_metrics.append(ops_slope_col)

    df[new_metrics] = df[new_metrics].replace([np.inf, -np.inf], pd.NA).astype('Float64')

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
    time_metric = 'time_sum' if is_view_process_based else 'time_max'
    compute_time_metric = f"compute_{time_metric}"
    time_boundary_metric = f"{time_boundary_layer}_{time_metric}"

    # Collect new columns and assign them in batch to avoid fragmentation warnings
    x_layer_metrics: Dict[str, pd.Series] = {}

    # Set relational time metrics for layers
    for layer in layers:
        layer_time = df[f"{layer}_{time_metric}"]

        time_frac_boundary_col = f"{layer}_time_frac_{time_boundary_layer}"
        x_layer_metrics[time_frac_boundary_col] = layer_time / df[time_boundary_metric]

        child_layers = [child for child, parent in layer_deps.items() if parent == layer]
        if not child_layers:
            continue

        o_time_col = f"o_{layer}_{time_metric}"
        o_time_frac_boundary_col = f"o_{layer}_time_frac_{time_boundary_layer}"
        o_time_frac_self_col = f"o_{layer}_time_frac_self"
        o_time_frac_total_col = o_time_col.replace(time_metric, 'time_frac_total')

        child_time_sum = sum(df[f"{child}_{time_metric}"].fillna(0) for child in child_layers)
        o_time = np.maximum(layer_time - child_time_sum, 0)
        o_time_sum = o_time.sum()

        o_time_series = pd.array(o_time, dtype='Float64')
        x_layer_metrics[o_time_col] = o_time_series
        x_layer_metrics[o_time_frac_boundary_col] = o_time_series / df[time_boundary_metric]
        x_layer_metrics[o_time_frac_self_col] = o_time_series / layer_time
        x_layer_metrics[o_time_frac_total_col] = pd.NA
        if o_time_sum > 0:
            x_layer_metrics[o_time_frac_total_col] = o_time_series / o_time_sum

        layer_has_time = layer_time.sum() > 0
        for child_layer in child_layers:
            time_frac_parent_col = f"{child_layer}_time_frac_parent"
            x_layer_metrics[time_frac_parent_col] = pd.NA
            if layer_has_time:
                x_layer_metrics[time_frac_parent_col] = df[f"{child_layer}_{time_metric}"] / layer_time

    # Set relational time metrics for derived metrics
    for layer in derived_metrics:
        for dm in derived_metrics[layer]:
            dm_col = f"{layer}_{dm}"
            dm_time_col = f"{dm_col}_{time_metric}"

            if dm_time_col not in df.columns:
                continue

            dm_time_frac_boundary_col = f"{dm_col}_time_frac_{time_boundary_layer}"
            dm_time_frac_parent_col = f"{dm_col}_time_frac_parent"
            dm_time_frac_total_col = f"{dm_col}_time_frac_total"

            dm_time = df[dm_time_col]
            dm_time_sum = dm_time.sum()

            x_layer_metrics[dm_time_frac_boundary_col] = dm_time / df[time_boundary_metric]
            x_layer_metrics[dm_time_frac_parent_col] = dm_time / df[f"{layer}_{time_metric}"]
            x_layer_metrics[dm_time_frac_total_col] = pd.NA

            if dm_time_sum > 0:
                x_layer_metrics[dm_time_frac_total_col] = dm_time / dm_time_sum

    # Set unoverlapped times if there is compute time
    if compute_time_metric in df.columns:
        compute_time = df[compute_time_metric].fillna(0).astype('Float64')
        # Set unoverlapped time metrics
        for async_layer in async_layers:
            time_col = f"{async_layer}_{time_metric}"

            u_time_col = f"u_{time_col}"
            u_time_frac_boundary_col = f"u_{async_layer}_time_frac_{time_boundary_layer}"
            u_time_frac_self_col = f"u_{async_layer}_time_frac_self"
            u_time_frac_total_col = f"u_{async_layer}_time_frac_total"

            layer_time = df[time_col]
            u_time = (layer_time - compute_time).clip(lower=0).astype('Float64')
            u_time_sum = u_time.sum()

            u_time_series = pd.array(u_time, dtype='Float64')
            x_layer_metrics[u_time_col] = u_time_series
            x_layer_metrics[u_time_frac_self_col] = u_time_series / layer_time
            x_layer_metrics[u_time_frac_boundary_col] = u_time_series / df[time_boundary_metric]
            x_layer_metrics[u_time_frac_total_col] = pd.NA
            if u_time_sum > 0:
                x_layer_metrics[u_time_frac_total_col] = u_time_series / u_time_sum

            parent_layer = layer_deps.get(async_layer)
            if parent_layer:
                u_time_frac_parent_col = f"u_{async_layer}_time_frac_parent"
                x_layer_metrics[u_time_frac_parent_col] = u_time_series / df[f"{parent_layer}_{time_metric}"]

    if x_layer_metrics:
        df = df.copy()
        df = df.assign(**x_layer_metrics)
        x_layer_cols = list(x_layer_metrics.keys())
        df[x_layer_cols] = df[x_layer_cols].replace([np.inf, -np.inf], pd.NA).astype('Float64')

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

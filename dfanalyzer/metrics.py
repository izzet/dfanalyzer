import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .types import Layer, MetricBoundaries, Score


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


def _find_metric(metrics, suffix):
    return [m for m in metrics if m.endswith(suffix)]


def _find_metric_pairs(metrics: pd.MultiIndex, metric_type1: str, metric_type2: str, agg_type: str):
    map1 = {
        metric_name[: -len(metric_type1)]: (metric_name, agg)
        for metric_name, agg in metrics
        if metric_name.endswith(metric_type1) and agg == agg_type
    }
    map2 = {
        metric_name[: -len(metric_type2)]: (metric_name, agg)
        for metric_name, agg in metrics
        if metric_name.endswith(metric_type2) and agg == agg_type
    }
    common_prefixes = set(map1.keys()).intersection(map2.keys())
    return [(map1[prefix], map2[prefix]) for prefix in sorted(list(common_prefixes))]


def set_main_metrics(df: pd.DataFrame):
    count_cols = [col for col in df.columns if col.endswith('count')]
    size_cols = [col for col in df.columns if col.endswith('size')]

    for size_col in size_cols:
        bw_col = size_col.replace('size', 'bw')
        count_col = size_col.replace('size', 'count')
        intensity_col = size_col.replace('size', 'intensity')
        time_col = size_col.replace('size', 'time')
        df[size_col] = np.where(df[size_col] > 0, df[size_col], np.nan)
        df[bw_col] = np.where(df[size_col] > 0, df[size_col] / df[time_col], np.nan)
        df[intensity_col] = np.where(df[size_col] > 0, df[count_col] / df[size_col], np.nan)

    for count_col in count_cols:
        ops_col = count_col.replace('count', 'ops')
        time_col = count_col.replace('count', 'time')
        df[ops_col] = df[count_col] / df[time_col]

    return df.sort_index(axis=1)


def set_view_metrics(df: pd.DataFrame, is_view_process_based: bool, epsilon=1e-9):
    metrics = set(df.columns.get_level_values(0))

    std_cols = [(metric, 'std') for metric in metrics]
    min_cols = [(metric, 'min') for metric in metrics]
    max_cols = [(metric, 'max') for metric in metrics]

    for std_col, min_col, max_col in zip(std_cols, min_cols, max_cols):
        if std_col not in df.columns:
            continue
        df.loc[df[min_col] == df[max_col], std_col] = 0

    for metric in metrics:
        if metric.endswith('count') or metric.endswith('size'):
            df[(metric, 'per')] = df[(metric, 'sum')] / df[(metric, 'sum')].sum()
        elif metric.endswith('time'):
            if is_view_process_based:
                df[(metric, 'per')] = df[(metric, 'max')] / df[(metric, 'max')].sum()
            else:
                df[(metric, 'per')] = df[(metric, 'sum')] / df[(metric, 'sum')].sum()

    for count_per_col, time_per_col in _find_metric_pairs(df.columns, 'count', 'time', 'per'):
        metric, _ = count_per_col
        ops_metric = metric.replace('count', 'ops')
        ops_slope = df[time_per_col] / df[count_per_col]
        df[(ops_metric, 'pct')] = ops_slope.rank(pct=True)
        df[(ops_metric, 'slope')] = ops_slope

    return df.sort_index(axis=1)


def set_cross_layer_metrics(
    df: pd.DataFrame,
    layer_defs: Dict[Layer, str],
    layer_deps: Dict[Layer, Optional[Layer]],
    is_view_process_based: bool,
) -> pd.DataFrame:
    time_metric = 'time_sum' if is_view_process_based else 'time_max'
    compute_time_metric = f"compute_{time_metric}"

    metric_cols = []

    # Set overhead time metrics
    for layer, parent in layer_deps.items():
        if not parent:
            continue
        child_layers = [child for child, parent in layer_deps.items() if parent == layer]
        if not child_layers:
            continue
        overhead_time_col = f"{layer}_overhead_{time_metric}"
        child_times = sum(df[f"{child}_{time_metric}"].fillna(0) for child in child_layers)
        df[overhead_time_col] = np.maximum(df[f"{layer}_{time_metric}"] - child_times, 0)
        df[overhead_time_col] = df[overhead_time_col].astype('double[pyarrow]')
        metric_cols.append(overhead_time_col)

    # Set unoverlapped times if there is compute time
    if compute_time_metric in df.columns:
        # Set unoverlapped time metrics (this has to come before time percentage calc.)
        for time_col in _find_metric(df.columns, time_metric):
            if (
                time_col.startswith('u_')
                or time_col.startswith('d_')
                or f'compute_{time_metric}' in time_col
                or 'app_' in time_col
                or 'training_' in time_col
            ):
                continue
            compute_times = df[compute_time_metric].fillna(0)
            df[f"u_{time_col}"] = np.maximum(df[time_col] - compute_times, 0)
            df[f"u_{time_col}"] = df[f"u_{time_col}"].astype('double[pyarrow]')

    return df.replace([np.inf, -np.inf], np.nan).sort_index(axis=1)


def set_metric_scores(
    df: pd.DataFrame,
    metric_boundaries: MetricBoundaries,
    unscored_metrics: List[str] = [],
) -> pd.DataFrame:
    metrics = [col for col in df.columns if col not in unscored_metrics and not col.startswith('d_')]

    score_cols = {}

    for metric in metrics:
        score_col = f"{metric}_score"
        if metric.endswith('_pct') or metric.endswith('_per') or metric.endswith('_util'):
            metric_value = df[metric]
            if metric.endswith('_util'):
                metric_value = 1 - metric_value
            score_cols[score_col] = np.digitize(metric_value, bins=PERCENTAGE_BINS, right=True)
        elif metric.endswith('_slope'):
            score_cols[score_col] = np.digitize(df[metric], bins=SLOPE_BINS, right=True)
        elif metric.endswith('_intensity_mean'):
            score_cols[score_col] = np.digitize(df[metric], bins=INTENSITY_BINS, right=True)
        if score_col in score_cols:
            score_cols[score_col] = np.where(pd.isna(df[metric]), np.nan, score_cols[score_col])

    for metric in metric_boundaries:
        score_col = f"{metric}_score"
        metric_pct = df[metric] / metric_boundaries[metric]
        if 'bw_mean' in metric:
            metric_pct = 1 - metric_pct
        score_cols[score_col] = np.digitize(metric_pct, bins=PERCENTAGE_BINS, right=True)
        score_cols[score_col] = np.where(np.isnan(df[metric]), np.nan, score_cols[score_col])

    if score_cols:
        score_df = pd.DataFrame(score_cols, index=df.index)
        score_df = score_df.astype('Int64')
        df = pd.concat([df, score_df], axis=1)

    return df.sort_index(axis=1)

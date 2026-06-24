"""Metric-driven severity scoring (analyzer-side).

Produces a CONTINUOUS severity in [0,1] per metric (same unit as the rule
engine's ``severity_score``), rather than discrete 1..5 bands — the band label
is derived from the score via the shared ``_severity_label`` thresholds.

The metric-driven *detection* signal is the SLOPE (WISIO's promise):
``ops_slope = time_share / op_share`` — an entity whose share of time is
disproportionate to its share of operations is slow per-op, i.e. a true
bottleneck (vs. merely busy). ``normalize_slope`` maps that slope onto [0,1] as
the continuous form of WISIO's angle-based slope bands. ``score_metrics``
annotates a view with ``{metric}_score`` columns (used for persisted scored
detail views); the metric fact builder uses ``normalize_slope`` on ``*_slope``.
"""
import numpy as np
import pandas as pd


# Slope severity ramp, centered on the PROPORTIONAL baseline. slope == 1 (45deg,
# time-share == op-share) is NOT a bottleneck (just busy) -> severity 0; only a
# disproportionate slope > 1 (slow per op) ramps up, reaching 1.0 as slope -> inf
# (90deg). This differs from WISIO's tan(15)..tan(75) display banding: we emit
# facts, so the proportional case must score 0, not mid-scale.
SLOPE_LOW_DEG = 45.0
SLOPE_HIGH_DEG = 90.0

# Intensity geometric range (kept for reference; intensity scoring is TODO).
INTENSITY_MIN = 1 / 1024
INTENSITY_MAX = 1 / 1024**3


def _clip01(value):
    if isinstance(value, pd.Series):
        return value.clip(lower=0.0, upper=1.0)
    return float(np.clip(value, 0.0, 1.0))


def normalize_slope(slope):
    """Map an ops-slope (time_share / op_share) onto a [0,1] bottleneck severity.

    ``(atan(slope)_deg - 45) / (90 - 45)`` clipped to [0,1]: slope <= 1
    (proportional or better — not a bottleneck) -> 0; slope > 1 ramps up as the
    entity grows disproportionately slow per op, approaching 1 as slope -> inf.
    NaN slopes pass through as NaN.
    """
    angle_deg = np.degrees(np.arctan(slope))
    sev = (angle_deg - SLOPE_LOW_DEG) / (SLOPE_HIGH_DEG - SLOPE_LOW_DEG)
    return _clip01(sev)


def score_metrics(df: pd.DataFrame, metric_boundaries: dict) -> pd.DataFrame:
    """Annotate ``df`` with continuous ``{metric}_score`` columns in [0,1]."""
    metrics = [col for col in df.columns if not col.startswith('d_')]

    df = df.copy()
    df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')

    score_cols = {}

    for metric in metrics:
        score_col = f"{metric}_score"
        if metric.endswith('_pct') or metric.endswith('_per'):
            score_cols[score_col] = _clip01(df[metric])
        elif metric.endswith('_util'):
            score_cols[score_col] = _clip01(1 - df[metric])
        elif metric.endswith('_slope'):
            score_cols[score_col] = normalize_slope(df[metric])
        # intensity scoring intentionally deferred (slope is the detection signal)

    for metric in metric_boundaries:
        score_col = f"{metric}_score"
        ratio = df[metric] / metric_boundaries[metric]
        if 'bw_mean' in metric:
            ratio = 1 - ratio
        score_cols[score_col] = _clip01(ratio)

    if score_cols:
        score_df = pd.DataFrame(score_cols, index=df.index).astype('float64')
        df = pd.concat([df, score_df], axis=1)

    return df.sort_index(axis=1)

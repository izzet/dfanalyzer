"""Measured zero remains evidence while undefined WiSIO slopes emit no facts."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import dftracer

source = Path(__file__).resolve().parents[1] / 'python' / 'dftracer'
if str(source) not in dftracer.__path__:
    dftracer.__path__.insert(0, str(source))
from dftracer.analyzer.metrics import set_view_metrics
from dftracer.analyzer.fact_engine import FactEngine
from dftracer.analyzer.fact_rules import build_fact_rule


@pytest.mark.parametrize('process_based', [True, False])
@pytest.mark.parametrize('dtype', ['object', 'Float64'])
@pytest.mark.parametrize('times', [[0., 2., pd.NA], [0., 0., pd.NA],
                                   [pd.NA, pd.NA, pd.NA], [0., np.inf, pd.NA]])
def test_measured_zero_family_has_undefined_slopes_without_findings(process_based, dtype, times):
    time_col = 'stall_decode_time_sum' if process_based else 'stall_decode_time_proc_max'
    frame = pd.DataFrame({
        'stall_decode_count_sum': pd.Series([1., 1., pd.NA], dtype=dtype),
        time_col: pd.Series(times, dtype=dtype),
    })
    out = set_view_metrics(frame, metric_boundaries={}, is_view_process_based=process_based)
    # Presence is not erased to avoid the division: the source zero survives.
    if pd.notna(times[0]):
        assert out.loc[0, time_col] == 0.
    assert pd.isna(out.loc[0, 'stall_decode_ops_slope'])
    assert pd.isna(out.loc[0, 'stall_decode_ops_percentile'])
    assert pd.isna(out.loc[2, 'stall_decode_ops_slope'])
    if pd.notna(times[1]) and times[1] == 2.:
        assert out.loc[1, 'stall_decode_ops_slope'] == pytest.approx(.5)
    else:
        assert out.stall_decode_ops_slope.isna().all()
        assert out.stall_decode_time_proc_frac_total.isna().all()
    rule = build_fact_rule({
        'id': 'test.undefined.slope', 'priority': 1, 'source_view': 'proc_name',
        'fact_type': 'zero_normalization_control',
        'required_metrics': ['stall_decode_ops_slope'],
        'when': 'stall_decode_ops_slope < 1.0', 'severity_score': '0.5',
    })
    # Each undefined row is evaluated through the real fact engine separately,
    # so a valid peer slope cannot mask a finding on a zero-duration row.
    for row in out.index[out.stall_decode_ops_slope.isna()]:
        facts = FactEngine([rule]).evaluate({('proc_name',): out.loc[[row]]}, raw_stats={})
        assert not facts.get(('proc_name',), [])


@pytest.mark.parametrize('counts', [[0., 0.], [pd.NA, pd.NA]])
def test_missing_or_zero_count_normalizer_also_suppresses_slope(counts):
    frame = pd.DataFrame({
        'read_count_sum': pd.Series(counts, dtype=object),
        'read_time_sum': pd.Series([1., 2.], dtype=object),
    })
    out = set_view_metrics(frame, metric_boundaries={}, is_view_process_based=True)
    assert out.read_ops_slope.isna().all()
    assert out.read_ops_percentile.isna().all()

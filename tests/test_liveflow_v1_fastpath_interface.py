"""Declared-family rules use their shared evaluator on both analysis paths."""
import sys
from pathlib import Path
from types import SimpleNamespace
import dftracer
source=Path(__file__).resolve().parents[1]/'python'/'dftracer'
if str(source) not in dftracer.__path__:dftracer.__path__.insert(0,str(source))
from dftracer.analyzer.config import AnalyzerPresetConfigLiveFlowV1
from dftracer.analyzer.fact_engine import FactEngine
from dftracer.analyzer.fastpath import FastPathPlan
from dftracer.analyzer.liveflow_v1 import declaration,PREPARED_WINDOW_METRICS


def fixture():
    preset=AnalyzerPresetConfigLiveFlowV1()
    facts=FactEngine.from_rule_file('liveflow-v1',strict_time_semantics=True,allow_mixed_time_aggregates=False)
    facts.liveflow_declaration=dict(declaration('dlio-pytorch'),input_families=['storage','decode'],split_wait=True,hdf5_decode_residual=True)
    return preset,facts


def test_declared_decode_rules_accept_shared_prepared_window_interface():
    preset,facts=fixture()
    assert FastPathPlan.build(preset,['window','proc_name'],list(preset.layer_defs),facts) is not None
    assert 'lf_v1_decode_wait' in PREPARED_WINDOW_METRICS
    assert 'reader_decode_ops_slope' in PREPARED_WINDOW_METRICS


def test_prepared_metrics_are_not_advertised_without_evaluator_declaration():
    preset,facts=fixture()
    del facts.liveflow_declaration
    assert FastPathPlan.build(preset,['window','proc_name'],list(preset.layer_defs),facts) is None


def test_unimplemented_metrics_still_decline():
    preset,facts=fixture()
    facts.rules=[SimpleNamespace(id='unsupported',source_view='window',rank_thread_only=False,worker_only=False,referenced_columns={'lf_v1_decode_wait','invented_decode_quantile'},derived_metrics={})]
    assert FastPathPlan.build(preset,['window','proc_name'],list(preset.layer_defs),facts) is None


def test_prepared_metrics_are_not_advertised_on_other_views():
    preset,facts=fixture()
    facts.rules=[SimpleNamespace(id='other-view',source_view='proc_name',rank_thread_only=False,worker_only=False,referenced_columns={'lf_v1_decode_wait'},derived_metrics={})]
    assert FastPathPlan.build(preset,['window','proc_name'],list(preset.layer_defs),facts) is None

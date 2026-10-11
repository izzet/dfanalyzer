"""Measured zero/absence survives both liveflow-v1 view paths and validity."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import dftracer
source = Path(__file__).resolve().parents[1] / 'python' / 'dftracer'
if str(source) not in dftracer.__path__:
    dftracer.__path__.insert(0, str(source))
from dftracer.analyzer.fastpath import FastPathPlan, preserve_measured_stall_zeros
from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.liveflow_v1 import declaration, prepare_views


def plan(name='liveflow-v1'):
    p = FastPathPlan.__new__(FastPathPlan)
    p.preset = SimpleNamespace(name=name)
    p.view_types = ['window', 'proc_name']
    p.terms = [(f, 'func_name == "' + f + '"', False)
               for f in ('stall_storage', 'stall_decode', 'compute')]
    p._mask_cache = {}
    return p


def test_fastpath_group_presence_and_legacy_scope():
    rows = []
    for window, pid, fn, duration in [(1,10,'stall_storage',0),
        (1,10,'stall_decode',0),(1,20,'stall_storage',2),
        (1,20,'compute',4),(2,10,'compute',3)]:
        rows.append(dict(window=window,pid=pid,proc_name=f'app#host#{pid}#1',
            func_name=fn,cat='stall',io_cat='',time=duration,size=0))
    events = pd.DataFrame(rows)
    views = plan().flat_views(events)
    w = views[('window',)]
    assert w.loc[1,'stall_storage_time_proc_max'] == 2
    assert w.loc[1,'stall_decode_time_proc_max'] == 0
    assert pd.isna(w.loc[2,'stall_storage_time_proc_max'])
    p = views[('proc_name',)]
    assert p.loc['app#host#10#1','stall_storage_time_sum'] == 0
    assert pd.isna(p.loc['app#host#20#1','stall_decode_time_sum'])
    assert pd.isna(plan('halo3d').flat_views(events)[('window',)].loc[1,'stall_decode_time_proc_max'])


def test_fullpath_process_flat_view_restores_only_observed_time(monkeypatch):
    # Feed the actual full-path publication method the aggregate representation
    # produced by the legacy groupby/zero replacement. Other metric transforms
    # are independent of this presence contract and kept identity here.
    import dftracer.analyzer.analyzer as module
    monkeypatch.setattr(module, 'set_view_metrics', lambda f, **k:f)
    monkeypatch.setattr(module, 'set_cross_layer_metrics', lambda f, **k:f)
    a = Analyzer.__new__(type('Concrete', (Analyzer,), {'read_trace':lambda *a:None}))
    a.preset = SimpleNamespace(name='liveflow-v1',async_layers=[],derived_metrics={},layer_deps={},time_boundary_layer=None)
    a.layers=[]
    a._set_additional_metrics=lambda f, **k:f
    frame=pd.DataFrame({'stall_storage_count_sum':[2,np.nan,1],
        'stall_storage_time_proc_max':[np.nan,np.nan,3.],
        'stall_storage_time_sum':[np.nan,np.nan,3.],
        'stall_storage_ops_slope':[np.nan,np.nan,1.]})
    result=a._process_flat_view(frame,('window',),{'window':{}})
    assert result.stall_storage_time_proc_max.iloc[0] == 0
    assert pd.isna(result.stall_storage_time_proc_max.iloc[1])
    assert result.stall_storage_time_proc_max.iloc[2] == 3
    assert pd.isna(result.stall_storage_ops_slope.iloc[0])


@pytest.mark.parametrize('legacy_null', [True, False])
def test_zero_stall_window_remains_valid(legacy_null):
    value=np.nan if legacy_null else 0.
    frame=pd.DataFrame([dict(step_nunique=3,compute_time_proc_max=1,
        comm_time_proc_max=1,fetch_iter_time_proc_max=1,
        stall_storage_time_proc_max=value,stall_storage_count_sum=2,
        stall_decode_time_proc_max=value,stall_decode_count_sum=2)])
    context=dict(native_window=1,trace_integrity={'lf_v1_trace_complete':1.})
    row=prepare_views({('window',):frame},declaration('halo3d'),context)[('window',)].iloc[0]
    assert row.lf_v1_valid == 1
    assert row.lf_v1_storage_wait == 0 and row.lf_v1_decode_wait == 0
    context['trace_integrity']['lf_v1_trace_complete']=0.
    assert prepare_views({('window',):frame},declaration('halo3d'),context)[('window',)].iloc[0].lf_v1_valid == 0


def test_absent_and_bad_counts_are_not_filled():
    frame=pd.DataFrame({'stall_decode_count_sum':[np.nan,0,-1,np.inf],
                        'stall_decode_time_proc_max':[np.nan]*4})
    assert preserve_measured_stall_zeros(frame).stall_decode_time_proc_max.isna().all()


def test_intermediate_zero_preservation_prevents_mixed_group_mean_bias():
    from dftracer.analyzer.analyzer import _replace_zeros_preserving_stall_time
    frame=pd.DataFrame({'count':[1,1,1], 'time':[0.,2.,0.],
        'time_sq':[0.,4.,0.], 'time_call_min':[0.,2.,0.]},
        index=pd.MultiIndex.from_tuples([('stall.storage',10),('stall.storage',20),
                                        ('compute',10)],names=['func_name','pid']))
    actual=_replace_zeros_preserving_stall_time(frame,preserve=True,by_function=True)
    assert actual.loc['stall.storage','time'].mean()==1.
    assert actual.loc['stall.storage','time'].min()==0.
    assert pd.isna(actual.loc[('compute',10),'time'])
    assert pd.isna(_replace_zeros_preserving_stall_time(frame).loc[('stall.storage',10),'time'])


def test_intermediate_missing_duration_is_not_measured_zero():
    from dftracer.analyzer.analyzer import _replace_zeros_preserving_stall_time
    frame=pd.DataFrame({'count':[1,1],'time_sum':[np.nan,0.]})
    result=_replace_zeros_preserving_stall_time(frame,preserve=True)
    assert pd.isna(result.time_sum.iloc[0]) and result.time_sum.iloc[1]==0.


@pytest.fixture
def aggregation_client():
    from distributed import Client
    with Client(processes=False,n_workers=1,threads_per_worker=2,dashboard_address=None) as client:
        yield client


@pytest.mark.parametrize('dask_input',[False,True])
def test_full_aggregation_zero_and_nonzero_processes(monkeypatch,dask_input,aggregation_client):
    import dask.dataframe as dd
    from dftracer.analyzer.config import AnalyzerPresetConfigLiveFlowV1
    monkeypatch.setenv('LIVEFLOW_WORKLOAD','halo3d')
    cls=type('Concrete', (Analyzer,), {'read_trace':lambda *a:None})
    a=cls(preset=AnalyzerPresetConfigLiveFlowV1(),checkpoint=False,checkpoint_dir='',
          time_granularity=1.,time_resolution=1.)
    events=pd.DataFrame([dict(window=1,proc_name=f'app#host#{pid}#1',cat='stall',
        io_cat='',acc_pat='',func_name='stall.storage',host_hash='host',
        time=duration,count=1,size=0,file_name='f',time_range=0)
        for pid,duration in [(10,0.),(20,2.)]])
    data=dd.from_pandas(events,npartitions=2) if dask_input else events
    hlm=a._compute_high_level_metrics(data,['window','proc_name'],'128MB')
    main=a._compute_main_view('stall_storage',hlm,['window','proc_name'],'128MB')
    view=a._compute_view('stall_storage',main,('window',),'window',['window','proc_name'])
    if dask_input: view=view.compute()
    assert view.loc[1,'time_proc_min']==0.
    assert view.loc[1,'time_proc_mean']==1.
    assert view.loc[1,'time_proc_max']==2.

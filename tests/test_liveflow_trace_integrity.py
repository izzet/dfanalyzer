import sys
from pathlib import Path
import pandas as pd
import dftracer
source = Path(__file__).resolve().parents[1] / 'python' / 'dftracer'
dftracer.__path__.insert(0, str(source))
from dftracer.analyzer.liveflow_v1 import window_trace_integrity, prepare_views, declaration


def traces():
    return pd.DataFrame([dict(pid=p, proc_name=f'app#host#{p}#{p}', step=s, cat='compute')
                         for p in (10, 20) for s in (100, 101, 102)])


def check(rows, drops=0, series='v1;first=100;n=3;b=1;wall=1,1,1'):
    return window_trace_integrity(rows, {10,20}, dict(step_series=series, completed_steps=102), drops)


def test_complete_and_binned_header():
    assert check(traces())['lf_v1_trace_complete'] == 1
    assert check(traces(), series='v1;first=100;n=3;b=2;wall=2,1')['lf_v1_trace_complete'] == 1


def test_missing_rank_step_hidden_by_other_rank():
    rows = traces()
    assert check(rows.drop(rows[(rows.pid==20)&(rows.step==101)].index))['lf_v1_trace_complete'] == 0


def test_worker_cannot_fill_main_thread_gap():
    rows=traces(); rows.loc[4,'proc_name']='app#host#20#99'
    assert check(rows)['lf_v1_trace_complete'] == 0


def test_complete_main_steps_with_worker_ring_loss():
    assert check(traces(), 1)['lf_v1_trace_complete'] == 0
    assert check(traces(), -1)['lf_v1_trace_complete'] == 0


def test_absent_or_malformed_boundary():
    assert check(traces(), series='bad')['lf_v1_trace_complete'] == 0
    assert window_trace_integrity(traces(),set(),{})['lf_v1_trace_complete'] == 0


def test_invalid_integrity_overrides_available_metrics():
    frame=pd.DataFrame([dict(step_nunique=3,compute_time_proc_max=1,comm_time_proc_max=1,
        fetch_iter_time_proc_max=1,stall_storage_time_proc_max=1,stall_storage_count_sum=1,
        stall_decode_time_proc_max=1,stall_decode_count_sum=1)])
    context=dict(native_window=1,trace_integrity=check(traces(),1))
    assert prepare_views({('window',):frame},declaration('halo3d'),context)[('window',)].iloc[0].lf_v1_valid == 0


def test_all_main_rows_missing_cannot_use_helper_threads():
    rows=traces(); rows.loc[rows.pid==20,'proc_name']='app#host#20#99'
    assert check(rows)['lf_v1_trace_complete'] == 0


def test_epoch_local_coverage_uses_control_count_without_guessed_offset():
    rows=traces(); rows['step']-=99
    evidence=dict(step_series='v1;first=100;n=3;b=1;wall=1,1,1',completed_steps=102)
    assert window_trace_integrity(rows,{10,20},evidence,0,global_steps=False)['lf_v1_trace_complete']==1
    assert window_trace_integrity(rows.drop(4),{10,20},evidence,0,global_steps=False)['lf_v1_trace_complete']==0
    rows.loc[rows.pid==20,'step']+=1
    assert window_trace_integrity(rows,{10,20},evidence,0,global_steps=False)['lf_v1_trace_complete']==0

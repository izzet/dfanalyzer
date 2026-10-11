"""Keep optimization ordering, expression semantics, and publication contracts."""
import collections,random
import pandas as pd
from dftracer.analyzer.analyzer import _PendingTraceEvents
from dftracer.analyzer.fact_engine import _eval_expr


def test_heap_matches_original_selector_across_counts_and_insertions():
    randomizer=random.Random(419)
    old=_PendingTraceEvents();new=_PendingTraceEvents()
    seen=collections.defaultdict(int)
    for window in range(20):
        targets={pid:seen[pid]+randomizer.randrange(10,30) for pid in range(8)}
        boundary=window*100+100
        for batch in range(5):
            for _ in range(100):
                pid=randomizer.randrange(40)
                item=({},randomizer.randrange(boundary-80,boundary+100),pid)
                old.append(item);new.append(item)
            while True:
                expected=old.pop_first_under_target(
                    lambda p:p in targets and seen[p]>=targets[p],
                    lambda e:e[2] not in targets and bool(boundary and e[1] and e[1]>boundary))
                actual=new.pop_for_window(targets,seen,boundary)
                assert actual==expected
                if actual is None:break
                seen[actual[2]]+=1
            assert list(new)==list(old)
        if window%3==0 and old:
            assert new.popleft()==old.popleft()
            item=({},boundary-10,None);new.appendleft(item);old.appendleft(item)
    assert list(new)==list(old)


def test_expression_projection_preserves_nullable_math_and_index():
    frame=pd.DataFrame({'a':pd.Series([0,2,None],dtype='Float64'), 'b':[1.,3.,4.],
                        **{f'unused_{i}':[float(i)]*3 for i in range(800)}})
    from dftracer.analyzer.fact_engine import _eval_python_expr
    for expression in ('fillna0(a) / b','max(a, b)','a > 1','(a + b) / 2','True'):
        actual=_eval_expr(frame,expression)
        # Same pandas engines on the entire frame, or original fallback.
        try:expected=frame.eval(expression,engine='numexpr')
        except Exception:
            try:expected=frame.eval(expression,engine='python')
            except Exception:expected=_eval_python_expr(frame,expression)
        if not isinstance(expected,pd.Series):expected=pd.Series(expected,index=frame.index)
        pd.testing.assert_series_equal(actual,expected)


def test_compact_native_publication_keeps_fact_payload_and_eos(monkeypatch):
    from types import SimpleNamespace
    from dftracer.analyzer.output import MofkaOutput
    from liveflow_transport import queue_io
    queue_io.reset();monkeypatch.setenv('LIVEFLOW_TRANSPORT','liveflow')
    out=MofkaOutput('', 'agentperf_facts')
    result=SimpleNamespace(analysis_facts={'window':[1]},flat_views={('window',):pd.DataFrame({'x':[2.]})},window_index=3,layers=[],raw_stats={},view_types=['window'])
    monkeypatch.setattr(out,'_send_fact_envelope',lambda result:out._producer.push(metadata={'artifact_type':'analysis_facts'},data=b'exact-facts'))
    consumer=queue_io.Consumer('agentperf_facts','test')
    monkeypatch.setenv('LIVEFLOW_AGENT_FACTS_ONLY','1')
    out.handle_result(result)
    assert consumer.pull().wait(0).data==b'exact-facts'
    assert consumer.pull().wait(0) is None
    # Diagnostic views still exist on the result; legacy emission remains opt-in default.
    assert len(result.flat_views)==1
    monkeypatch.delenv('LIVEFLOW_AGENT_FACTS_ONLY')
    out.handle_result(result)
    assert consumer.pull().wait(0).data==b'exact-facts'
    assert consumer.pull().wait(0).metadata['artifact_type']=='flat_view'
    out.finish_stream()
    assert consumer.pull().wait(0).metadata['name']=='end'


def test_bulk_sweep_matches_legacy_with_unordered_worker_timestamps():
    r=random.Random(73)
    old=_PendingTraceEvents();new=_PendingTraceEvents()
    seen_old=collections.defaultdict(int);seen_new=collections.defaultdict(int)
    for window in range(8):
        for _ in range(2000):
            pid=r.choice([None]+list(range(40)))
            item=({},r.randrange(0,1000),pid)
            old.append(item);new.append(item)
        targets={pid:seen_old[pid]+r.randrange(0,20) for pid in range(8)}
        boundary=window*120;carried=[];selected=[]
        while old:
            item=old.popleft();_,ts,pid=item
            if (pid in targets and seen_old[pid]>=targets[pid]) or (boundary and ts and ts>boundary):carried.append(item)
            else:
                selected.append(item)
                if pid is not None:seen_old[pid]+=1
        old.extend(carried)
        actual=new.sweep_for_window(targets,seen_new,boundary)
        for _,_,pid in actual:
            if pid is not None:seen_new[pid]+=1
        assert actual==selected
        assert all(seen_new.get(pid,0)==seen_old.get(pid,0) for pid in set(seen_new)|set(seen_old))
        assert list(new)==list(old)


def test_idle_ring_vector_check_finds_late_high_lane(monkeypatch):
    import time,uuid
    from pathlib import Path
    from dftracer.analyzer.streaming.continuous import BufferedTraceConsumer
    from liveflow_transport import shm_io
    from shm_reader import ShmRingReader,ShmRingWriter
    name='agentperf-test-'+uuid.uuid4().hex
    reader=ShmRingReader(name,lane_count=2048,lane_bytes=32768,create=True)
    monkeypatch.setattr(shm_io,'_seg_name',lambda topic:name)
    monkeypatch.setattr(shm_io,'_receipt_journal',lambda:None)
    monkeypatch.setattr(shm_io,'_trace_persistor',None)
    source=shm_io.Consumer('trace','test');count=[0];original=source._fill
    def fill(*args,**kwargs):count[0]+=1;return original(*args,**kwargs)
    source._fill=fill
    buffered=BufferedTraceConsumer(source);buffered.start();writer=None
    try:
        time.sleep(.02)
        assert count[0]==0
        writer=ShmRingWriter(name,2047)
        assert writer.write(shm_io.pack({'type':'batch'},b'late-lane'))
        event=buffered.pull().wait(2000)
        assert event is not None and event.data==b'late-lane'
        assert count[0]>=1
        snapshot=buffered.snapshot_committed()
        assert snapshot.remaining==0
        snapshot.close()
        assert all(s.dropped==0 and s.head==s.tail for s in reader.stats())
    finally:
        buffered.close()
        if writer:writer.close()
        source._r.close();reader.close();(Path('/dev/shm')/name).unlink()

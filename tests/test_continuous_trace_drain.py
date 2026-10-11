"""Real SHM records exceed ring capacity before the first control boundary."""
import sys, time, uuid
from pathlib import Path
import pytest
import dftracer
source=Path(__file__).resolve().parents[1]/'python'/'dftracer'
dftracer.__path__.insert(0,str(source))
from dftracer.analyzer.streaming.continuous import BufferedTraceConsumer
from shm_reader import ShmRingReader, ShmRingWriter
from liveflow_transport import shm_io


def eventually(predicate):
    end=time.monotonic()+5
    while not predicate():
        assert time.monotonic()<end,'continuous reader did not drain'
        time.sleep(.001)


@pytest.fixture
def consumer(monkeypatch):
    name='lf-continuous-'+uuid.uuid4().hex
    reader=ShmRingReader(name,lane_count=2,lane_bytes=32768,create=True)
    monkeypatch.setattr(shm_io,'_seg_name',lambda topic:name)
    monkeypatch.setattr(shm_io,'_receipt_journal',lambda:None)
    monkeypatch.setattr(shm_io,'_trace_persistor',None)
    raw=shm_io.Consumer('trace','continuous-test')
    buffered=BufferedTraceConsumer(raw);buffered.start()
    writer=ShmRingWriter(name,0)
    try:
        yield reader,writer,buffered
    finally:
        buffered.close();writer.close();raw._r.close();reader.close()
        (Path('/dev/shm')/name).unlink()


def write(writer,n):
    assert writer.write(shm_io.pack({'type':'batch','ordinal':n},b'x'*2048))


def empty_ring(reader):
    return all(s.head==s.tail for s in reader.stats())


def test_drains_before_first_control_and_during_analysis_without_loss(consumer):
    reader,writer,buffer=consumer
    # No application pull/snapshot/control boundary for 320KiB, ten ring capacities.
    for block in range(20):
        for i in range(8):write(writer,block*8+i)
        eventually(lambda:empty_ring(reader))
    assert reader.totals()==(160,0)
    first=buffer.snapshot_committed()
    assert first.remaining==160
    # While first window's analysis/snapshot consumer is idle, keep draining the
    # next window. Its records cannot extend the first snapshot's finite prefix.
    for block in range(20,40):
        for i in range(8):write(writer,block*8+i)
        eventually(lambda:empty_ring(reader))
    assert first.remaining==160
    assert [first.next().metadata['ordinal'] for _ in range(160)]==list(range(160))
    assert first.next() is None
    second=buffer.snapshot_committed()
    assert [second.next().metadata['ordinal'] for _ in range(second.remaining)]==list(range(160,320))
    assert reader.totals()==(320,0)
    assert buffer.pull().wait(0) is None


def test_snapshot_close_preserves_unconsumed_prefix(consumer):
    reader,writer,buffer=consumer
    for n in range(4):write(writer,n)
    eventually(lambda:empty_ring(reader))
    snapshot=buffer.snapshot_committed()
    assert snapshot.next().metadata['ordinal']==0
    write(writer,4);eventually(lambda:empty_ring(reader))
    with pytest.raises(RuntimeError,match='active committed snapshot'):
        buffer.pull().wait(0)
    snapshot.close()
    assert [buffer.pull().wait(1000).metadata['ordinal'] for _ in range(4)]==[1,2,3,4]


def test_background_failure_is_not_silent():
    class Failed:
        def pull(self):return self
        def wait(self,timeout_ms):raise ValueError('reader failure')
    buffered=BufferedTraceConsumer(Failed());buffered.start()
    with pytest.raises(RuntimeError,match='continuous trace drain failed'):
        buffered.pull().wait(1000)
    with pytest.raises(RuntimeError,match='continuous trace drain failed'):
        buffered.close()
    assert not buffered._thread.is_alive()

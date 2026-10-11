"""Drain SHM trace records independently of control-window analysis.

Only transport records are buffered here. Rank counts, timestamp fences,
normalization and window membership remain owned by the existing analyzer.
"""
from collections import deque
from threading import Condition, Event, Thread
import time

import numpy as np


class _Future:
    def __init__(self, consumer):
        self.consumer = consumer

    def wait(self, timeout_ms=1000):
        return self.consumer._next(timeout_ms)


class _Snapshot:
    def __init__(self, consumer, records):
        self.consumer = consumer
        self.records = records

    @property
    def remaining(self):
        return len(self.records)

    def next(self):
        with self.consumer._condition:
            self.consumer._check_error()
            if not self.records:
                return None
            record = self.records.popleft()
            if not self.records:
                self.consumer._active_snapshot = None
            return record

    def close(self):
        with self.consumer._condition:
            # Unconsumed snapshot records precede records received after it.
            self.consumer._buffer.extendleft(reversed(self.records))
            self.records.clear()
            if self.consumer._active_snapshot is self:
                self.consumer._active_snapshot = None
            self.consumer._condition.notify_all()


class BufferedTraceConsumer:
    """One serialized ring reader feeding the agent's private record buffer."""
    def __init__(self, source):
        self.source = source
        self._condition = Condition()
        self._buffer = deque()
        self._active_snapshot = None
        self._stop = Event()
        self._error = None
        self._ring_heads = self._ring_tails = None
        # Only SHM consumers expose this ring layout. A vectorized committed
        # head/tail check avoids thousands of empty Python lane generators on
        # every background poll. It never reads payloads or advances tails;
        # the source remains the sole serialized reader.
        reader = getattr(source, "_r", None)
        if reader is not None and source.__class__.__module__.endswith("shm_io"):
            from shm_reader import SEG_HEADER_SIZE, LANE_SIZE, _OFF_HEAD, _OFF_TAIL
            self._ring_heads = np.ndarray(
                (reader.lane_count,), dtype="<u8", buffer=reader._mm,
                offset=SEG_HEADER_SIZE + _OFF_HEAD, strides=(LANE_SIZE,))
            self._ring_tails = np.ndarray(
                (reader.lane_count,), dtype="<u8", buffer=reader._mm,
                offset=SEG_HEADER_SIZE + _OFF_TAIL, strides=(LANE_SIZE,))
        self._thread = Thread(target=self._drain, name='liveflow-trace-drain', daemon=True)

    def start(self):
        self._thread.start()

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError('continuous trace drain failed') from self._error

    def _drain(self):
        try:
            while not self._stop.is_set():
                # Serialize source reads with boundary snapshots. Nonblocking
                # pulls avoid holding this lock while awaiting producer data.
                with self._condition:
                    buffered = getattr(self.source, "_buf", ())
                    idle = (self._ring_heads is not None and not buffered
                            and not np.any(self._ring_heads.copy() > self._ring_tails))
                    record = None if idle else self.source.pull().wait(timeout_ms=0)
                    if record is not None:
                        self._buffer.append(record)
                        self._condition.notify_all()
                if record is None:
                    self._stop.wait(0.001)
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def pull(self):
        return _Future(self)

    def _next(self, timeout_ms):
        deadline = time.monotonic() + max(0, timeout_ms) / 1000
        with self._condition:
            if self._active_snapshot is not None:
                raise RuntimeError('ordinary pull during active committed snapshot')
            while True:
                self._check_error()
                if self._buffer:
                    return self._buffer.popleft()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set():
                    return None
                self._condition.wait(remaining)

    def snapshot_committed(self):
        with self._condition:
            self._check_error()
            if self._active_snapshot is not None:
                raise RuntimeError('another committed snapshot is active')
            # Capture the same finite per-lane committed prefix as before.
            # Include transport records already evacuated before the boundary.
            source_snapshot = self.source.snapshot_committed()
            try:
                while source_snapshot.remaining:
                    self._buffer.append(source_snapshot.next())
            finally:
                source_snapshot.close()
            snapshot = _Snapshot(self, self._buffer)
            self._buffer = deque()
            if snapshot.remaining:
                self._active_snapshot = snapshot
            return snapshot

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.ident is not None:
            self._thread.join()
        self._check_error()

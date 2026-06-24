"""Streaming driver tests: analyze_stream windows a chrome-event stream and dispatches
each completed window with a monotonic window_index. The per-window analysis itself
(_analyze_window_events) is exercised by the e2e/round-trip path; here we stub it to
test the driver logic (WindowBuffer integration + window_index sequencing) without a
dask cluster, so the test is fast and dependency-light."""
import types

import pytest

from dftracer.analyzer.analyzer import Analyzer


class _StubAnalyzer(Analyzer):
    """Minimal concrete Analyzer that records windows instead of analyzing them."""

    def __init__(self):
        self.layers = []
        self.recorded = []

    def read_trace(self, *a, **k):  # abstract on the base
        raise NotImplementedError

    def _analyze_window_events(self, events, window_index, view_types, **kw):
        self.recorded.append((window_index, len(events)))
        return types.SimpleNamespace(analysis_facts={}, window_index=window_index, flat_views={})


def _stream(pid=1):
    def marker(name):
        return {"name": name, "cat": "pipeline", "pid": pid, "tid": pid, "ph": "X", "ts": 0, "dur": 1}

    io = [{"name": "read", "cat": "posix", "pid": pid, "tid": pid, "ph": "X", "ts": i, "dur": 1}
          for i in range(3)]
    return ([marker("epoch.start")] + io + [marker("epoch.block")]
            + [marker("epoch.start")] + io + [marker("epoch.block")])


def test_analyze_stream_windows_with_monotonic_index():
    events = _stream()
    it = iter(events)
    a = _StubAnalyzer()
    out = []
    results = a.analyze_stream(
        pull_event=lambda: next(it, None),
        view_types=["time_range"],
        output_handler=out.append,
    )
    # two windows, dispatched with window_index 0 then 1; each carries its events
    assert [w for w, _ in a.recorded] == [0, 1]
    assert all(n >= 3 for _, n in a.recorded)            # io events buffered into each window
    assert [r.window_index for r in results] == [0, 1]
    assert len(out) == 2                                 # output_handler called per window


def test_analyze_stream_respects_max_windows():
    events = _stream()
    it = iter(events)
    a = _StubAnalyzer()
    a.analyze_stream(pull_event=lambda: next(it, None), view_types=["time_range"], max_windows=1)
    assert [w for w, _ in a.recorded] == [0]             # stopped after one window


def test_analyze_stream_rejects_non_dict_event():
    a = _StubAnalyzer()
    with pytest.raises(ValueError):
        a.analyze_stream(pull_event=lambda: 123, view_types=["time_range"])

import pytest

from dftracer.analyzer.streaming.window_buffer import WindowTracker, WindowBuffer


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _push_all(buffer: WindowBuffer, events):
    emitted = []
    for event in events:
        out = buffer.push(event)
        if out:
            emitted.append(out)
    return emitted


def test_epoch_buffer_buffers_until_epoch_block():
    buffer = WindowBuffer()
    emitted = _push_all(
        buffer,
        [
            {"name": "data", "value": 0, "pid": 1},
            {"name": "epoch.start", "pid": 1},
            {"name": "data", "value": 1, "pid": 1},
            {"name": "epoch.block", "pid": 1},
        ],
    )

    assert len(emitted) == 1
    buf = emitted[0]
    assert isinstance(buf, list)
    assert len(buf) == 3
    assert all(item.get("epoch") == 1 for item in buf)


def test_epoch_buffer_multiple_epochs():
    buffer = WindowBuffer()
    emitted = _push_all(
        buffer,
        [
            {"name": "epoch.start", "pid": 1},
            {"name": "data", "value": 10, "pid": 1},
            {"name": "epoch.block", "pid": 1},
            {"name": "epoch.start", "pid": 1},
            {"name": "data", "value": 20, "pid": 1},
            {"name": "data", "value": 21, "pid": 1},
            {"name": "epoch.block", "pid": 1},
        ],
    )

    assert len(emitted) == 2
    assert all(item.get("epoch") == 1 for item in emitted[0])
    assert all(item.get("epoch") == 2 for item in emitted[1])


def test_epoch_buffer_pid_waits_for_all_pids():
    buffer = WindowBuffer()

    assert buffer.push({"name": "epoch.start", "pid": 10}) is None
    assert buffer.push({"name": "epoch.start", "pid": 20}) is None
    assert buffer.push({"name": "data", "value": 1, "pid": 10}) is None
    assert buffer.push({"name": "data", "value": 2, "pid": 20}) is None

    assert buffer.push({"name": "epoch.block", "pid": 10}) is None
    emitted = buffer.push({"name": "epoch.block", "pid": 20})
    assert emitted is not None
    assert all(item.get("epoch") == 1 for item in emitted)


def test_epoch_buffer_missing_pid_raises():
    buffer = WindowBuffer()
    buffer.push({"name": "epoch.start", "pid": 1})
    buffer.push({"name": "data", "value": 1, "pid": 1})

    with pytest.raises(ValueError):
        buffer.push({"name": "epoch.block"})

def test_pid_end_without_prior_seen_raises():
    buffer = WindowBuffer()
    with pytest.raises(ValueError):
        buffer.push({"name": "epoch.block", "pid": 99})


def test_data_without_pid_raises():
    buffer = WindowBuffer()
    with pytest.raises(ValueError):
        buffer.push({"name": "data", "value": 123})


def test_overlapping_epochs_fast_process_starts_next_before_slow_finishes():
    buffer = WindowBuffer()
    emitted = _push_all(
        buffer,
        [
            {"name": "epoch.start", "pid": 1},
            {"name": "data", "pid": 1, "value": "a1"},
            {"name": "epoch.start", "pid": 2},
            {"name": "data", "pid": 2, "value": "b1"},
            {"name": "epoch.block", "pid": 2},
            {"name": "epoch.start", "pid": 1},
            {"name": "data", "pid": 1, "value": "a2"},
            {"name": "epoch.block", "pid": 1},
        ],
    )

    assert len(emitted) == 1
    vals = [item.get("value") for item in emitted[0]]
    assert "a1" in vals and "b1" in vals
    assert "a2" not in vals


def test_epoch_buffer_custom_names():
    buffer = WindowBuffer(epoch_start_name="start", epoch_end_name="done", process_key="rank")
    emitted = _push_all(
        buffer,
        [
            {"name": "start", "rank": 7},
            {"name": "data", "rank": 7, "value": 1},
            {"name": "done", "rank": 7},
        ],
    )
    assert len(emitted) == 1
    assert all(item.get("epoch") == 1 for item in emitted[0])
    assert all(item.get("window") == 1 for item in emitted[0])


def test_window_boundary_tracker_queues_future_boundaries_from_fast_rank():
    tracker = WindowTracker(num_ranks=2)

    assert tracker.observe_start_boundary(pid=1) == 1
    assert tracker.observe_end_boundary(pid=1, boundary_ts_ns=100) == []
    assert tracker.observe_start_boundary(pid=1) == 2
    assert tracker.observe_end_boundary(pid=1, boundary_ts_ns=200) == []

    assert tracker.observe_start_boundary(pid=2) == 1
    completed = tracker.observe_end_boundary(pid=2, boundary_ts_ns=150)
    assert len(completed) == 1
    assert completed[0].window_index == 1
    assert completed[0].ranks_received == 2
    assert completed[0].boundary_ts_ns == 150

    assert tracker.observe_start_boundary(pid=2) == 2
    completed = tracker.observe_end_boundary(pid=2, boundary_ts_ns=250)
    assert len(completed) == 1
    assert completed[0].window_index == 2
    assert completed[0].ranks_received == 2
    assert completed[0].boundary_ts_ns == 250


def test_window_boundary_tracker_exposes_current_and_next_window_labels():
    tracker = WindowTracker(num_ranks=2)

    assert tracker.current_window(7) == 1
    assert tracker.next_window(7) == 1

    assert tracker.observe_start_boundary(pid=7) == 1
    assert tracker.observe_end_boundary(pid=7, boundary_ts_ns=10) == []
    assert tracker.current_window(7) == 2
    assert tracker.next_window(7) == 2


def test_window_boundary_tracker_uses_start_markers_for_active_window_assignment():
    tracker = WindowTracker(num_ranks=1)

    assert tracker.current_window(3) == 1
    assert tracker.observe_start_boundary(pid=3) == 1
    assert tracker.current_window(3) == 1
    assert tracker.observe_end_boundary(pid=3, boundary_ts_ns=10)[0].window_index == 1

    assert tracker.observe_start_boundary(pid=3) == 2
    assert tracker.current_window(3) == 2
    assert tracker.observe_end_boundary(pid=3, boundary_ts_ns=20)[0].window_index == 2


def test_window_boundary_tracker_advances_with_end_only_boundaries():
    tracker = WindowTracker(num_ranks=2)

    assert tracker.observe_end_boundary(pid=1, boundary_ts_ns=100) == []
    assert tracker.observe_end_boundary(pid=2, boundary_ts_ns=120)[0].window_index == 1

    assert tracker.current_window(1) == 2
    assert tracker.current_window(2) == 2

    assert tracker.observe_end_boundary(pid=1, boundary_ts_ns=200) == []
    completed = tracker.observe_end_boundary(pid=2, boundary_ts_ns=220)
    assert len(completed) == 1
    assert completed[0].window_index == 2
    assert completed[0].ranks_received == 2
    assert completed[0].boundary_ts_ns == 220

"""Unit tests for WindowBuffer: it tags events with the analysis `window` (the
longitudinal axis) and does NOT blindly copy that number into epoch/step."""
from dftracer.analyzer.streaming.window_buffer import WindowBuffer


def _buf():
    return WindowBuffer(window_start_name="epoch.start", window_end_name="epoch.block",
                        process_key="pid")


def test_stamps_window_not_epoch_or_step():
    buf = _buf()
    assert buf.push({"name": "epoch.start", "pid": 1}) is None
    assert buf.push({"name": "read", "pid": 1}) is None
    out = buf.push({"name": "epoch.block", "pid": 1})

    assert out is not None, "window should complete on the end marker"
    assert out, "completed window should carry its buffered events"
    for ev in out:
        assert ev["window"] == 1            # the analysis window is stamped
        assert "epoch" not in ev            # NOT a blind copy of the window
        assert "step" not in ev


def test_window_number_increments_per_start_marker():
    buf = _buf()
    buf.push({"name": "epoch.start", "pid": 1})
    buf.push({"name": "read", "pid": 1})
    buf.push({"name": "epoch.block", "pid": 1})

    buf.push({"name": "epoch.start", "pid": 1})
    buf.push({"name": "read", "pid": 1})
    out2 = buf.push({"name": "epoch.block", "pid": 1})

    assert all(ev["window"] == 2 for ev in out2)


def test_real_epoch_attribute_is_preserved_not_overwritten():
    # If an event already carries a real epoch attribute, the buffer must not
    # clobber it with the window number.
    buf = _buf()
    buf.push({"name": "epoch.start", "pid": 1})
    buf.push({"name": "read", "pid": 1, "epoch": 7})
    out = buf.push({"name": "epoch.block", "pid": 1})

    read_ev = next(ev for ev in out if ev["name"] == "read")
    assert read_ev["window"] == 1
    assert read_ev["epoch"] == 7            # preserved, not overwritten by window=1

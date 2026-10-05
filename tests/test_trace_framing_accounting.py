"""Writer target vs analyzer `seen`: the `[` framing line is counted by the writer."""
import json

from dftracer.analyzer.analyzer import split_trace_batch_lines


def _batch(pid, n, framing=True):
    ev = [json.dumps({"name": "e%d" % i, "pid": pid, "ph": "X"}) for i in range(n)]
    return ("[\n" if framing else "") + "\n".join(ev) + "\n"


def writer_count(payload):
    # shm_ring_writer.cpp: ++trace_events_written_ for every '\n'
    return payload.count("\n")


def test_first_flush_target_is_events_plus_framing():
    payload = _batch(4242, 79)
    lines, framing = split_trace_batch_lines(payload)
    assert writer_count(payload) == 80
    assert framing == 1 and len(lines) + framing == writer_count(payload)


def test_later_flush_has_no_framing():
    payload = _batch(4242, 5, framing=False)
    lines, framing = split_trace_batch_lines(payload)
    assert framing == 0 and len(lines) == writer_count(payload)


def test_rank_and_mpich_helper_each_pid_balances():
    # Cray-MPICH helper (unexec'd fork of the rank) writes its own framed batch under its pid.
    for pid, n in ((100, 79), (101, 3)):
        p = _batch(pid, n)
        lines, framing = split_trace_batch_lines(p)
        assert len(lines) + framing == writer_count(p)

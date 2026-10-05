"""Offline contracts for completed-window rank and worker provenance."""

import hashlib
import inspect
import json
import dataclasses as dc
import pandas as pd
import pytest
from pathlib import Path

# dftracer owns the namespace package in the venv.  Extend its package search
# path so this focused test imports this worktree's analyzer source without an
# install or shadow mutation.
import dftracer

SOURCE_NAMESPACE = str(Path(__file__).resolve().parents[1] / "python" / "dftracer")
if SOURCE_NAMESPACE not in dftracer.__path__:
    dftracer.__path__.insert(0, SOURCE_NAMESPACE)
import dftracer.analyzer as analyzer_namespace

SOURCE_ANALYZER = str(Path(SOURCE_NAMESPACE) / "analyzer")
if SOURCE_ANALYZER not in analyzer_namespace.__path__:
    analyzer_namespace.__path__.insert(0, SOURCE_ANALYZER)

from dftracer.analyzer.fastpath import (
    FastPathPlan,
    completed_window_stall_storage_provenance,
    completed_window_worker_roster_provenance,
)
from dftracer.analyzer.config import AnalyzerPresetConfigDLIOAILogging
from dftracer.analyzer.fact_engine import FactEngine
from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.streaming.window_buffer import WindowTracker
import dftracer.analyzer.fact_engine as fact_engine_module

assert Path(fact_engine_module.__file__).resolve() == Path(SOURCE_ANALYZER, "fact_engine.py").resolve()


def _event(pid, *, category="stall", function="stall.storage", duration=1.0):
    return {"pid": pid, "cat": category, "func_name": function, "time": duration}


def _write_roster_record(directory, *, prefix="probe-prefix", host="node-a", rank=0,
                         worker_id=0, pid=20, expected_workers=2, extra=None):
    record = {
        "schema_version": "dlio-worker-posix-roster.v1", "run_prefix": prefix,
        "host": host, "rank": rank, "worker_id": worker_id, "pid": pid,
        "expected_workers": expected_workers,
    }
    if extra:
        record.update(extra)
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:16]
    path = directory / (
        f"dlio-worker-posix-roster.v1.{digest}.rank-{rank}.worker-{worker_id}.json"
    )
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _roster_traces(*, include_second=True, include_unrelated=False):
    rows = [
        {"pid": 20, "proc_name": "worker-20", "cat": "posix_reader_lustre", "func_name": "pread"},
    ]
    if include_second:
        rows.append(
            {"pid": 21, "proc_name": "worker-21", "cat": "posix_reader_lustre", "func_name": "pread"}
        )
    if include_unrelated:
        rows.append(
            {"pid": 99, "proc_name": "other", "cat": "posix_reader_lustre", "func_name": "pread"}
        )
    return pd.DataFrame(rows)


def _roster_provenance(directory, traces, mapping, *, rank_pids=(10,)):
    return completed_window_worker_roster_provenance(
        traces, mapping, roster_dir=str(directory), roster_prefix="probe-prefix",
        expected_workers=2, expected_host="node-a", expected_ranks=1,
        expected_rank_pids=rank_pids,
    )


@pytest.mark.parametrize(
    "events, expected, covered, observed, event_count",
    [
        ([_event(10), _event(11)], [10, 11], 1.0, [10, 11], 2.0),
        ([_event(10, duration=0.0), _event(11, duration=0.0)], [10, 11], 1.0, [10, 11], 2.0),
        ([_event(10, category="compute", function="step")], [10, 11], 0.0, [], 0.0),
        ([_event(10)], [10, 11], 0.0, [10], 1.0),
        ([_event(10), _event(99)], [10, 11], 0.0, [10], 1.0),
    ],
)
def test_completed_window_stall_storage_coverage(events, expected, covered, observed, event_count):
    provenance = completed_window_stall_storage_provenance(pd.DataFrame(events), expected)
    assert provenance["stall_storage_rank_coverage"] == covered
    assert provenance["stall_storage_expected_rank_pids"] == expected
    assert provenance["stall_storage_observed_rank_pids"] == observed
    assert provenance["stall_storage_raw_event_count"] == event_count


def test_missing_pid_schema_fails_closed():
    provenance = completed_window_stall_storage_provenance(
        pd.DataFrame([{"cat": "stall", "func_name": "stall.storage", "time": 0.0}]), [10])
    assert provenance["stall_storage_rank_coverage"] == 0.0
    assert provenance["stall_storage_raw_event_count"] == 0.0


def test_worker_roster_whitelists_only_complete_same_window_reader_posix_rows(tmp_path):
    _write_roster_record(tmp_path, worker_id=0, pid=20)
    _write_roster_record(tmp_path, worker_id=1, pid=21)
    provenance = _roster_provenance(
        tmp_path, _roster_traces(include_unrelated=True),
        {"worker-20": 20, "worker-21": 21, "other": 99},
    )
    assert provenance["worker_roster_coverage"] == 1.0
    assert provenance["worker_roster_pids"] == [20, 21]
    assert provenance["worker_roster_observed_count"] == 2.0


@pytest.mark.parametrize("mutation", ["missing", "forged", "extra", "duplicate"])
def test_worker_roster_invalid_or_incomplete_fails_closed(tmp_path, mutation):
    _write_roster_record(tmp_path, worker_id=0, pid=20)
    if mutation == "forged":
        _write_roster_record(tmp_path, worker_id=1, pid=21, extra={"forged": True})
    elif mutation == "extra":
        _write_roster_record(tmp_path, rank=1, worker_id=1, pid=21)
        _write_roster_record(tmp_path, rank=1, worker_id=0, pid=22)
        _write_roster_record(tmp_path, worker_id=1, pid=21)
    elif mutation == "duplicate":
        _write_roster_record(tmp_path, worker_id=1, pid=20)
    else:
        # The absent worker-1 record is indistinguishable from a record that
        # has not become visible at this completed-window boundary: both are
        # intentionally suppressed rather than retried or inferred.
        pass
    provenance = _roster_provenance(
        tmp_path, _roster_traces(), {"worker-20": 20, "worker-21": 21},
    )
    assert provenance["worker_roster_coverage"] == 0.0


def test_worker_roster_late_record_can_only_enable_a_later_completed_window(tmp_path):
    _write_roster_record(tmp_path, worker_id=0, pid=20)
    first = _roster_provenance(
        tmp_path, _roster_traces(), {"worker-20": 20, "worker-21": 21},
    )
    _write_roster_record(tmp_path, worker_id=1, pid=21)
    later = _roster_provenance(
        tmp_path, _roster_traces(), {"worker-20": 20, "worker-21": 21},
    )
    assert first["worker_roster_coverage"] == 0.0
    assert later["worker_roster_coverage"] == 1.0


def test_worker_roster_requires_every_whitelisted_pid_to_have_reader_posix(tmp_path):
    _write_roster_record(tmp_path, worker_id=0, pid=20)
    _write_roster_record(tmp_path, worker_id=1, pid=21)
    provenance = _roster_provenance(
        tmp_path, _roster_traces(include_second=False), {"worker-20": 20},
    )
    assert provenance["worker_roster_coverage"] == 0.0
    assert provenance["worker_roster_observed_count"] == 1.0


def test_worker_roster_requires_complete_control_rank_pid_set(tmp_path):
    _write_roster_record(tmp_path, worker_id=0, pid=20)
    _write_roster_record(tmp_path, worker_id=1, pid=21)
    provenance = _roster_provenance(
        tmp_path, _roster_traces(), {"worker-20": 20, "worker-21": 21}, rank_pids=(),
    )
    assert provenance["worker_roster_coverage"] == 0.0


@pytest.mark.parametrize("category, rank, roster_pid, rank_pids", [
    ("stdio_reader", 0, 20, ()),
    ("posix_reader_lustre", 1, 20, ()),
    ("posix_reader_lustre", 0, 20, (20,)),
])
def test_worker_roster_rejects_non_posix_wrong_rank_or_control_pid(
    tmp_path, category, rank, roster_pid, rank_pids,
):
    _write_roster_record(tmp_path, rank=rank, worker_id=0, pid=roster_pid)
    _write_roster_record(tmp_path, rank=rank, worker_id=1, pid=21)
    traces = _roster_traces()
    traces.loc[traces["pid"] == 20, "cat"] = category
    provenance = _roster_provenance(
        tmp_path, traces, {"worker-20": 20, "worker-21": 21}, rank_pids=rank_pids,
    )
    assert provenance["worker_roster_coverage"] == 0.0


def test_worker_roster_rejects_fractional_raw_pid_without_truncation(tmp_path):
    _write_roster_record(tmp_path, worker_id=0, pid=20)
    _write_roster_record(tmp_path, worker_id=1, pid=21)
    traces = _roster_traces()
    traces["pid"] = traces["pid"].astype(float)
    traces.loc[traces["pid"] == 20, "pid"] = 20.5
    provenance = _roster_provenance(
        tmp_path, traces, {"worker-20": 20, "worker-21": 21},
    )
    assert provenance["worker_roster_coverage"] == 0.0


def test_worker_roster_provenance_reaches_only_worker_only_fact_context(tmp_path):
    rule = tmp_path / "roster.yaml"
    rule.write_text("""\
schema_version: analysisfact-rules.v1
rules:
  - id: fixture.worker-roster-coverage
    source_view: proc_name
    emit_mode: window
    worker_only: true
    window_reducer: min
    window_reducer_metric: reader_posix_read_ops_slope
    fact_type: fixture
    scope_layer: reader_posix
    required_metrics:
      - reader_posix_read_ops_slope
      - worker_roster_coverage
      - worker_roster_expected_count
      - worker_roster_observed_count
    when: reader_posix_read_ops_slope > 0.0 and worker_roster_coverage == 1.0 and worker_roster_expected_count == worker_roster_observed_count
    severity_score: reader_posix_read_ops_slope
""")
    engine = FactEngine.from_rule_file(str(rule), strict_time_semantics=True,
                                       allow_mixed_time_aggregates=False)
    plan = FastPathPlan.build(
        AnalyzerPresetConfigDLIOAILogging(), ["window", "proc_name"],
        list(AnalyzerPresetConfigDLIOAILogging().layer_defs), engine,
    )
    assert plan is not None
    engine.rules = [dc.replace(engine.rules[0], worker_only=False)]
    assert FastPathPlan.build(
        AnalyzerPresetConfigDLIOAILogging(), ["window", "proc_name"],
        list(AnalyzerPresetConfigDLIOAILogging().layer_defs), engine,
    ) is None
    engine = FactEngine.from_rule_file(str(rule), strict_time_semantics=True,
                                       allow_mixed_time_aggregates=False)
    frame = pd.DataFrame({
        "reader_posix_read_ops_slope": [0.5],
        # A row cannot forge analyzer-owned roster provenance.
        "worker_roster_coverage": [1.0],
        "worker_roster_expected_count": [2.0],
        "worker_roster_observed_count": [2.0],
    }, index=["worker"])
    context = {
        "native_window": 4, "worker_pids": [22], "row_pid_by_index": {"worker": 22},
        "window_metrics": {
            "domain_coherent": 1.0, "domain_procs_active": 1.0,
            "window_wall_time_proc_crit": 1.0, "domain_input_time_proc_crit": 0.0,
            "domain_decode_time_proc_crit": 0.0, "domain_comm_time_proc_crit": 0.0,
            "domain_compute_time_proc_crit": 1.0, "residual_time_proc_crit": 0.0,
            "epoch_nunique": 1.0, "step_nunique": 1.0, "step_min": 1.0, "step_max": 1.0,
        },
        "completed_window_worker_roster_provenance": {
            "worker_roster_coverage": 1.0,
            "worker_roster_expected_count": 2.0,
            "worker_roster_observed_count": 2.0,
        },
    }
    emitted = engine.evaluate({("proc_name",): frame}, raw_stats={}, context=context)
    assert len(emitted[("proc_name",)]) == 1
    del context["completed_window_worker_roster_provenance"]
    assert engine.evaluate({("proc_name",): frame}, raw_stats={}, context=context) == {}
    context["completed_window_worker_roster_provenance"] = {
        "worker_roster_coverage": "forged", "worker_roster_expected_count": 2.0,
        "worker_roster_observed_count": 2.0,
    }
    assert engine.evaluate({("proc_name",): frame}, raw_stats={}, context=context) == {}


def test_analyze_mofka_keeps_roster_inputs_optional():
    signature = inspect.signature(Analyzer.analyze_mofka)
    for name in (
        "worker_roster_dir", "worker_roster_prefix", "worker_roster_expected_workers",
    ):
        assert signature.parameters[name].default is None


def test_next_window_raw_stall_storage_cannot_cover_completed_window():
    """Use the native boundary tracker: window N+1 is not evidence for N."""
    tracker = WindowTracker(num_ranks=1, require_explicit_start=True)
    rank_pid = 10

    assert tracker.observe_start_boundary(rank_pid) == 1
    window_n_event = Analyzer._attach_stream_window(
        _event(rank_pid, category="compute", function="step"),
        tracker.current_window(rank_pid),
        native_window=True,
    )
    completed = tracker.observe_end_boundary(rank_pid, boundary_ts_ns=100)
    assert [boundary.window_index for boundary in completed] == [1]

    assert tracker.observe_start_boundary(rank_pid) == 2
    window_n_plus_one_event = Analyzer._attach_stream_window(
        _event(rank_pid, duration=0.0),
        tracker.current_window(rank_pid),
        native_window=True,
    )
    traces = pd.DataFrame([window_n_event, window_n_plus_one_event])

    completed_window = traces.loc[traces["window"] == completed[0].window_index]
    next_window = traces.loc[traces["window"] == tracker.current_window(rank_pid)]
    completed_provenance = completed_window_stall_storage_provenance(
        completed_window, [rank_pid])
    next_provenance = completed_window_stall_storage_provenance(next_window, [rank_pid])

    assert completed_provenance["stall_storage_rank_coverage"] == 0.0
    assert completed_provenance["stall_storage_raw_event_count"] == 0.0
    # The emitted zero-duration raw event proves this is window isolation, not
    # duration filtering or an absent event.
    assert next_provenance["stall_storage_rank_coverage"] == 1.0
    assert next_provenance["stall_storage_raw_event_count"] == 1.0


def test_coverage_reaches_only_proc_name_fact_context(tmp_path):
    rule = tmp_path / "coverage.yaml"
    rule.write_text("""\
schema_version: analysisfact-rules.v1
rules:
  - id: fixture.stall-storage-coverage
    source_view: proc_name
    emit_mode: window
    worker_only: true
    window_reducer: min
    window_reducer_metric: reader_posix_read_ops_slope
    fact_type: fixture
    scope_layer: reader_posix
    required_metrics:
      - reader_posix_read_ops_slope
      - stall_storage_rank_coverage
    when: reader_posix_read_ops_slope > 0.0 and stall_storage_rank_coverage == 1.0
    severity_score: reader_posix_read_ops_slope
""")
    engine = FactEngine.from_rule_file(str(rule), strict_time_semantics=True,
                                       allow_mixed_time_aggregates=False)
    frame = pd.DataFrame({"reader_posix_read_ops_slope": [0.5],
                          "stall_storage_rank_coverage": [1.0]}, index=["worker"])
    window_metrics = {
        "domain_coherent": 1.0, "domain_procs_active": 1.0,
        "window_wall_time_proc_crit": 1.0, "domain_input_time_proc_crit": 0.0,
        "domain_decode_time_proc_crit": 0.0, "domain_comm_time_proc_crit": 0.0,
        "domain_compute_time_proc_crit": 1.0, "residual_time_proc_crit": 0.0,
        "epoch_nunique": 1.0, "step_nunique": 1.0, "step_min": 1.0, "step_max": 1.0,
    }
    context = {
        "native_window": 4, "worker_pids": [22], "row_pid_by_index": {"worker": 22},
        "window_metrics": window_metrics,
        "completed_window_provenance": {"stall_storage_rank_coverage": 1.0},
    }
    emitted = engine.evaluate({("proc_name",): frame}, raw_stats={}, context=context)
    assert len(emitted[("proc_name",)]) == 1
    context["completed_window_provenance"] = {"stall_storage_rank_coverage": 0.0}
    assert engine.evaluate({("proc_name",): frame}, raw_stats={}, context=context) == {}
    del context["completed_window_provenance"]
    assert engine.evaluate({("proc_name",): frame}, raw_stats={}, context=context) == {}
    context["completed_window_provenance"] = {"stall_storage_rank_coverage": "invalid"}
    assert engine.evaluate({("proc_name",): frame}, raw_stats={}, context=context) == {}
    context["completed_window_provenance"] = {"stall_storage_rank_coverage": float("nan")}
    assert engine.evaluate({("proc_name",): frame}, raw_stats={}, context=context) == {}

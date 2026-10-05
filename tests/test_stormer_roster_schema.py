"""Schema-dispatched worker roster provenance: DLIO default unchanged, StorMer accepted, all else
fails closed; plus the StorMer input rule end to end on a synthetic StorMer roster (CPU, no run
data).  Offline control-path checks only."""

import hashlib
import json
from pathlib import Path

import dftracer

# Run against THIS tree's analyzer sources, not the venv's installed copy.
_SRC = str(Path(__file__).resolve().parents[1] / "python/dftracer")
if _SRC not in list(dftracer.__path__):
    dftracer.__path__.insert(0, _SRC)
import dftracer.analyzer as _analyzer_ns  # noqa: E402
_SRC_AN = str(Path(_SRC) / "analyzer")
if _SRC_AN not in _analyzer_ns.__path__:
    _analyzer_ns.__path__.insert(0, _SRC_AN)

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dftracer.analyzer.fastpath import completed_window_worker_roster_provenance as roster_fn  # noqa: E402

DLIO = "dlio-worker-posix-roster.v1"
STORMER = "stormer-worker-posix-roster.v1"
PREFIX = "lf_test_prefix"
HOST = "h0"


def _write(directory, schema, *, rank, worker_id, pid, prefix=PREFIX, workers=2):
    digest = hashlib.sha256(prefix.encode()).hexdigest()[:16]
    (directory / f"{schema}.{digest}.rank-{rank}.worker-{worker_id}.json").write_text(json.dumps({
        "schema_version": schema, "run_prefix": prefix, "host": HOST, "rank": rank,
        "worker_id": worker_id, "pid": pid, "expected_workers": workers}))


def _fixture(directory, schema):
    for w, pid in enumerate((20, 21)):
        _write(directory, schema, rank=0, worker_id=w, pid=pid)
    rows = [{"pid": 10, "proc_name": "rank-10", "cat": "compute", "func_name": "compute"}]
    rows += [{"pid": p, "proc_name": f"worker-{p}", "cat": "posix_reader_lustre", "func_name": "read"}
             for p in (20, 21)]
    return pd.DataFrame(rows), {"rank-10": 10, "worker-20": 20, "worker-21": 21}


def _call(directory, traces, idx, **kw):
    return roster_fn(traces, idx, roster_dir=str(directory), roster_prefix=PREFIX,
                     expected_workers=2, expected_host=HOST, expected_ranks=1,
                     expected_rank_pids=[10], **kw)


def test_dlio_default_is_unchanged(tmp_path):
    traces, idx = _fixture(tmp_path, DLIO)
    assert _call(tmp_path, traces, idx)["worker_roster_coverage"] == 1.0
    assert _call(tmp_path, traces, idx, roster_schema=DLIO)["worker_roster_coverage"] == 1.0


def test_stormer_schema_accepted_only_when_requested(tmp_path):
    traces, idx = _fixture(tmp_path, STORMER)
    got = _call(tmp_path, traces, idx, roster_schema=STORMER)
    assert got["worker_roster_coverage"] == 1.0 and got["worker_roster_pids"] == [20, 21]
    # The default (DLIO) request must NOT silently accept StorMer records.
    assert _call(tmp_path, traces, idx)["worker_roster_coverage"] == 0.0


@pytest.mark.parametrize("written,requested", [(DLIO, STORMER), (STORMER, DLIO)])
def test_cross_schema_fails_closed(tmp_path, written, requested):
    traces, idx = _fixture(tmp_path, written)
    assert _call(tmp_path, traces, idx, roster_schema=requested)["worker_roster_coverage"] == 0.0


def test_unknown_schema_fails_closed_even_if_records_match(tmp_path):
    traces, idx = _fixture(tmp_path, "other-worker-posix-roster.v1")
    assert _call(tmp_path, traces, idx, roster_schema="other-worker-posix-roster.v1"
                 )["worker_roster_coverage"] == 0.0


@pytest.mark.parametrize("schema", (DLIO, STORMER))
def test_both_schemas_keep_the_same_integrity_checks(tmp_path, schema):
    traces, idx = _fixture(tmp_path, schema)
    # missing worker row in the window -> not covered
    partial = traces[traces["pid"] != 21]
    assert _call(tmp_path, partial, idx, roster_schema=schema)["worker_roster_coverage"] == 0.0
    # a record whose file name does not match its contents -> whole roster rejected
    (tmp_path / "junk.json").write_text(json.dumps({"schema_version": schema}))
    assert _call(tmp_path, traces, idx, roster_schema=schema)["worker_roster_coverage"] == 0.0


# ---- StorMer input rule end to end on a synthetic StorMer roster -------------------------------
_RULE = (Path(__file__).resolve().parents[1]
         / "python/dftracer/analyzer/configs/fact_rules/liveflow-stormer-input-slope-exposure.yaml")


def _ev(pid, name, cat, func, seconds, io_cat=0):
    return {"window": 7, "epoch": 1, "step": 1, "pid": pid, "proc_name": name, "cat": cat,
            "func_name": func, "io_cat": io_cat, "time": seconds, "size": 0}


def _e2e(tmp_path, monkeypatch, *, schema=STORMER, request_schema=STORMER, stall=6.0):
    from dask.distributed import Client, LocalCluster
    from dftracer.analyzer.config import AnalyzerPresetConfigStormerXLayer
    from dftracer.analyzer.dftracer import DFTracerAnalyzer
    from dftracer.analyzer.fastpath import completed_window_stall_storage_provenance  # noqa: F401
    monkeypatch.setattr("socket.gethostname", lambda: HOST)
    for w, pid in enumerate((20, 21)):
        _write(tmp_path, schema, rank=0, worker_id=w, pid=pid)
    rows = [_ev(10, "rank-10", "window", "window.stop", 10.0), _ev(10, "rank-10", "compute", "compute", 4.0)]
    if stall > 0:   # StorMer omits zero-duration stall records
        rows.append(_ev(10, "rank-10", "stall", "stall.storage", stall))
    rows += [_ev(20, "worker-20", "posix_reader_lustre", "read", 2.0, 1),
             _ev(21, "worker-21", "posix_reader_lustre", "read", 1.0, 1)]
    traces = pd.DataFrame(rows)
    idx = {n: int(p) for n, p in traces[["proc_name", "pid"]].drop_duplicates().itertuples(index=False)}
    roster = roster_fn(traces, idx, roster_dir=str(tmp_path), roster_prefix=PREFIX, expected_workers=2,
                       expected_host=HOST, expected_ranks=1, expected_rank_pids=[10],
                       roster_schema=request_schema)
    context = {"native_window": 7, "rank_pids": [10],
               "worker_pids": roster["worker_roster_pids"] if roster["worker_roster_coverage"] == 1.0 else [],
               "row_pid_by_index": idx, "completed_window_worker_roster_provenance": roster}
    cluster = client = None
    try:
        cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=":0")
        client = Client(cluster)
        an = DFTracerAnalyzer(preset=AnalyzerPresetConfigStormerXLayer(), checkpoint=False,
                              checkpoint_dir="", time_granularity=1.0, time_resolution=1.0,
                              facts_config={"enabled": True, "rule_file": str(_RULE),
                                            "emit_analysis_facts": True, "emit_flat_views": True})
        an._debug_window_provenance = {"rank_pids": [10]}
        an._fact_window_context = context
        res = an._analyze_trace_fast(traces=traces, proc_view_types=["window", "proc_name"],
                                     metric_boundaries={}, raw_stats={})
        assert res is not None
        return res.analysis_facts
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()


def test_stormer_rule_fires_on_a_stormer_roster(tmp_path, monkeypatch):
    facts = _e2e(tmp_path, monkeypatch)
    (fact,) = facts[("proc_name",)]
    m = fact.evidence["metrics"]
    assert m["worker_roster_coverage"] == 1.0
    assert m["measured_total_wait_time_proc_crit"] > 0
    assert fact.opportunity_tags == ["reader_parallelism"]


@pytest.mark.parametrize("case", ("dlio_schema_roster", "wrong_schema_requested", "no_measured_wait"))
def test_stormer_rule_suppressed_end_to_end(tmp_path, monkeypatch, case):
    if case == "dlio_schema_roster":
        facts = _e2e(tmp_path, monkeypatch, schema=DLIO, request_schema=STORMER)
    elif case == "wrong_schema_requested":
        facts = _e2e(tmp_path, monkeypatch, schema=STORMER, request_schema=DLIO)
    else:
        facts = _e2e(tmp_path, monkeypatch, stall=0.0)
    assert facts == {}, case

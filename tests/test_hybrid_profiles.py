import json

import pytest
from dask.distributed import Client, LocalCluster
from omegaconf import OmegaConf

from dftracer.analyzer.config import AnalyzerPresetConfigDLIOAILogging, AnalyzerPresetConfigPOSIX
from dftracer.analyzer.dftracer import DFTracerAnalyzer


TRACE_CONTENT = [
    "[",
    json.dumps(
        {
            "id": 1,
            "name": "HH",
            "cat": "dftracer",
            "pid": 1,
            "tid": 1,
            "ph": "M",
            "args": {"hhash": "h1", "name": "hostA", "value": "h1"},
        }
    ),
    json.dumps(
        {
            "id": 2,
            "name": "FH",
            "cat": "dftracer",
            "pid": 1,
            "tid": 1,
            "ph": "M",
            "args": {"hhash": "h1", "name": "/tmp/data/file.bin", "value": "f1"},
        }
    ),
    json.dumps(
        {
            "id": 3,
            "name": "read",
            "cat": "POSIX",
            "pid": 1,
            "tid": 1,
            "ph": "X",
            "ts": 6000123,
            "dur": 200,
            "args": {"hhash": "h1", "fhash": "f1", "ret": 4096, "offset": 0},
        }
    ),
    json.dumps(
        {
            "id": 4,
            "name": "open64",
            "cat": "POSIX",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 3, "dur_sum": 300},
        }
    ),
    "]",
]

FULL_TRACE_CONTENT = [
    "[",
    json.dumps(
        {
            "id": 1,
            "name": "HH",
            "cat": "dftracer",
            "pid": 1,
            "tid": 1,
            "ph": "M",
            "args": {"hhash": "h1", "name": "hostA", "value": "h1"},
        }
    ),
    json.dumps(
        {
            "id": 2,
            "name": "FH",
            "cat": "dftracer",
            "pid": 1,
            "tid": 1,
            "ph": "M",
            "args": {"hhash": "h1", "name": "/tmp/data/file.bin", "value": "f1"},
        }
    ),
    json.dumps(
        {
            "id": 3,
            "name": "read",
            "cat": "POSIX",
            "pid": 1,
            "tid": 1,
            "ph": "X",
            "ts": 6000123,
            "dur": 200,
            "args": {"hhash": "h1", "fhash": "f1", "ret": 4096, "offset": 0},
        }
    ),
    json.dumps(
        {
            "id": 4,
            "name": "ai_root",
            "cat": "ai_root",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 1, "dur_sum": 5000000},
        }
    ),
    json.dumps(
        {
            "id": 5,
            "name": "train",
            "cat": "pipeline",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 1, "dur_sum": 5000000},
        }
    ),
    json.dumps(
        {
            "id": 6,
            "name": "epoch.1",
            "cat": "pipeline",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 1, "dur_sum": 5000000},
        }
    ),
    json.dumps(
        {
            "id": 7,
            "name": "compute",
            "cat": "compute",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 2, "dur_sum": 4000000},
        }
    ),
    json.dumps(
        {
            "id": 8,
            "name": "fetch.iter",
            "cat": "pipeline",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 1, "dur_sum": 1000000},
        }
    ),
    json.dumps(
        {
            "id": 9,
            "name": "item",
            "cat": "data",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 2, "dur_sum": 2000000},
        }
    ),
    json.dumps(
        {
            "id": 10,
            "name": "item",
            "cat": "data",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 1, "dur": 500000},
        }
    ),
    json.dumps(
        {
            "id": 11,
            "name": "open64",
            "cat": "POSIX",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 3, "dur_sum": 300},
        }
    ),
    "]",
]


@pytest.fixture
def dask_client():
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        scheduler_port=0,
        silence_logs="error",
    )
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


@pytest.fixture
def hybrid_trace_path(tmp_path):
    trace_path = tmp_path / "hybrid.pfw"
    trace_path.write_text("\n".join(TRACE_CONTENT) + "\n", encoding="utf-8")
    return trace_path


@pytest.fixture
def full_hybrid_trace_path(tmp_path):
    trace_path = tmp_path / "full-hybrid.pfw"
    trace_path.write_text("\n".join(FULL_TRACE_CONTENT) + "\n", encoding="utf-8")
    return trace_path


def make_analyzer(tmp_path, time_granularity, preset=None):
    return DFTracerAnalyzer(
        preset=OmegaConf.structured(preset or AnalyzerPresetConfigPOSIX()),
        checkpoint=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        debug=False,
        quantile_stats=False,
        time_approximate=True,
        time_granularity=time_granularity,
        time_resolution=10**6,
        time_sliced=False,
        verbose=False,
    )


def test_read_trace_standardizes_profiles_and_aligns_to_5s_grid(dask_client, hybrid_trace_path, tmp_path):
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    read_result = analyzer.read_trace(
        trace_path=str(hybrid_trace_path),
        extra_columns=None,
        extra_columns_fn=None,
    )

    traces = read_result.traces.compute()
    profiles = read_result.profiles.compute()

    assert len(traces) == 1
    assert len(profiles) == 1

    profile = profiles.iloc[0]
    assert profile["func_name"] == "open64"
    assert profile["proc_name"] == "app#hostA#1#1"
    assert profile["file_name"] == "/tmp/data/file.bin"
    assert profile["count"] == 3
    assert profile["time"] == pytest.approx(0.0003)
    assert profile["size"] is None or str(profile["size"]) == "<NA>"
    assert profile["time_start"] == 5_000_000
    assert profile["time_end"] == 10_000_000
    assert profile["time_range"] == 1
    assert profile["time_start"] % 5_000_000 == 0


def test_read_trace_rejects_non_multiple_profile_granularity(dask_client, hybrid_trace_path, tmp_path):
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=7)

    with pytest.raises(NotImplementedError, match="integer multiple of 5s"):
        analyzer.read_trace(
            trace_path=str(hybrid_trace_path),
            extra_columns=None,
            extra_columns_fn=None,
        )


def test_analyze_trace_reconciles_hybrid_hlm_and_tracks_raw_stats(dask_client, hybrid_trace_path, tmp_path):
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    result = analyzer.analyze_trace(
        trace_path=str(hybrid_trace_path),
        view_types=["file_name", "proc_name", "time_range"],
    )

    assert result.profiles is not None
    assert int(result.raw_stats.trace_event_count) == 1
    assert int(result.raw_stats.profile_event_count) == 3
    assert int(result.raw_stats.total_event_count) == 4

    posix_hlm = result.get_hlm("posix").compute().reset_index()

    assert set(posix_hlm["func_name"]) == {"read", "open64"}
    by_func = posix_hlm.set_index("func_name")
    assert int(by_func.loc["read", "count"]) == 1
    assert int(by_func.loc["open64", "count"]) == 3


def test_analyze_trace_reconciles_profiles_per_layer_for_full_hybrid_case(
    dask_client,
    full_hybrid_trace_path,
    tmp_path,
):
    analyzer = make_analyzer(
        tmp_path=tmp_path,
        time_granularity=5,
        preset=AnalyzerPresetConfigDLIOAILogging(),
    )

    result = analyzer.analyze_trace(
        trace_path=str(full_hybrid_trace_path),
        view_types=["proc_name", "time_range"],
    )

    assert result.profiles is not None

    app_hlm = result.get_hlm("app").compute().reset_index()
    compute_hlm = result.get_hlm("compute").compute().reset_index()
    data_hlm = result.get_hlm("data_loader").compute().reset_index()
    posix_hlm = result.get_hlm("posix").compute().reset_index()

    assert set(app_hlm["func_name"]) == {"ai_root"}
    assert set(compute_hlm["func_name"]) == {"compute"}
    assert set(data_hlm["func_name"]) == {"item"}
    assert set(posix_hlm["func_name"]) == {"read", "open64"}

    assert int(compute_hlm.iloc[0]["count"]) == 2
    assert int(data_hlm.iloc[0]["count"]) == 3
    assert int(posix_hlm.set_index("func_name").loc["open64", "count"]) == 3

import json

import pytest
from dask.distributed import Client, LocalCluster
from omegaconf import OmegaConf

from dftracer.analyzer.config import AnalyzerPresetConfigPOSIX
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


def make_analyzer(tmp_path, time_granularity):
    return DFTracerAnalyzer(
        preset=OmegaConf.structured(AnalyzerPresetConfigPOSIX()),
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

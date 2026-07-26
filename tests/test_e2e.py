import gzip
import json
import os
import pathlib
import pytest
import random
import shutil
from dask.distributed import LocalCluster
from dftracer.analyzer import init_with_hydra
from glob import glob

# Ships as .pfw.gz, so the native indexer reads it end to end.
GZ_TRACE_FIXTURE = pathlib.Path("tests/data/extracted/dftracer-dlio/trace-0-of-8.pfw.gz")


# Full test matrix for comprehensive testing
full_analyzer_trace_params = [
    ("darshan", "posix", "tests/data/extracted/darshan-posix"),
    ("darshan", "posix", "tests/data/extracted/darshan-posix-dxt"),
    ("dftracer", "dlio", "tests/data/extracted/dftracer-dlio"),
    ("dftracer", "dlio-prev", "tests/data/extracted/dftracer-dlio-prev"),
    ("dftracer", "posix", "tests/data/extracted/dftracer-posix"),
    ("recorder", "posix", "tests/data/extracted/recorder-posix-parquet"),
]
full_checkpoint_params = [True, False]

# Reduced matrix for smoke testing (fast runs)
smoke_analyzer_trace_params = [random.choice(full_analyzer_trace_params)]
smoke_checkpoint_params = [False]  # Skip checkpoint to make tests faster


@pytest.fixture(scope="session")
def dask_cluster():
    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    # This teardown code runs after all tests are done
    cluster.close()


@pytest.mark.full
@pytest.mark.parametrize("analyzer, preset, trace_path", full_analyzer_trace_params)
@pytest.mark.parametrize("checkpoint", full_checkpoint_params)
def test_e2e_full(
    analyzer: str,
    preset: str,
    trace_path: str,
    checkpoint: bool,
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """Full test suite with all parameter combinations."""
    _test_e2e(analyzer, preset, trace_path, checkpoint, tmp_path, dask_cluster)


@pytest.mark.smoke
@pytest.mark.parametrize("analyzer, preset, trace_path", smoke_analyzer_trace_params)
@pytest.mark.parametrize("checkpoint", smoke_checkpoint_params)
def test_e2e_smoke(
    analyzer: str,
    preset: str,
    trace_path: str,
    checkpoint: bool,
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """Smoke test with minimal parameter combinations for quick validation."""
    _test_e2e(analyzer, preset, trace_path, checkpoint, tmp_path, dask_cluster)


@pytest.mark.smoke
def test_json_output_file(tmp_path: pathlib.Path, dask_cluster: LocalCluster) -> None:
    """Verify JSON output file is created with the expected schema and views."""
    checkpoint_dir = f"{tmp_path}/checkpoints"
    scheduler_address = dask_cluster.scheduler_address
    output_path = tmp_path / "analysis.json"
    hydra_overrides = [
        "analyzer=dftracer",
        "analyzer/preset=posix",
        "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={checkpoint_dir}",
        "cluster=external",
        "cluster.restart_on_connect=True",
        f"cluster.scheduler_address={scheduler_address}",
        "output=json",
        f"output.file_path={output_path}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        "trace_path=tests/data/extracted/dftracer-posix",
        "view_types=[time_range,proc_name]",
    ]

    dfa = init_with_hydra(hydra_overrides=hydra_overrides)
    result = dfa.analyze_trace()
    dfa.output.handle_result(result)

    assert output_path.exists(), f"Expected JSON output at {output_path}"
    with output_path.open() as f:
        payload = json.load(f)

    assert payload["schema_version"] == "1"
    assert "raw_stats" in payload
    assert "views" in payload
    assert "time_range" in payload["views"]
    assert "proc_name" in payload["views"]
    assert "summary" in payload["views"]["time_range"]
    assert "additional_metrics" in payload["views"]["time_range"]
    assert "flat_views" not in payload

    dfa.shutdown()
    assert dfa.client.status == "closed", "Dask client should be closed after shutdown"


def _test_e2e(
    analyzer: str,
    preset: str,
    trace_path: str,
    checkpoint: bool,
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """Common test logic extracted to avoid duplication."""
    checkpoint_dir = f"{tmp_path}/checkpoints"
    scheduler_address = dask_cluster.scheduler_address

    view_types = ["proc_name", "time_range"]
    if trace_path.endswith("darshan-posix"):
        view_types = ["file_name", "proc_name"]

    hydra_overrides = [
        f"analyzer={analyzer}",
        f"analyzer/preset={preset}",
        f"analyzer.checkpoint={checkpoint}",
        f"analyzer.checkpoint_dir={checkpoint_dir}",
        "cluster=external",
        f"cluster.restart_on_connect={True}",
        f"cluster.scheduler_address={scheduler_address}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={trace_path}",
        f"view_types=[{','.join(view_types)}]",
    ]

    # Allow enabling debug logs for investigation via env var
    if os.getenv("DFANALYZER_DEBUG", "").lower() in {"1", "true", "yes"}:
        hydra_overrides.append("debug=True")

    assign_epochs = analyzer == "dftracer" and preset.startswith("dlio")
    if assign_epochs:
        hydra_overrides.append("analyzer.assign_epochs=True")

    dfa = init_with_hydra(hydra_overrides=hydra_overrides)

    assert dfa.hydra_config.analyzer.checkpoint == checkpoint
    assert dfa.hydra_config.analyzer.checkpoint_dir == checkpoint_dir
    assert dfa.hydra_config.analyzer.preset.name == preset
    assert dfa.hydra_config.trace_path == trace_path
    if assign_epochs:
        assert dfa.hydra_config.analyzer.assign_epochs

    # Run the main function
    result = dfa.analyze_trace()

    assert len(result.flat_views) == len(dfa.hydra_config.view_types), (
        f"Expected {len(dfa.hydra_config.view_types)} views, got {len(result.flat_views)}"
    )
    assert len(result.layers) == len(dfa.hydra_config.analyzer.preset.layer_defs), (
        f"Expected {len(dfa.hydra_config.analyzer.preset.layer_defs)} layers, got {len(result.layers)}"
    )
    if checkpoint:
        assert any(glob(f"{result.checkpoint_dir}/*.parquet")), "No checkpoint found"

    # Shutdown the Dask client and cluster
    dfa.shutdown()

    # Verify that the Dask client is closed
    assert dfa.client.status == "closed", "Dask client should be closed after shutdown"


def _hydra_overrides_for_path(trace_path, tmp_path, scheduler_address, preset="posix"):
    return [
        "analyzer=dftracer",
        f"analyzer/preset={preset}",
        "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        "cluster=external",
        "cluster.restart_on_connect=True",
        f"cluster.scheduler_address={scheduler_address}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={trace_path}",
        "view_types=[proc_name]",
    ]


@pytest.mark.smoke
def test_read_trace_handles_mixed_pfw_and_pfw_gz(
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """A trace dir mixing .pfw and .pfw.gz must analyze cleanly.

    The dir is built in tmp_path rather than reusing a shipped fixture so the
    uncompressed member is guaranteed to be a real trace.
    """
    assert GZ_TRACE_FIXTURE.exists(), f"Missing fixture: {GZ_TRACE_FIXTURE}"

    mixed_dir = tmp_path / "mixed_traces"
    mixed_dir.mkdir()
    shutil.copy(GZ_TRACE_FIXTURE, mixed_dir / "trace_b.pfw.gz")
    with gzip.open(GZ_TRACE_FIXTURE, "rb") as src, (mixed_dir / "trace_a.pfw").open("wb") as dst:
        shutil.copyfileobj(src, dst)

    overrides = _hydra_overrides_for_path(mixed_dir, tmp_path, dask_cluster.scheduler_address)
    dfa = init_with_hydra(hydra_overrides=overrides)
    try:
        result = dfa.analyze_trace()
        assert len(result.flat_views) == 1, "Expected one flat view"
    finally:
        dfa.shutdown()


@pytest.mark.smoke
def test_read_trace_skips_zero_byte_traces_alongside_valid(
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """A trace dir mixing valid and zero-byte files must analyze cleanly,
    with the empty files reported and skipped instead of crashing the run."""
    assert GZ_TRACE_FIXTURE.exists(), f"Missing fixture: {GZ_TRACE_FIXTURE}"

    trace_dir = tmp_path / "partial_empty"
    trace_dir.mkdir()
    shutil.copy(GZ_TRACE_FIXTURE, trace_dir / "valid.pfw.gz")
    (trace_dir / "empty.pfw").write_bytes(b"")
    (trace_dir / "empty.pfw.gz").write_bytes(b"")

    overrides = _hydra_overrides_for_path(trace_dir, tmp_path, dask_cluster.scheduler_address)
    dfa = init_with_hydra(hydra_overrides=overrides)
    try:
        result = dfa.analyze_trace()
        assert len(result.flat_views) == 1
    finally:
        dfa.shutdown()


@pytest.mark.smoke
def test_read_trace_raises_clear_error_when_all_traces_are_zero_byte(
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """All-empty trace dir must raise ValueError with an actionable message.

    The native indexer happily 'indexes' zero-byte files and yields no events,
    so without this check the run reports a successful analysis of 0 events.
    """
    trace_dir = tmp_path / "all_empty"
    trace_dir.mkdir()
    (trace_dir / "a.pfw").write_bytes(b"")
    (trace_dir / "b.pfw.gz").write_bytes(b"")

    overrides = _hydra_overrides_for_path(trace_dir, tmp_path, dask_cluster.scheduler_address)
    dfa = init_with_hydra(hydra_overrides=overrides)
    try:
        with pytest.raises(ValueError, match="zero bytes"):
            dfa.analyze_trace()
    finally:
        dfa.shutdown()

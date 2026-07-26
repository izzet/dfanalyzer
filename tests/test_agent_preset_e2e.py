"""End-to-end coverage for the ``agent`` analyzer preset.

``tests/data`` ships no recorded agent trace, so these tests run against the
small synthetic workflow in ``tests/agent_trace.py``.
"""

import pathlib
import pytest
from dask.distributed import LocalCluster
from dftracer.analyzer import init_with_hydra

from .agent_trace import write_agent_trace

AGENT_LAYERS = ["workflow", "step", "llm", "tool", "data", "message", "judge", "posix"]


@pytest.fixture(scope="session")
def dask_cluster():
    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    cluster.close()


def _agent_overrides(trace_path, tmp_path, scheduler_address, view_types="[proc_name,time_range]"):
    return [
        "analyzer=dftracer",
        "analyzer/preset=agent",
        "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        "cluster=external",
        "cluster.restart_on_connect=True",
        f"cluster.scheduler_address={scheduler_address}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={trace_path}",
        f"view_types={view_types}",
    ]


@pytest.fixture
def agent_trace_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    trace_dir = tmp_path / "agent_traces"
    trace_dir.mkdir()
    write_agent_trace(trace_dir / "trace-0-of-1.pfw.gz")
    return trace_dir


@pytest.mark.smoke
def test_agent_preset_registers_expected_layers() -> None:
    """The preset is reachable as analyzer/preset=agent with its full layer tree."""
    from hydra import compose, initialize
    from dftracer.analyzer.config import init_hydra_config_store

    with initialize(version_base=None, config_path=None):
        init_hydra_config_store()
        cfg = compose(config_name="config", overrides=["analyzer/preset=agent"])

    preset = cfg.analyzer.preset
    assert preset.name == "agent"
    assert list(preset.layer_defs.keys()) == AGENT_LAYERS
    # Dependency tree: step under workflow, the rest under step, data under tool.
    assert preset.layer_deps["step"] == "workflow"
    assert preset.layer_deps["data"] == "tool"
    for layer in ("llm", "tool", "message", "judge"):
        assert preset.layer_deps[layer] == "step"
    # Token-normalized metrics are declared for both view types.
    for view_type in ("proc_name", "time_range"):
        metrics = preset.additional_metrics[view_type]
        assert "bytes_read_per_output_token" in metrics
        assert "io_ops_per_output_token" in metrics
        assert "io_time_per_output_token" in metrics
        assert "bytes_per_io_op" in metrics
        assert "metadata_op_frac" in metrics
        assert "read_write_ratio" in metrics
    # step is promoted into the HLM groupby so agent spans survive rollup.
    assert "step" in preset.hlm_fields
    assert preset.time_correlation.enabled
    assert preset.time_correlation.layer == "step"


@pytest.mark.smoke
def test_agent_preset_analyzes_agent_trace(
    agent_trace_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """A synthetic agent trace analyzes cleanly across all agent layers."""
    overrides = _agent_overrides(agent_trace_dir, tmp_path, dask_cluster.scheduler_address)
    dfa = init_with_hydra(hydra_overrides=overrides)
    try:
        result = dfa.analyze_trace()
        assert list(result.layers) == AGENT_LAYERS
        assert len(result.flat_views) == 2
        assert result.raw_stats.total_event_count > 0

        proc_view = result.flat_views[("proc_name",)]
        # Every agent layer contributes a count column to the flat view.
        for layer in AGENT_LAYERS:
            assert f"{layer}_count_sum" in proc_view.columns, f"missing {layer}_count_sum"
        # Token-normalized additional metrics are materialized as columns. Their
        # values stay null until the native indexer can surface args.* fields.
        for metric in ("bytes_read_per_output_token", "bytes_per_io_op", "read_write_ratio"):
            assert metric in proc_view.columns, f"missing {metric}"
    finally:
        dfa.shutdown()


@pytest.mark.smoke
def test_agent_preset_runs_on_non_agent_trace(
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """The preset must not break on traces with no agent categories.

    The agent layers legitimately end up empty; the run must still complete
    rather than failing on the token-normalized metrics whose input columns are
    absent.
    """
    overrides = _agent_overrides(
        "tests/data/extracted/dftracer-dlio",
        tmp_path,
        dask_cluster.scheduler_address,
        view_types="[time_range]",
    )
    dfa = init_with_hydra(hydra_overrides=overrides)
    try:
        result = dfa.analyze_trace()
        assert list(result.layers) == AGENT_LAYERS
        assert result.raw_stats.total_event_count == 125669
    finally:
        dfa.shutdown()

"""The Dask and pandas view paths must produce identical results.

`_compute_view` runs on either a Dask frame or a pandas one depending on
whether `compute_views` decided the layer was small enough to materialise.
Both paths are expressed once through `DataFrameOps`, but the engines differ
in ways that are easy to get wrong and hard to notice -- most of them only
show up on empty layers, and only as a dtype:

* Dask coerces every partition to a meta schema, so helpers that short-circuit
  on an empty frame still come out with the populated schema.
* Where no meta is given, Dask infers one by running the function against a
  synthetic non-empty frame, so an empty partition still gets the schema the
  function would have produced had there been rows.

Comparing the two paths end to end is the only check that catches those.
"""

import pathlib
import pytest
from dask.distributed import LocalCluster
from dftracer.analyzer import init_with_hydra
from pandas.testing import assert_frame_equal


# A POSIX-only trace analysed with the AI preset: small enough to run twice in
# a smoke test, but its 14 layers include several the trace has no events for.
# Those empty layers are the whole point -- the engines only disagree there, and
# a fixture whose layers are all populated makes this test vacuous.
TRACE_PATH = "tests/data/extracted/dftracer-posix"
PRESET = "ai"


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    cluster.close()


def _analyze(tmp_path: pathlib.Path, cluster, view_materialize_max_bytes: int, checkpoint: bool = False):
    overrides = [
        "analyzer=dftracer",
        f"analyzer/preset={PRESET}",
        f"analyzer.checkpoint={checkpoint}",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        f"analyzer.view_materialize_max_bytes={view_materialize_max_bytes}",
        "cluster=external",
        "cluster.restart_on_connect=False",
        f"cluster.scheduler_address={cluster.scheduler_address}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={TRACE_PATH}",
        "view_types=[time_range,proc_name]",
    ]
    dfa = init_with_hydra(hydra_overrides=overrides)
    try:
        result = dfa.analyze_trace()
        return {key: view.copy() for key, view in result.flat_views.items()}
    finally:
        dfa.shutdown()


@pytest.mark.smoke
def test_view_paths_agree(tmp_path: pathlib.Path, dask_cluster) -> None:
    """Materialising a layer must not change a single value or dtype."""
    on_dask = _analyze(tmp_path / "dask", dask_cluster, 0)
    on_pandas = _analyze(tmp_path / "pandas", dask_cluster, 256 * 1024**2)

    assert set(on_dask) == set(on_pandas), (
        f"different view keys: dask={sorted(map(str, on_dask))} pandas={sorted(map(str, on_pandas))}"
    )
    assert on_dask, "expected at least one flat view"

    for view_key in on_dask:
        assert_frame_equal(
            on_dask[view_key],
            on_pandas[view_key],
            check_dtype=True,
            check_exact=False,
            rtol=1e-9,
            obj=f"flat view {view_key}",
        )


@pytest.mark.smoke
def test_materialised_views_can_be_checkpointed(tmp_path: pathlib.Path, dask_cluster) -> None:
    """Checkpointing must survive a view that materialised to pandas.

    `store_view` writes through Dask, so a materialised layer has to be wrapped
    back into a Dask frame first -- and views are indexed by their view type,
    which is a MultiIndex whenever there is more than one. Both facts are easy
    to miss because every other test here disables checkpointing.
    """
    checkpoint_dir = tmp_path / "ckpt" / "checkpoints"

    cold = _analyze(tmp_path / "ckpt", dask_cluster, 256 * 1024**2, checkpoint=True)

    written = sorted(p.name for p in checkpoint_dir.iterdir()) if checkpoint_dir.is_dir() else []
    assert written, "no checkpoints written -- the test would pass without exercising store_view"

    # Reruns hit the read-back path instead of recomputing, so it needs covering too.
    warm = _analyze(tmp_path / "ckpt", dask_cluster, 256 * 1024**2, checkpoint=True)

    baseline = _analyze(tmp_path / "nockpt", dask_cluster, 256 * 1024**2)
    assert set(baseline) == set(cold) == set(warm), "checkpointing changed which views were produced"

    for view_key in baseline:
        # Index types are compared on the cold run only. Restoring from Parquet
        # widens string indexes to `large_string[pyarrow]`, which predates
        # materialisation -- a pure-Dask run (view_materialize_max_bytes=0) does
        # it too -- so requiring it here would assert something never true.
        for label, actual, check_index_type in (("cold", cold, True), ("warm", warm, False)):
            assert_frame_equal(
                baseline[view_key],
                actual[view_key],
                check_dtype=True,
                check_index_type=check_index_type,
                check_exact=False,
                rtol=1e-9,
                obj=f"flat view {view_key} ({label} checkpoint)",
            )


@pytest.mark.smoke
def test_materialize_gate_is_disablable(tmp_path: pathlib.Path, dask_cluster) -> None:
    """0 must keep everything on Dask, so the Dask path stays exercisable."""
    from dftracer.analyzer.analyzer import Analyzer

    seen = []
    original = Analyzer._materialize_if_small

    def spy(self, main_view):
        out = original(self, main_view)
        seen.append(out is not main_view)
        return out

    Analyzer._materialize_if_small = spy
    try:
        _analyze(tmp_path / "off", dask_cluster, 0)
        assert seen and not any(seen), "gate should not have materialised anything when disabled"
        seen.clear()
        _analyze(tmp_path / "on", dask_cluster, 256 * 1024**2)
        assert any(seen), "gate should have materialised at least one layer when enabled"
    finally:
        Analyzer._materialize_if_small = original

import numpy as np
import pandas as pd

from dftracer.analyzer.analyzer import Analyzer


DERIVED_METRICS = {
    "read": "io_cat == 1",
    "write": "io_cat == 2",
    "metadata": "io_cat == 3",
}

SIZE_DERIVED_METRICS = ["read", "write"]


def _build_hlm_df(n_rows: int = 30_000) -> pd.DataFrame:
    io_cat = np.tile(np.array([1, 2, 3, 1, 2], dtype=np.int64), int(np.ceil(n_rows / 5)))[:n_rows]
    idx = np.arange(n_rows, dtype=np.int64)
    return pd.DataFrame(
        {
            "io_cat": io_cat,
            "count": (idx % 17) + 1,
            "time": ((idx % 23) + 1).astype(float),
            "size": ((idx % 101) + 1) * 4096,
            "size_bin_0_4kb": (idx % 2).astype(np.int64),
            "func_name": np.where(io_cat == 1, "read", np.where(io_cat == 2, "write", "metadata")),
        }
    )


def test_set_layer_metrics_correctness() -> None:
    hlm = _build_hlm_df(n_rows=2_000)
    out = Analyzer.set_layer_metrics(
        hlm=hlm,
        derived_metrics=DERIVED_METRICS,
        size_derived_metrics=SIZE_DERIVED_METRICS,
    )

    # Size columns should only be created for metrics explicitly listed in size_derived_metrics.
    assert "read_size" in out.columns
    assert "write_size" in out.columns
    assert "metadata_size" not in out.columns

    read_mask = hlm["io_cat"] == 1
    write_mask = hlm["io_cat"] == 2
    metadata_mask = hlm["io_cat"] == 3

    assert np.allclose(
        out.loc[read_mask, "read_count"].astype(float),
        pd.to_numeric(hlm.loc[read_mask, "count"], errors="coerce").astype(float),
        equal_nan=True,
    )
    assert out.loc[~read_mask, "read_count"].isna().all()

    assert np.allclose(
        out.loc[write_mask, "write_time"].astype(float),
        pd.to_numeric(hlm.loc[write_mask, "time"], errors="coerce").astype(float),
        equal_nan=True,
    )
    assert out.loc[~write_mask, "write_time"].isna().all()

    # String-derived columns carry original values for matching rows and missing values otherwise.
    # Downstream unique_set_flatten skips missing values.
    assert (out.loc[read_mask, "read_func_name"] == hlm.loc[read_mask, "func_name"]).all()
    assert out.loc[~read_mask, "read_func_name"].isna().all()
    assert (out.loc[metadata_mask, "metadata_func_name"] == hlm.loc[metadata_mask, "func_name"]).all()


def test_set_layer_metrics_perf_smoke() -> None:
    hlm = _build_hlm_df(n_rows=50_000)
    out = None
    for _ in range(8):
        out = Analyzer.set_layer_metrics(
            hlm=hlm,
            derived_metrics=DERIVED_METRICS,
            size_derived_metrics=SIZE_DERIVED_METRICS,
        )
    assert out is not None
    assert int(out["read_count"].notna().sum()) > 0

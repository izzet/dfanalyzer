"""Eval 1: Resolution matching throughput at different event scales.

Measures (all pandas-based, no Dask overhead):
  - Profile expansion (sub-bucket splitting) throughput
  - Resolution matching (expand + reconcile) throughput at different scales
  - Full analyze_trace pipeline latency for synthetic and real data
  - Memory footprint (RSS delta)

Usage:
  pytest benchmarks/eval_throughput.py -v -s
  pytest benchmarks/eval_throughput.py -v -s -k "expansion"
  pytest benchmarks/eval_throughput.py -v -s -k "resolution_matching"
  pytest benchmarks/eval_throughput.py -v -s -k "pipeline"
"""

import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.config import AnalyzerPresetConfigDLIOAILogging
from dftracer.analyzer.dftracer import DFTracerAnalyzer

from conftest import (
    DFT_AGG_FULL_DIR,
    DFT_AGG_SELECTIVE_DIR,
    DFT_NORMAL_DIR,
    generate_synthetic_trace,
    make_analyzer,
    requires_data,
    subset_trace_dir,
    write_synthetic_trace,
)

# Number of runs per benchmark. Override via BENCH_RUNS env var.
BENCH_RUNS = int(os.environ.get("BENCH_RUNS", "5"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure(fn):
    """Run fn once, return (result, wall_seconds, peak_memory_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def _measure_n(fn, n=None, warmup=1):
    """Run fn n times (after warmup runs), return (last_result, times, peaks).

    times and peaks are arrays of length n.
    """
    if n is None:
        n = BENCH_RUNS
    # Warmup runs (not measured)
    for _ in range(warmup):
        fn()

    times = []
    peaks = []
    result = None
    for _ in range(n):
        result, elapsed, peak_mb = _measure(fn)
        times.append(elapsed)
        peaks.append(peak_mb)
    return result, np.array(times), np.array(peaks)


def _fmt_stats(times, peaks, n_rows=None):
    """Format mean +/- std for time and memory."""
    t_mean, t_std = times.mean(), times.std()
    p_mean = peaks.mean()
    parts = [f"{t_mean:.3f}s +/- {t_std:.3f}s", f"{p_mean:.1f} MB peak"]
    if n_rows is not None and t_mean > 0:
        parts.append(f"{n_rows / t_mean:,.0f} rows/s")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Micro-benchmark: _expand_profile_buckets at different scales
# ---------------------------------------------------------------------------

EXPANSION_SCALES = [1_000, 10_000, 100_000, 1_000_000]


@pytest.mark.parametrize("n_rows", EXPANSION_SCALES)
def test_expansion_throughput_raw(n_rows):
    """Benchmark _expand_profile_buckets directly on a pandas DataFrame (no dask)."""
    import numpy as np
    from dftracer.analyzer.analyzer import Analyzer

    data = {
        "cat": ["POSIX"] * n_rows,
        "func_name": [f"read_{i % 5}" for i in range(n_rows)],
        "pid": pd.array([1] * n_rows, dtype="Int64"),
        "tid": pd.array([1] * n_rows, dtype="Int64"),
        "epoch": pd.array([0] * n_rows, dtype="Int64"),
        "step": pd.array([0] * n_rows, dtype="Int64"),
        "file_hash": ["f1"] * n_rows,
        "host_hash": ["h1"] * n_rows,
        "file_name": ["/tmp/file.bin"] * n_rows,
        "host_name": ["hostA"] * n_rows,
        "proc_name": ["app#hostA#1#1"] * n_rows,
        "io_cat": pd.array([1] * n_rows, dtype="Int8"),
        "acc_pat": pd.array([0] * n_rows, dtype="Int8"),
        "count": pd.array(np.random.randint(1, 20, n_rows), dtype="Int64"),
        "time": np.random.uniform(0.001, 0.1, n_rows).astype("float64"),
        "size": pd.array(np.random.randint(1024, 65536, n_rows), dtype="Int64"),
        "time_min": np.random.uniform(0.0005, 0.01, n_rows).astype("float64"),
        "time_max": np.random.uniform(0.05, 0.2, n_rows).astype("float64"),
        "size_min": pd.array(np.random.randint(512, 4096, n_rows), dtype="Int64"),
        "size_max": pd.array(np.random.randint(4096, 65536, n_rows), dtype="Int64"),
        "offset_min": pd.array(np.random.randint(0, 1000, n_rows), dtype="Int64"),
        "offset_max": pd.array(np.random.randint(1000, 100000, n_rows), dtype="Int64"),
        "time_range": pd.array(np.arange(n_rows) % 100, dtype="Int64"),
        "time_start": pd.array((np.arange(n_rows) % 100) * 5_000_000, dtype="Int64"),
        "time_end": pd.array(((np.arange(n_rows) % 100) + 1) * 5_000_000, dtype="Int64"),
    }
    pdf = pd.DataFrame(data)

    def do_expand():
        return Analyzer._expand_profile_buckets(
            df=pdf,
            expansion_factor=5,
            time_granularity=1,
            time_resolution=10**6,
            distribution="uniform",
        )

    result, times, peaks = _measure_n(do_expand)

    assert len(result) == n_rows * 5

    print(f"\n  [expand/raw]  {n_rows:>10,} rows → {len(result):>10,} rows | "
          f"{_fmt_stats(times, peaks, n_rows)} (n={len(times)})")


# ---------------------------------------------------------------------------
# Macro-benchmark: full analyze_trace on synthetic data
# ---------------------------------------------------------------------------

SYNTHETIC_SCALES = [
    (100, 100),       # tiny
    (1_000, 1_000),   # small
    (10_000, 10_000), # medium
    (50_000, 50_000), # large
]


@pytest.mark.parametrize("n_profiles,n_traces", SYNTHETIC_SCALES)
def test_synthetic_pipeline_throughput(n_profiles, n_traces, dask_client, tmp_path):
    """Benchmark full analyze_trace pipeline on synthetic hybrid traces."""
    events = generate_synthetic_trace(n_profiles=n_profiles, n_traces=n_traces)
    trace_path = tmp_path / f"synthetic-{n_profiles}-{n_traces}.pfw"
    write_synthetic_trace(trace_path, events)

    analyzer = make_analyzer(tmp_path, time_granularity=5)

    def do_analyze():
        return analyzer.analyze_trace(
            trace_path=str(trace_path),
            view_types=["proc_name", "time_range"],
        )

    result, times, peaks = _measure_n(do_analyze, warmup=0)

    total_events = n_profiles + n_traces
    print(f"\n  [pipeline] {total_events:>10,} events ({n_profiles} C + {n_traces} X) | "
          f"{_fmt_stats(times, peaks)} (n={len(times)})")

    assert result.raw_stats is not None


# ---------------------------------------------------------------------------
# Macro-benchmark: full analyze_trace on real data
# ---------------------------------------------------------------------------

@requires_data
@pytest.mark.parametrize("dataset_name,dataset_dir", [
    ("dft-normal", DFT_NORMAL_DIR),
    ("dft-agg-selective", DFT_AGG_SELECTIVE_DIR),
    ("dft-agg-full", DFT_AGG_FULL_DIR),
])
def test_real_data_pipeline_throughput(dataset_name, dataset_dir, dask_client, tmp_path):
    """Benchmark full analyze_trace pipeline on real UNet3D traces.

    Uses BENCH_N_FILES env var to control subset size (default 4).
    Set BENCH_N_FILES=0 for full dataset.
    """
    trace_dir = subset_trace_dir(dataset_dir, tmp_path, dataset_name)
    analyzer = make_analyzer(tmp_path, time_granularity=5, checkpoint=True)

    def do_analyze():
        return analyzer.analyze_trace(
            trace_path=trace_dir,
            view_types=["proc_name", "time_range"],
        )

    result, times, peaks = _measure_n(do_analyze, warmup=0)

    trace_count = int(result.raw_stats.trace_event_count)
    profile_count = int(result.raw_stats.profile_event_count)
    total_count = int(result.raw_stats.total_event_count)

    print(f"\n  [{dataset_name}] {total_count:>10,} events "
          f"({trace_count:,} X + {profile_count:,} C) | "
          f"{_fmt_stats(times, peaks)} (n={len(times)})")

    assert total_count > 0


# ---------------------------------------------------------------------------
# Resolution matching throughput: expand + reconcile (pure pandas)
# ---------------------------------------------------------------------------

RESOLUTION_SCALES = [
    (5_000, 5_000),       # 10K total
    (50_000, 50_000),     # 100K total
    (500_000, 500_000),   # 1M total
]


def _make_synthetic_hlm(n_rows, prefix="trace", n_funcs=5, n_buckets=100):
    """Create a synthetic HLM-like DataFrame with n_rows."""
    rng = np.random.default_rng(42)
    funcs = [f"{prefix}_{i}" for i in range(n_funcs)]
    return pd.DataFrame({
        "proc_name": [f"app#hostA#1#1"] * n_rows,
        "time_range": pd.array(np.arange(n_rows) % n_buckets, dtype="Int64"),
        "cat": [f"POSIX"] * n_rows,
        "io_cat": pd.array(rng.integers(1, 4, n_rows), dtype="Int8"),
        "acc_pat": pd.array([0] * n_rows, dtype="Int8"),
        "func_name": [funcs[i % n_funcs] for i in range(n_rows)],
        "count": pd.array(rng.integers(1, 100, n_rows), dtype="Int64"),
        "time": rng.uniform(0.001, 1.0, n_rows).astype("float64"),
        "size": pd.array(rng.integers(0, 65536, n_rows), dtype="Int64"),
    })


def _reconcile_hlm_pandas(trace_hlm, profile_hlm, hlm_groupby):
    """Pure-pandas version of Analyzer._reconcile_hlm for benchmarking."""
    hlm_agg = {"time": "sum", "count": "sum", "size": "sum"}

    # Find profile-only rows (not overlapping with trace)
    trace_keys = trace_hlm[hlm_groupby].drop_duplicates().assign(_trace_present=1)
    profile_hlm = profile_hlm.merge(trace_keys, how="left", on=hlm_groupby)
    profile_only = profile_hlm[profile_hlm["_trace_present"].isna()].drop(
        columns=["_trace_present"]
    )

    # Concat and re-aggregate
    combined = pd.concat([trace_hlm, profile_only], ignore_index=True)
    return combined.groupby(hlm_groupby).agg(hlm_agg).reset_index()


@pytest.mark.parametrize("n_profiles,n_traces", RESOLUTION_SCALES)
def test_resolution_matching_throughput(n_profiles, n_traces):
    """Benchmark resolution matching (expand + reconcile) at different scales.

    Pure pandas, no Dask, no full pipeline. Measures the cost of:
    1. Expanding profile rows from 5s → 1s buckets (5x expansion)
    2. Reconciling expanded profile HLM with trace HLM
    """
    from dftracer.analyzer.analyzer import Analyzer

    # Create synthetic HLM-like data
    trace_hlm = _make_synthetic_hlm(n_traces, prefix="read", n_funcs=5)
    profile_df = _make_synthetic_hlm(n_profiles, prefix="write", n_funcs=5)

    # Add columns needed for expansion
    profile_df["time_start"] = pd.array(
        (profile_df["time_range"].to_numpy(dtype=np.int64) * 5_000_000), dtype="Int64"
    )
    profile_df["time_end"] = pd.array(
        profile_df["time_start"].to_numpy(dtype=np.int64) + 5_000_000, dtype="Int64"
    )
    # Add min/max columns for expansion
    profile_df["time_min"] = profile_df["time"] * 0.1
    profile_df["time_max"] = profile_df["time"] * 2.0

    hlm_groupby = ["proc_name", "time_range", "cat", "io_cat", "acc_pat", "func_name"]

    def do_resolve():
        # Step 1: Expand profiles 5s → 1s (5x)
        expanded = Analyzer._expand_profile_buckets(
            df=profile_df,
            expansion_factor=5,
            time_granularity=1,
            time_resolution=10**6,
            distribution="uniform",
        )
        # Step 2: Reconcile expanded profiles with trace HLM
        return _reconcile_hlm_pandas(trace_hlm, expanded, hlm_groupby)

    result, times, peaks = _measure_n(do_resolve)

    total_events = n_profiles + n_traces
    expanded_profiles = n_profiles * 5
    print(f"\n  [resolve] {total_events:>10,} events "
          f"({n_profiles:,} profiles →{expanded_profiles:,} expanded + {n_traces:,} traces) | "
          f"{_fmt_stats(times, peaks, total_events)} (n={len(times)})")

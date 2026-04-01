"""Shared fixtures for evaluation benchmarks."""

import json
import os

import pytest
from dask.distributed import Client, LocalCluster
from omegaconf import OmegaConf

from dftracer.analyzer.config import (
    AnalyzerPresetConfigDLIOAILogging,
    AnalyzerPresetConfigPOSIX,
)
from dftracer.analyzer.dftracer import DFTracerAnalyzer


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATA_ROOT = "/p/lustre5/iopp/rayandrew/dfprofiler/results/unet3d"

# dft-normal latest symlink is broken; use the known-good directory
DFT_NORMAL_DIR = os.path.join(
    DATA_ROOT, "dft-normal", "2026-02-28-19-00-40-1269894-0ddL"
)
DFT_AGG_SELECTIVE_DIR = os.path.join(DATA_ROOT, "dft-agg-selective", "latest")
DFT_AGG_FULL_DIR = os.path.join(DATA_ROOT, "dft-agg-full", "latest")


def data_available():
    """Check if the real UNet3D trace data is accessible."""
    return os.path.isdir(DFT_NORMAL_DIR) and os.path.isdir(DFT_AGG_SELECTIVE_DIR)


requires_data = pytest.mark.skipif(
    not data_available(),
    reason=f"UNet3D trace data not available at {DATA_ROOT}",
)

# Number of trace files to use in real-data benchmarks.
# Override via BENCH_N_FILES env var.  Default is 4 (~2 min per dataset).
BENCH_N_FILES = int(os.environ.get("BENCH_N_FILES", "4"))


MIN_TRACE_FILE_BYTES = 500_000  # filter out non-worker processes (~10KB normal, ~1KB selective)


def subset_trace_dir(dataset_dir, tmp_path, label, n_files=None):
    """Create a temp directory with symlinks to the first n_files traces.

    Filters out small files (non-worker processes) before subsetting so that
    comparisons across different dataset runs are apples-to-apples.
    Returns the original dir if n_files is None or >= total worker files.
    """
    import glob
    if n_files is None:
        n_files = BENCH_N_FILES
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*-app.pfw.gz")))
    # Keep only worker-sized files for fair cross-run comparison
    files = [f for f in all_files if os.path.getsize(f) >= MIN_TRACE_FILE_BYTES]
    if n_files <= 0 or n_files >= len(files):
        # Use all worker files — symlink into subset dir to exclude small files
        if len(files) == len(all_files):
            return dataset_dir
        subset_dir = tmp_path / f"subset_{label}"
        subset_dir.mkdir(exist_ok=True)
        for f in files:
            link = subset_dir / os.path.basename(f)
            if not link.exists():
                link.symlink_to(f)
        return str(subset_dir)
    subset_dir = tmp_path / f"subset_{label}"
    subset_dir.mkdir(exist_ok=True)
    for f in files[:n_files]:
        link = subset_dir / os.path.basename(f)
        if not link.exists():
            link.symlink_to(f)
    return str(subset_dir)


# ---------------------------------------------------------------------------
# Dask cluster
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dask_client():
    import logging
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        processes=False,
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


# ---------------------------------------------------------------------------
# Analyzer factory
# ---------------------------------------------------------------------------

def make_analyzer(
    tmp_path,
    time_granularity=5,
    preset=None,
    profile_distribution="uniform",
    profile_time_granularity=5,
    checkpoint=False,
):
    return DFTracerAnalyzer(
        preset=OmegaConf.structured(preset or AnalyzerPresetConfigDLIOAILogging()),
        checkpoint=checkpoint,
        checkpoint_dir=str(tmp_path / "checkpoints") if tmp_path else "/tmp/benchmarks",
        debug=False,
        profile_distribution=profile_distribution,
        profile_time_granularity=profile_time_granularity,
        quantile_stats=False,
        time_approximate=True,
        time_granularity=time_granularity,
        time_resolution=10**6,
        time_sliced=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Synthetic trace generation
# ---------------------------------------------------------------------------

METADATA_EVENTS = [
    {"id": 1, "name": "HH", "cat": "dftracer", "pid": 1, "tid": 1,
     "ph": "M", "args": {"hhash": "h1", "name": "hostA", "value": "h1"}},
    {"id": 2, "name": "FH", "cat": "dftracer", "pid": 1, "tid": 1,
     "ph": "M", "args": {"hhash": "h1", "name": "/tmp/data/file.bin", "value": "f1"}},
]


def generate_synthetic_trace(n_profiles, n_traces=0, bucket_width_us=5_000_000):
    """Generate a synthetic trace with n_profiles counter events and n_traces duration events.

    Returns a list of JSON event dicts (without the wrapping brackets).
    """
    events = list(METADATA_EVENTS)
    event_id = len(events) + 1

    func_names = ["read", "write", "open64", "close", "stat"]
    for i in range(n_traces):
        ts = 5_000_000 + i * 100
        events.append({
            "id": event_id, "name": func_names[i % len(func_names)],
            "cat": "POSIX", "pid": 1, "tid": 1, "ph": "X",
            "ts": ts, "dur": 50 + (i % 200),
            "args": {"hhash": "h1", "fhash": "f1", "ret": 4096, "offset": i * 4096},
        })
        event_id += 1

    for i in range(n_profiles):
        bucket_idx = i % 20  # spread across 20 buckets (100s of simulated time)
        ts = 5_000_000 + bucket_idx * bucket_width_us
        events.append({
            "id": event_id, "name": func_names[i % len(func_names)],
            "cat": "POSIX", "pid": 1, "tid": 1, "ph": "C",
            "ts": ts,
            "args": {
                "hhash": "h1", "fhash": "f1",
                "dft_cnt": 10 + (i % 5),
                "dur_sum": 1000 + i * 10,
                "dur_min": 50 + (i % 30),
                "dur_max": 200 + (i % 50),
                "ret_sum": 40960,
                "ret_min": 4096,
                "ret_max": 4096,
            },
        })
        event_id += 1

    return events


def write_synthetic_trace(path, events):
    """Write events to a .pfw file."""
    lines = ["["] + [json.dumps(e) for e in events] + ["]"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Tests for TYPE_SYSTEM event parsing and system metrics integration."""

import json
import logging
import os
import pandas as pd
import pytest
from dask.distributed import Client, LocalCluster
from omegaconf import OmegaConf

from dftracer.analyzer.config import AnalyzerPresetConfigPOSIX
from dftracer.analyzer.constants import COL_TIME_RANGE
from dftracer.analyzer.dftracer import (
    DFTracerAnalyzer,
    SYSTEM_OUTPUT_COLUMNS,
    TYPE_SYSTEM,
)

pytestmark = [pytest.mark.smoke, pytest.mark.full]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYSTEM_TRACE_DIR = os.path.join(
    os.path.dirname(__file__), "data", "extracted", "dftracer-system"
)


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


def _make_analyzer(tmp_path, time_granularity=5):
    return DFTracerAnalyzer(
        preset=OmegaConf.structured(AnalyzerPresetConfigPOSIX()),
        checkpoint=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        debug=False,
        time_granularity=time_granularity,
        time_resolution=10**6,
        time_approximate=True,
        time_sliced=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Tests: parsing
# ---------------------------------------------------------------------------


class TestSystemEventParsing:
    """Test that cat='sys' ph='C' events are parsed as TYPE_SYSTEM."""

    def test_read_trace_returns_system_metrics(self, dask_client, tmp_path):
        analyzer = _make_analyzer(tmp_path)
        result = analyzer.read_trace(
            trace_path=SYSTEM_TRACE_DIR,
            extra_columns=None,
            extra_columns_fn=None,
        )
        assert result.system_metrics is not None
        sm = result.system_metrics.compute()
        assert not sm.empty

    def test_system_metrics_has_expected_columns(self, dask_client, tmp_path):
        analyzer = _make_analyzer(tmp_path / "cols")
        result = analyzer.read_trace(
            trace_path=SYSTEM_TRACE_DIR,
            extra_columns=None,
            extra_columns_fn=None,
        )
        sm = result.system_metrics.compute()
        for col in SYSTEM_OUTPUT_COLUMNS:
            assert col in sm.columns, f"Missing column: {col}"

    def test_no_traces_or_profiles_from_system_only_file(self, dask_client, tmp_path):
        analyzer = _make_analyzer(tmp_path / "noprof")
        result = analyzer.read_trace(
            trace_path=SYSTEM_TRACE_DIR,
            extra_columns=None,
            extra_columns_fn=None,
        )
        traces = result.traces.compute()
        assert traces.empty
        assert result.profiles is None


# ---------------------------------------------------------------------------
# Tests: standardization
# ---------------------------------------------------------------------------


class TestSystemMetricsStandardization:
    """Test that raw system events are correctly aggregated per time_range."""

    @pytest.fixture(autouse=True)
    def setup(self, dask_client, tmp_path):
        analyzer = _make_analyzer(tmp_path / "std")
        result = analyzer.read_trace(
            trace_path=SYSTEM_TRACE_DIR,
            extra_columns=None,
            extra_columns_fn=None,
        )
        self.sm = result.system_metrics.compute()

    def test_rows_per_time_range(self):
        # 10.5s of data at 5s granularity => 3 time_range buckets
        assert len(self.sm) == 3

    def test_time_ranges_are_sequential(self):
        trs = self.sm[COL_TIME_RANGE].sort_values().values
        assert trs[1] == trs[0] + 1
        assert trs[2] == trs[1] + 1

    def test_host_hash_present(self):
        assert (self.sm["host_hash"] == "731340f16729098d").all()

    def test_cpu_percentages_in_range(self):
        for col in ["sys_cpu_iowait_pct", "sys_cpu_user_pct", "sys_cpu_system_pct", "sys_cpu_idle_pct"]:
            assert (self.sm[col] >= 0).all(), f"{col} has negative values"
            assert (self.sm[col] <= 100).all(), f"{col} exceeds 100%"

    def test_cpu_percentages_sum_reasonable(self):
        # user + system + iowait + idle should be close to 100%
        total = (
            self.sm["sys_cpu_user_pct"]
            + self.sm["sys_cpu_system_pct"]
            + self.sm["sys_cpu_iowait_pct"]
            + self.sm["sys_cpu_idle_pct"]
        )
        for val in total:
            assert 99.0 <= val <= 101.0, f"CPU pct sum out of range: {val}"

    def test_core_max_ge_aggregate(self):
        # Per-core max iowait should be >= aggregate mean iowait
        assert (self.sm["sys_core_iowait_pct_max"] >= self.sm["sys_cpu_iowait_pct"]).all()

    def test_core_p95_between_mean_and_max(self):
        assert (self.sm["sys_core_iowait_pct_p95"] >= self.sm["sys_cpu_iowait_pct"]).all()
        assert (self.sm["sys_core_iowait_pct_p95"] <= self.sm["sys_core_iowait_pct_max"]).all()

    def test_memory_available_positive(self):
        assert (self.sm["sys_mem_available"] > 0).all()


# ---------------------------------------------------------------------------
# Tests: mixed trace (system + I/O events)
# ---------------------------------------------------------------------------


class TestMixedTrace:
    """Test system metrics with a synthetic trace containing both I/O and system events."""

    @pytest.fixture(autouse=True)
    def setup(self, dask_client, tmp_path):
        self.trace_dir = tmp_path / "mixed"
        self.trace_dir.mkdir()

        events = [
            # Metadata
            {"name": "HH", "cat": "dftracer", "pid": 1, "tid": 1, "ph": "M",
             "args": {"hhash": "h1", "name": "hostA", "value": "h1"}},
            {"name": "FH", "cat": "dftracer", "pid": 1, "tid": 1, "ph": "M",
             "args": {"hhash": "h1", "name": "/tmp/file.bin", "value": "f1"}},
            # I/O trace event
            {"name": "read", "cat": "POSIX", "pid": 1, "tid": 1, "ph": "X",
             "ts": 5_000_100, "dur": 200,
             "args": {"hhash": "h1", "fhash": "f1", "ret": 4096, "offset": 0}},
            # System events (same time window)
            {"name": "cpu", "cat": "sys", "pid": 0, "tid": 0, "ph": "C",
             "ts": 5_000_000,
             "args": {"hhash": "h1", "user_pct": 10.0, "system_pct": 5.0,
                       "iowait_pct": 3.0, "idle_pct": 82.0}},
            {"name": "cpu-0", "cat": "sys", "pid": 0, "tid": 0, "ph": "C",
             "ts": 5_000_000,
             "args": {"hhash": "h1", "user_pct": 20.0, "system_pct": 8.0,
                       "iowait_pct": 7.0, "idle_pct": 65.0}},
            {"name": "cpu-1", "cat": "sys", "pid": 0, "tid": 0, "ph": "C",
             "ts": 5_000_000,
             "args": {"hhash": "h1", "user_pct": 5.0, "system_pct": 2.0,
                       "iowait_pct": 1.0, "idle_pct": 92.0}},
            {"name": "memory", "cat": "sys", "pid": 0, "tid": 0, "ph": "C",
             "ts": 5_000_000,
             "args": {"hhash": "h1", "MemAvailable": 1000000, "Dirty": 0.5,
                       "Cached": 10.0}},
        ]

        trace_file = self.trace_dir / "mixed_trace.pfw"
        lines = ["["] + [json.dumps(e) for e in events] + ["]"]
        trace_file.write_text("\n".join(lines) + "\n")

        analyzer = _make_analyzer(tmp_path / "mixed_analysis")
        self.result = analyzer.read_trace(
            trace_path=str(self.trace_dir),
            extra_columns=None,
            extra_columns_fn=None,
        )

    def test_traces_and_system_both_present(self):
        traces = self.result.traces.compute()
        assert not traces.empty, "Should have I/O trace events"
        assert self.result.system_metrics is not None, "Should have system metrics"

    def test_system_metrics_values(self):
        sm = self.result.system_metrics.compute()
        assert len(sm) == 1  # one time_range bucket
        row = sm.iloc[0]
        assert row["sys_cpu_user_pct"] == pytest.approx(10.0)
        assert row["sys_cpu_iowait_pct"] == pytest.approx(3.0)
        assert row["sys_core_iowait_pct_max"] == pytest.approx(7.0)  # cpu-0
        assert row["sys_mem_available"] == pytest.approx(1000000)
        assert row["sys_mem_dirty"] == pytest.approx(0.5)

    def test_profiles_not_contaminated(self):
        """System events should NOT appear as profiles."""
        assert self.result.profiles is None


# ---------------------------------------------------------------------------
# Tests: variable time granularity
# ---------------------------------------------------------------------------


class TestVariableTimeGranularity:
    """Test that system metrics aggregate correctly with different time_granularity."""

    def test_granularity_1s(self, dask_client, tmp_path):
        analyzer = _make_analyzer(tmp_path / "g1", time_granularity=1)
        result = analyzer.read_trace(
            trace_path=SYSTEM_TRACE_DIR,
            extra_columns=None,
            extra_columns_fn=None,
        )
        sm = result.system_metrics.compute()
        # 10.5s at 1s granularity => ~11 buckets (some may have 0-1 samples)
        assert len(sm) >= 10
        assert len(sm) <= 12

    def test_granularity_10s(self, dask_client, tmp_path):
        analyzer = _make_analyzer(tmp_path / "g10", time_granularity=10)
        result = analyzer.read_trace(
            trace_path=SYSTEM_TRACE_DIR,
            extra_columns=None,
            extra_columns_fn=None,
        )
        sm = result.system_metrics.compute()
        # 10.5s at 10s granularity => 2 buckets
        assert len(sm) == 2

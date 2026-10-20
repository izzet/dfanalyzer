import re

import pytest

from tests.test_preset_detection import run_cli, write_flavor_trace

pytestmark = [pytest.mark.smoke, pytest.mark.full]


def trace_count(stdout):
    match = re.search(r"Trace Count\s*│\s*count\s*│\s*([\d,]+)", stdout)
    assert match, stdout[-2000:]
    return int(match.group(1).replace(",", ""))


def test_changing_time_granularity_keeps_event_counts(tmp_path):
    trace = write_flavor_trace(tmp_path / "trace", "dlio")
    counts = []
    for granularity in (1, 10, 1, 5):
        result = run_cli(trace, tmp_path / f"run-{len(counts)}", f"analyzer.time_granularity={granularity}")
        assert result.returncode == 0, result.stderr[-2000:]
        counts.append(trace_count(result.stdout))
    assert len(set(counts)) == 1, counts

"""Per-step evidence of a window (liveflow-v1): the critical rank's step_series and the node's counters reach the
window metrics (liveflow_v1.window_step_evidence, forwarded by analyzer.window_horizon_metrics)."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ANALYZER_DIR = Path(__file__).resolve().parents[1] / "python" / "dftracer" / "analyzer"
_pkg = types.ModuleType("dftracer")
_pkg.__path__ = [str(_ANALYZER_DIR.parent)]
sys.modules.setdefault("dftracer", _pkg)
_apkg = types.ModuleType("dftracer.analyzer")
_apkg.__path__ = [str(_ANALYZER_DIR)]
sys.modules.setdefault("dftracer.analyzer", _apkg)
_spec = importlib.util.spec_from_file_location("dftracer.analyzer.liveflow_v1", _ANALYZER_DIR / "liveflow_v1.py")
lf = importlib.util.module_from_spec(_spec)
sys.modules["dftracer.analyzer.liveflow_v1"] = lf
_spec.loader.exec_module(lf)


def _stop(**kw):
    ev = {"name": "window.stop", "configured_total_steps": 4800, "completed_steps": 800,
          "configured_window_steps": 331, "planner_history_ready": 1}
    ev.update(kw)
    return ev


def test_critical_rank_series_and_node_counters_are_selected():
    slow = "v1;first=470;n=3;b=1;wall=9000,9100,9200;storage=10,10,10"
    fast = "v1;first=470;n=3;b=1;wall=8000,8000,8000;storage=1,1,1"
    ev = [_stop(step_series=fast, vm_allocstall=5, vm_pgscan_direct=7, vm_compact_stall=1, mem_anon_kb=100, mem_avail_kb=900),
          _stop(step_series=slow, vm_allocstall=6, vm_pgscan_direct=7, vm_compact_stall=1, mem_anon_kb=120, mem_avail_kb=800)]
    got = lf.window_step_evidence(ev)
    assert got["step_series"] == slow                                    # largest total step wall = critical rank
    assert got["vm_allocstall"] == 6 and got["mem_anon_kb"] == 120 and got["mem_avail_kb"] == 800   # min for available
    assert set(lf.STEP_EVIDENCE_FIELDS) == {"step_series", "vm_allocstall", "vm_pgscan_direct", "vm_compact_stall",
                                            "mem_anon_kb", "mem_avail_kb"}


def test_malformed_or_absent_series_are_not_invented():
    assert lf.window_step_evidence([_stop()]) == {}
    assert "step_series" not in lf.window_step_evidence([_stop(step_series="garbage;wall=x")])
    assert lf.window_step_evidence([_stop(step_series=5)]) == {}


def test_window_horizon_metrics_forwards_the_evidence():
    import dftracer
    if str(_ANALYZER_DIR.parent) not in getattr(dftracer, "__path__", []):
        pytest.skip("run against the staged runtime (installed analyzer predates step evidence)")
    from dftracer.analyzer import analyzer
    if not hasattr(analyzer, "window_step_evidence"):
        pytest.skip("installed analyzer predates step evidence; run from the staged runtime")
    series = "v1;first=470;n=2;b=1;wall=9000,9100"
    metrics = analyzer.window_horizon_metrics([_stop(step_series=series, vm_allocstall=3)])
    assert metrics["step_series"] == series and metrics["vm_allocstall"] == 3 and metrics["remaining_steps"] == 4000

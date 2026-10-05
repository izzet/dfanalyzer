"""Regression guard + behavior test for the 2026-09-27 always-on view_keys diagnostic.

FactEngine.evaluate() must log the runtime flat_views keys and per-view row counts on every
call, unconditionally (no --debug dependency, no gate on self.rules or on whether any rule
matched), so a future run can directly answer "did the ('proc_name',) view even reach
evaluate()?" without relying on DEBUG-level facts.rule.* tracing. See
liveflow_shm/STORMER-LOOP-DESIGN-2026-09-26.md sec 18 continuation.
"""
import importlib.util
import re
import sys
import types
from pathlib import Path

import pandas as pd

_ANALYZER_DIR = Path(__file__).resolve().parents[1] / "python" / "dftracer" / "analyzer"
FACT_ENGINE_PY = _ANALYZER_DIR / "fact_engine.py"
_RULE_FILE = _ANALYZER_DIR / "configs" / "fact_rules" / "liveflow-stormer-loop.yaml"

_dftracer_pkg = types.ModuleType("dftracer")
_dftracer_pkg.__path__ = [str(_ANALYZER_DIR.parent)]
sys.modules["dftracer"] = _dftracer_pkg
_analyzer_pkg = types.ModuleType("dftracer.analyzer")
_analyzer_pkg.__path__ = [str(_ANALYZER_DIR)]
sys.modules["dftracer.analyzer"] = _analyzer_pkg


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(
        modname, _ANALYZER_DIR / filename, submodule_search_locations=[str(_ANALYZER_DIR)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_load("dftracer.analyzer.types", "types.py")
_load("dftracer.analyzer.fact_rules", "fact_rules.py")
fact_engine_mod = _load("dftracer.analyzer.fact_engine", "fact_engine.py")

FactEngine = fact_engine_mod.FactEngine


def test_evaluate_logs_view_keys_before_the_no_rules_early_return():
    """Static check: the new log call must appear BEFORE `if not self.rules: return {}`,
    so it fires even when there are zero loaded rules -- the most silent failure mode."""
    text = FACT_ENGINE_PY.read_text()
    m = re.search(r"def evaluate\(", text)
    assert m is not None
    body = text[m.start():]
    log_idx = body.index('"facts.evaluate.view_keys"')
    guard_idx = body.index("if not self.rules:")
    assert log_idx < guard_idx, (
        "facts.evaluate.view_keys must be logged before the empty-rules early return, "
        "or a run with zero loaded rules would show total silence again"
    )


def test_evaluate_logs_view_keys_with_empty_flat_views(monkeypatch):
    """Behavioral check: call evaluate() with an empty flat_views dict and confirm the new
    log line fires with an empty view_keys list, rather than being skipped by some
    truthiness check on flat_views itself."""
    eng = FactEngine.from_rule_file(
        str(_RULE_FILE), strict_time_semantics=False, allow_mixed_time_aggregates=True
    )

    captured = []

    class _Capture:
        def info(self, event, **kwargs):
            captured.append((event, kwargs))

        def __getattr__(self, name):
            def _noop(*a, **k):
                pass
            return _noop

    monkeypatch.setattr(fact_engine_mod, "logger", _Capture())

    result = eng.evaluate(flat_views={}, raw_stats={})
    assert result == {}
    events = [e for e, _ in captured if e == "facts.evaluate.view_keys"]
    assert len(events) == 1, "expected exactly one facts.evaluate.view_keys log call"
    payload = next(p for e, p in captured if e == "facts.evaluate.view_keys")
    assert payload["view_keys"] == []
    assert payload["view_row_counts"] == {}
    assert set(payload["rules_by_view_keys"]) == {"proc_name"}


def test_evaluate_logs_view_keys_and_row_counts_with_real_views(monkeypatch):
    """Behavioral check: with a non-empty flat_views dict, the log line reports the actual
    keys present and their row counts -- this is exactly what a future run needs to answer
    whether ('proc_name',) reached evaluate() at all."""
    eng = FactEngine.from_rule_file(
        str(_RULE_FILE), strict_time_semantics=False, allow_mixed_time_aggregates=True
    )

    captured = []

    class _Capture:
        def info(self, event, **kwargs):
            captured.append((event, kwargs))

        def __getattr__(self, name):
            def _noop(*a, **k):
                pass
            return _noop

    monkeypatch.setattr(fact_engine_mod, "logger", _Capture())

    flat_views = {
        ("window",): pd.DataFrame({"a": [1, 2, 3]}),
        ("proc_name",): pd.DataFrame({"b": [1, 2]}),
    }
    eng.evaluate(flat_views=flat_views, raw_stats={})
    payload = next(p for e, p in captured if e == "facts.evaluate.view_keys")
    assert set(payload["view_keys"]) == {("window",), ("proc_name",)}
    assert payload["view_row_counts"] == {
        str(("window",)): 3,
        str(("proc_name",)): 2,
    }

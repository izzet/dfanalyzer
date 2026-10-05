"""Static regression guard for the 2026-09-28 identity/state diagnostic (patch 14, follow-up to
patch 13's facts.evaluate.view_keys line). The v4.5.7 rerun proved that line never fires despite
FastPathPlan.build() logging a correctly-populated fact_engine (2 rules) 300ms earlier on the
same window. By elimination, the break is one of _evaluate_analysis_facts()'s two silent-capable
early returns (`self.fact_engine is None` has no logging at all; the emit_analysis_facts branch
is DEBUG-only and also never fired). This test statically confirms the new unconditional INFO
line is present, before both early returns, and that fastpath.py's existing enabled log now also
carries a comparable fact_engine_id so the two can be joined by id() in the next run's log.
"""
import re
from pathlib import Path

_ANALYZER_DIR = Path(__file__).resolve().parents[1] / "python" / "dftracer" / "analyzer"
ANALYZER_PY = _ANALYZER_DIR / "analyzer.py"
FASTPATH_PY = _ANALYZER_DIR / "fastpath.py"


def test_evaluate_analysis_facts_logs_entry_state_before_both_early_returns():
    text = ANALYZER_PY.read_text()
    m = re.search(r"def _evaluate_analysis_facts\(", text)
    assert m is not None
    body = text[m.start():]
    log_idx = body.index('"facts.evaluate_analysis_facts.entry"')
    fact_engine_guard_idx = body.index("if self.fact_engine is None:")
    emit_guard_idx = body.index("if not self.facts_config.emit_analysis_facts:")
    assert log_idx < fact_engine_guard_idx
    assert log_idx < emit_guard_idx


def test_evaluate_analysis_facts_entry_reports_fact_engine_identity_and_emit_flag():
    text = ANALYZER_PY.read_text()
    m = re.search(r"def _evaluate_analysis_facts\(", text)
    body = text[m.start():]
    call_end = body.index('"facts.evaluate_analysis_facts.entry"')
    window = body[call_end:call_end + 600]
    assert "fact_engine_is_none=" in window
    assert "fact_engine_id=" in window
    assert "emit_analysis_facts=" in window
    assert "analyzer_id=" in window


def test_fastpath_enabled_now_reports_fact_engine_id_for_comparison():
    text = FASTPATH_PY.read_text()
    idx = text.index('"analyzer.fastpath.enabled"')
    window = text[idx:idx + 400]
    assert "fact_engine_id=id(fact_engine)" in window, (
        "analyzer.fastpath.enabled must carry the same fact_engine_id=id(...) shape as "
        "_evaluate_analysis_facts.entry so the two can be joined in a run's log to check "
        "whether self.fact_engine is literally a different object by window-processing time"
    )

"""Regression for deep DFTracer nesting in streaming ingestion."""

import pandas as pd

from dftracer.analyzer.dftracer import DFTracerAnalyzer


def test_trace_level_over_int8_range_is_representable():
    """A realistic 32-rank loader can emit nesting level 132."""
    analyzer = object.__new__(DFTracerAnalyzer)
    analyzer.time_approximate = False
    columns = analyzer._get_columns(None)

    assert columns["level"] == "Int32"
    frame = pd.DataFrame({"level": [132]}).astype({"level": columns["level"]})
    assert frame.loc[0, "level"] == 132


def test_live_window_aliases_over_int8_range_are_representable():
    """A control window is also the compatibility epoch/step value."""
    columns = {"epoch": "Int32", "step": "Int32", "window": "Int32"}
    frame = pd.DataFrame({name: pd.Series([132], dtype=dtype) for name, dtype in columns.items()})

    assert frame.to_dict("list") == {"epoch": [132], "step": [132], "window": [132]}

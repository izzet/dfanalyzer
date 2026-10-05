"""The DLIO preset's reader_decode layer must select TF's `data.decode` events (and still `decode`)."""
import os
import sys

import pandas as pd

import importlib.util

# Load THIS tree's config.py as a stand-alone package: an installed or staged dftracer.analyzer may otherwise win the import.
import types
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "dftracer", "analyzer")
_pkg = types.ModuleType("lf_an_under_test")
_pkg.__path__ = [_DIR]
sys.modules["lf_an_under_test"] = _pkg
_spec = importlib.util.spec_from_file_location("lf_an_under_test.config", os.path.join(_DIR, "config.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
AnalyzerPresetConfigDLIOAILogging = _mod.AnalyzerPresetConfigDLIOAILogging


def test_reader_decode_layer_selects_tf_and_two_stage_names():
    cond = AnalyzerPresetConfigDLIOAILogging().layer_defs["reader_decode"]
    df = pd.DataFrame({
        "cat": ["data", "data", "data", "posix", "compute"],
        "func_name": ["data.decode", "decode", "data.read", "data.decode", "decode"],
        "dur": [1, 2, 4, 8, 16],
    })
    assert df.query(cond)["dur"].tolist() == [1, 2]


if __name__ == "__main__":
    test_reader_decode_layer_selects_tf_and_two_stage_names()
    print("PASS test_reader_decode_layer_selects_tf_and_two_stage_names")

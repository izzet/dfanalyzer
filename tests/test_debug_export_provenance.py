import json

import pandas as pd

from dftracer.analyzer.analyzer import Analyzer


class _DebugExportHarness:
    _debug_artifact_suffix = Analyzer._debug_artifact_suffix

    def __init__(self, output_dir, provenance):
        self.debug_output_dir = str(output_dir)
        self._debug_window_provenance = provenance


def test_debug_flat_view_preserves_index_and_online_window_provenance(tmp_path):
    harness = _DebugExportHarness(
        tmp_path,
        {
            "window_index": 7,
            "rank_pids": [101, 102],
            "boundary_ts_ns_by_pid": {"101": 10, "102": 12},
        },
    )
    view = pd.DataFrame({"reader_posix_ops_slope": [0.25, 1.5]}, index=pd.Index([3, 4], name="epoch"))

    Analyzer._write_debug_flat_view(harness, view, ("epoch",))

    exported = pd.read_csv(tmp_path / "flat_view_epoch_window_000007.csv")
    assert exported["__liveflow_window_index"].tolist() == [7, 7]
    assert json.loads(exported["__liveflow_rank_pids"].iloc[0]) == [101, 102]
    assert json.loads(exported["__liveflow_boundary_ts_ns_by_pid"].iloc[0]) == {"101": 10, "102": 12}
    assert json.loads(exported["__dfanalyzer_view_key"].iloc[0]) == ["epoch"]
    assert exported["__dfanalyzer_index_0_epoch"].tolist() == [3, 4]
    assert exported["reader_posix_ops_slope"].tolist() == [0.25, 1.5]

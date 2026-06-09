"""Parity between the two trace-transform paths.

The distributed HLM (``dftracer.utils.dfanalyzer.worker_hlm_partial``) reads raw
IPC and bypasses the analyzer's ``postread_trace``, so it re-implements the same
row transforms on an Arrow table. Historically those two paths silently drifted
(POSIX sub-layer categorization, ignored-func/file filtering, and time-range
bucket origin were all missing from the HLM path), producing wrong per-layer
counts/times.
"""

import pandas as pd
import pyarrow as pa
import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.full]
pytest.importorskip("dftracer.utils.dfanalyzer")
pytest.importorskip("dftracer.analyzer.dftracer")

from dftracer.analyzer.dftracer import (  # noqa: E402
    DFTracerAnalyzer,
    IGNORED_FILE_PATTERNS,
    IGNORED_FUNC_NAMES,
    IGNORED_FUNC_PATTERNS,
)
from dftracer.utils.dfanalyzer import (  # noqa: E402
    _apply_hlm_filters,
    _augment_posix_cat,
    _rebucket_time_range,
)

POSIX_CAT_RULES = [list(rule) for rule in DFTracerAnalyzer.POSIX_CAT_RULES]

# rows exercise: /data->_reader, /checkpoint->_checkpoint, /lustre->_lustre,
# ignored func name (exact), ignored func pattern (substring), ignored file,
# null file_name, and non-posix rows that must be untouched.
ROWS = [
    # rid, cat, func_name, file_name
    (0, "posix", "read", "/p/lustre/data/train/img.npz"),  # -> posix_reader_lustre
    (1, "posix", "write", "/p/lustre/checkpoint/ck.bin"),  # -> posix_checkpoint_lustre
    (2, "posix", "read", "/scratch/data/x.npz"),  # -> posix_reader
    (3, "posix", "open", "/ssd/misc/y"),  # -> posix_ssd
    (4, "reader", "NPZReader.open", ""),  # kept, cat untouched
    (5, "reader", "NPZReader.next", ""),  # dropped: matches "Reader.next"
    (6, "data", "item", ""),  # kept, cat untouched
    (7, "posix", "read", "/usr/lib/python/site.py"),  # dropped: ignored file
    (8, "compute", "DLIOBenchmark.initialize", ""),  # dropped: ignored func name
    (9, "checkpoint", "Checkpointing.finalize", ""),  # dropped: matches func pattern
    (10, "posix", "fread", None),  # null file_name, cat untouched
]


def _make_pandas():
    # Real traces reach postread as nullable "string" dtype (normalize_arrow_dtypes).
    return pd.DataFrame(
        {
            "rid": [r[0] for r in ROWS],
            "cat": pd.array([r[1] for r in ROWS], dtype="string"),
            "func_name": pd.array([r[2] for r in ROWS], dtype="string"),
            "file_name": pd.array([r[3] for r in ROWS], dtype="string"),
        }
    )


def _make_arrow():
    return pa.table(
        {
            "rid": pa.array([r[0] for r in ROWS], pa.int64()),
            "cat": pa.array([r[1] for r in ROWS], pa.string()),
            "func_name": pa.array([r[2] for r in ROWS], pa.string()),
            "file_name": pa.array([r[3] for r in ROWS], pa.string()),
        }
    )


def test_filter_and_categorize_parity():
    # pandas (trace-view) path
    pdf = _make_pandas()
    pdf = DFTracerAnalyzer._apply_ignore_filters(pdf, IGNORED_FILE_PATTERNS, IGNORED_FUNC_NAMES, IGNORED_FUNC_PATTERNS)
    pdf = DFTracerAnalyzer._fix_file_posix_category(pdf)
    pandas_out = dict(zip(pdf["rid"], pdf["cat"]))

    # Arrow (distributed HLM) path
    tbl = _make_arrow()
    tbl = _apply_hlm_filters(tbl, IGNORED_FILE_PATTERNS, IGNORED_FUNC_NAMES, IGNORED_FUNC_PATTERNS)
    tbl = _augment_posix_cat(tbl, POSIX_CAT_RULES)
    adf = tbl.to_pandas()
    arrow_out = dict(zip(adf["rid"], adf["cat"]))

    assert arrow_out == pandas_out, f"paths diverged:\n  arrow={arrow_out}\n  pandas={pandas_out}"
    # sanity: the expected survivors and composite cats
    assert pandas_out == {
        0: "posix_reader_lustre",
        1: "posix_checkpoint_lustre",
        2: "posix_reader",
        3: "posix_ssd",
        4: "reader",
        6: "data",
        10: "posix",  # cat untouched: null file_name fails the base condition
    }


def test_rebucket_time_range_matches_reference():
    origin = 1_000_000_000
    bw = 1_000_000  # 1s in us
    starts = [origin, origin + 500_000, origin + 1_500_000, origin + 4_200_000]
    tbl = pa.table(
        {
            "time_start": pa.array(starts, pa.int64()),
            "time_range": pa.array([s // bw for s in starts], pa.int64()),  # absolute (wrong)
        }
    )
    out = _rebucket_time_range(tbl, origin, bw).column("time_range").to_pylist()
    expected = [int((s - origin) // bw) for s in starts]
    assert out == expected == [0, 0, 1, 4]


def test_rebucket_noop_without_origin():
    tbl = pa.table({"time_start": pa.array([5], pa.int64()), "time_range": pa.array([99], pa.int64())})
    assert _rebucket_time_range(tbl, None, None).column("time_range").to_pylist() == [99]

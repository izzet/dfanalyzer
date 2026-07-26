"""Guard the Streamlit front end.

The app sat broken for 162 commits because nothing imported it: `rules.py` was
deleted, `XFER_SIZE_BIN_LABELS` was renamed, `AnalysisResult.characteristics`
went away, and `dask_jobqueue`'s import-time signal handler made the whole
package unimportable off the main thread. The render test below catches every
one of those in a couple of seconds.
"""

import pathlib
import pytest

pytest.importorskip("streamlit", reason="streamlit is an optional extra: pip install .[app]")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(pathlib.Path(__file__).resolve().parents[1] / "streamlit_app.py")
POSIX_TRACE_DIR = pathlib.Path("tests/data/extracted/dftracer-posix")

# Ground truth for the dftracer-posix fixture, matching tests/test_e2e.py.
EXPECTED_EVENTS = "2,056"
EXPECTED_PROCESSES = "1"


@pytest.mark.smoke
def test_streamlit_app_renders():
    """The app imports and lays out its form without raising."""
    at = AppTest.from_file(APP_PATH, default_timeout=120).run()

    assert not at.exception, f"app raised on first render: {at.exception[0].value}"
    assert at.file_uploader, "expected a trace file uploader"
    assert at.button, "expected an Analyze button"

    # Uncompressed .pfw parses to nothing (see #63), so it must not be offered.
    assert at.file_uploader[0].allowed_type == [".pfw.gz"]


@pytest.mark.full
def test_streamlit_app_analyzes_trace():
    """Uploading a real trace produces the same counts the e2e suite asserts."""
    traces = sorted(POSIX_TRACE_DIR.glob("*.pfw.gz"))
    assert traces, f"no traces in {POSIX_TRACE_DIR}; the conftest extraction should provide them"

    at = AppTest.from_file(APP_PATH, default_timeout=900).run()
    uploader = at.file_uploader[0]
    for trace in traces:
        uploader.upload(trace.name, trace.read_bytes())
    uploader.run(timeout=900)
    at.button[0].click().run(timeout=900)

    assert not at.exception, f"analysis raised: {at.exception[0].value}"
    assert not at.error, f"analysis reported an error: {[e.value for e in at.error]}"

    metrics = {m.label: m.value for m in at.metric}
    assert metrics.get("Events") == EXPECTED_EVENTS, (
        f"expected {EXPECTED_EVENTS} events, got {metrics.get('Events')}. A zero or short "
        "count means the trace was not read, not that the analysis is empty."
    )
    assert metrics.get("Processes") == EXPECTED_PROCESSES
    assert at.dataframe, "expected a per-layer breakdown table"

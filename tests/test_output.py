import io
import json
import multiprocessing
import socket
import pandas as pd
import pytest

from dftracer.analyzer.output import MofkaOutput, ZMQOutput
from dftracer.analyzer.streaming.mofka_io import open_consumer
from dftracer.analyzer.streaming.zmq_io import open_consumer as open_zmq_consumer
from dftracer.analyzer.types import AnalyzerResultType, RawStats


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _sample_result():
    flat_view = pd.DataFrame(
        {
            "epoch_time_sum": [10.0],
            "fetch_data_time_frac_epoch": [0.75],
        },
        index=pd.Index([7], name="epoch"),
    )
    result = AnalyzerResultType(
        _hlms={},
        _main_views={},
        _metric_boundaries={},
        checkpoint_dir=".",
        flat_views={("epoch",): flat_view},
        layers=["epoch"],
        raw_stats=RawStats(
            job_time=10,
            time_granularity=1,
            time_resolution=1,
            total_event_count=1,
            unique_file_count=1,
            unique_host_count=1,
            unique_process_count=1,
        ),
        view_types=["epoch"],
        views={},
    )
    return flat_view, result


def test_mofka_output_roundtrip(bedrock_mofka):
    group_file, topic_name = bedrock_mofka

    flat_view, result = _sample_result()

    def run_producer():
        output = MofkaOutput(group_file=group_file, topic_name=topic_name)
        output.handle_result(result)
        del output

    proc = multiprocessing.Process(target=run_producer, daemon=True)
    proc.start()

    driver, consumer = open_consumer(group_file, topic_name)
    try:
        future = consumer.pull()
        event = future.wait(timeout_ms=5000)
        assert event is not None, "No event received from MofkaOutput"

        metadata = event.metadata
        assert isinstance(metadata, dict)
        assert metadata.get("view_type") == "epoch"
        assert metadata.get("view_len") == len(flat_view)

        payload = event.data
        assert payload is not None, "No data payload received"
        if isinstance(payload, list):
            assert payload, "Empty data payload list"
            assert all(isinstance(item, (bytes, bytearray)) for item in payload), "Unexpected payload chunk type"
            payload = b"".join(payload)
        restored = pd.read_parquet(io.BytesIO(payload))
        assert len(restored) == len(flat_view)
        assert list(restored.columns) == list(flat_view.columns)
        assert list(restored.index) == list(flat_view.index)
        assert restored.index.name == flat_view.index.name
        event.acknowledge()
    finally:
        del consumer
        del driver
        proc.join(timeout=5)


def test_zmq_output_roundtrip():
    zmq = pytest.importorskip("zmq")
    flat_view, result = _sample_result()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    address = f"tcp://127.0.0.1:{port}"
    context, consumer = open_zmq_consumer(address, bind=True)
    consumer.setsockopt(zmq.RCVTIMEO, 5000)
    output = ZMQOutput(address=address, bind=False)
    try:
        output.handle_result(result)
        parts = consumer.recv_multipart()
        assert len(parts) == 2

        metadata = json.loads(parts[0].decode("utf-8"))
        assert metadata.get("view_type") == "epoch"
        assert metadata.get("view_len") == len(flat_view)

        payload = parts[1]
        restored = pd.read_parquet(io.BytesIO(payload))
        assert len(restored) == len(flat_view)
        assert list(restored.columns) == list(flat_view.columns)
        assert list(restored.index) == list(flat_view.index)
        assert restored.index.name == flat_view.index.name
    finally:
        output.close()
        consumer.close(linger=0)
        context.term()

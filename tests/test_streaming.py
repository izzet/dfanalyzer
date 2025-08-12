import json
import os
import pandas as pd
import pathlib
import pytest
import threading
import time
import zmq
from dfanalyzer import init_with_hydra
from dask.distributed import Client


def test_zmq(tmp_path: pathlib.Path) -> None:
    """Test ZMQ analysis pipeline with single trace file."""
    # Path to the test trace file
    # trace_file_path = "tests/data/extracted/dftracer-dlio/trace-0-of-8.pfw"
    # trace_file_path = "/usr/workspace/izzet/projects/dfanalyzer/tmp/bert_v100-1.pfw"
    # trace_file_path = "/usr/workspace/izzet/projects/dfanalyzer/tmp/unet3d_v100-1.pfw"
    trace_file_path = "/usr/workspace/izzet/projects/dfanalyzer-streaming/tmp/unet3d_v100-ai_logging.pfw"
    view_types = ['epoch']

    # ZMQ configuration
    zmq_port = 5555
    zmq_address = f"tcp://localhost:{zmq_port}"

    # Start ZMQ publisher in a separate thread
    publisher_thread = threading.Thread(target=_zmq_publisher, args=(trace_file_path, zmq_address), daemon=True)
    publisher_thread.start()

    # Give publisher time to start
    time.sleep(1)

    # Test 1: Initialize DFAnalyzer with ZMQ configuration (original approach)
    print("=== Test 1: Using DFAnalyzer initialization ===")
    dfa = init_with_hydra(
        hydra_overrides=[
            "analyzer=dftracer",
            "analyzer/preset=dlio-ailogging",
            f"analyzer.checkpoint={False}",
            f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
            f"cluster.processes={False}",
            f"hydra.run.dir={tmp_path}",
            f"hydra.runtime.output_dir={tmp_path}",
            "percentile=0.95",
            f"+trace_address={zmq_address}",
            f"view_types=[{','.join(view_types)}]",
        ]
    )

    print('client address', dfa.client.dashboard_link)


    extra_columns = {'epoch': 'Int8'}
    extra_columns_fn = lambda json_dict: {'epoch': json_dict.get('epoch', None)}

    # For streaming, we get a stream object instead of analysis results
    # from tornado.ioloop import IOLoop
    # stream = dfa.analyzer.read_zmq(
    #     trace_address=zmq_address,
    #     extra_columns=extra_columns,
    #     extra_columns_fn=extra_columns_fn,
    #     # loop=IOLoop.current(),
    # )

    stream = dfa.analyzer.analyze_zmq(
        address=zmq_address,
        view_types=view_types,
        extra_columns=extra_columns,
        extra_columns_fn=extra_columns_fn,
    )

    # Basic assertions to verify the stream was created
    assert stream is not None
    assert hasattr(stream, 'sink')  # Stream should have sink method

    # Set up a data collector to verify streaming works
    collected_data = []

    def analyze_data(data):
        print('*' * 33)
        print(f"Analyzing data:", data)
        print('data column names', data.columns if isinstance(data, pd.DataFrame) else 'N/A')
        result = {}
        if len(data) > 0:
            result = dfa.analyzer._analyze_trace(
                traces=data,
                view_types=view_types,
                logical_view_types=False,
                raw_stats={},
                metric_boundaries={},
            )
            result.flat_views[('epoch',)].to_json(f'stream_output_epoch_{time.time()}.json', orient='index')
            # result.flat_views[('proc_name',)].to_json(f'stream_output_proc_name_{time.time()}.json', orient='index')
        print('Analysis result', result)
        print('*' * 33)
        return result

    # Add sink to collect data
    # stream = stream.epoch_window_via_dict()  # .gather()
    # stream = stream.map(lambda x: x.get('args', {}).get('name')).sink(print)
    # stream = stream.sink(print)
    # stream
    # stream.map(lambda x,y: (x,len(y))).sink(print)
    # stream = dfa.analyzer.postread_zmq(
    #     stream,
    #     view_types=view_types,
    #     extra_columns=extra_columns,
    #     extra_columns_fn=extra_columns_fn,
    # )
    stream = stream.map(lambda result: result.flat_views[('epoch',)].to_json(f'stream_output_epoch_{time.time()}.json', orient='index'))

    stream.visualize()

    # Start the stream (this was missing!)
    stream.start()

    # Let the stream process data for a few seconds
    print("Waiting for stream to process data...")
    time.sleep(200)

    # Stop the stream
    stream.stop()

    # Verify we received some data
    print(f"Collected {len(collected_data)} data items")
    assert len(collected_data) > 0, "Test 1: No data was collected from the stream"

    # Shutdown the Dask client and cluster
    dfa.shutdown()

    # Wait for publisher thread to finish
    publisher_thread.join(timeout=5)


def _zmq_publisher(trace_file_path: str, zmq_address: str) -> None:
    """ZMQ publisher that sends trace data line by line."""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(zmq_address)

    # Give time for connections to establish
    time.sleep(5)

    try:
        with open(trace_file_path, 'r') as f:
            f.seek(0, os.SEEK_SET)
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        # Validate JSON before sending
                        json.loads(line)
                        print(f"Sending line: {line[:100]}...")  # Print first 100 chars
                        socket.send_string(line)
                        # Small delay to simulate streaming
                        # time.sleep(0.00001)
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
    except FileNotFoundError:
        pytest.fail(f"Trace file not found: {trace_file_path}")
    finally:
        socket.close()
        context.term()


def test_zmq_basic_connection(tmp_path: pathlib.Path) -> None:
    """Test basic ZMQ connection without analysis pipeline."""
    # Path to the test trace file
    trace_file_path = "tests/data/extracted/dftracer-dlio/trace-0-of-8.pfw"

    # ZMQ configuration
    zmq_port = 5556  # Different port to avoid conflicts
    zmq_address = f"tcp://localhost:{zmq_port}"

    # Start ZMQ publisher in a separate thread
    publisher_thread = threading.Thread(target=_zmq_publisher, args=(trace_file_path, zmq_address), daemon=True)
    publisher_thread.start()

    # Give publisher time to start
    time.sleep(1)

    # Test direct ZMQ connection
    from dfanalyzer.utils.streaming import is_streaming_available

    if not is_streaming_available:
        pytest.skip("streamz not installed")
    from streamz import Stream
    from tornado.ioloop import IOLoop

    # Create a simple stream to test ZMQ connectivity
    collected_messages = []

    def collect_raw_message(msg):
        print(f"Raw message received: {msg[:100]}...")  # Print first 100 chars
        collected_messages.append(msg)

    client = Client(processes=False)

    # Create stream with explicit SUB socket (matching PUB)
    stream = Stream.from_zmq(zmq_address, sock_type=zmq.SUB, subscribe=b"")  # type: ignore
    stream.scatter().map(lambda msg: msg.decode("utf-8")).gather().sink(collect_raw_message)  # type: ignore
    # stream.map(lambda msg: msg.decode("utf-8")).sink(collect_raw_message)  # type: ignore

    stream.visualize('stream_basic')  # type: ignore
    # Start the stream
    stream.start()  # type: ignore

    # Let it collect for a few seconds
    print("Testing basic ZMQ connection...")
    time.sleep(3)

    # Stop the stream
    stream.stop()  # type: ignore

    print(f"Collected {len(collected_messages)} raw messages")

    # Basic connectivity test
    assert len(collected_messages) > 0, "No messages received from ZMQ publisher"

    # Verify we can parse the first message as JSON
    if collected_messages:
        try:
            json.loads(collected_messages[0])
            print("✓ First message is valid JSON")
        except json.JSONDecodeError:
            pytest.fail("First received message is not valid JSON")

    # Wait for publisher thread to finish
    publisher_thread.join(timeout=5)


def test_stream_dask(tmp_path):
    from dask.distributed import Client
    from streamz import Stream
    from time import sleep

    def inc(x):
        sleep(1)  # simulate actual work
        return x + 1

    client = Client()  # Start a local Dask cluster

    source = Stream()
    (
        source.scatter()  # scatter local elements to cluster, creating a DaskStream
        .map(inc)  # map a function remotely
        .buffer(8)  # allow eight futures to stay on the cluster at any time
        .gather()  # bring results back to local process
        .sink(print)
    )  # call print locally

    for i in range(10):
        source.emit(i)

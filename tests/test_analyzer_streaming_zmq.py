import json
import multiprocessing
import pathlib
import pytest
import time

from dftracer.analyzer import init_with_hydra
from dftracer.analyzer.streaming.zmq_io import open_producer


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def test_analyzer_dftracer_analyze_zmq_stops_on_end(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("zmq")

    zmq_port = 5570
    bind_address = f"tcp://*:{zmq_port}"
    connect_address = f"tcp://127.0.0.1:{zmq_port}"

    dfa = init_with_hydra(
        hydra_overrides=[
            "analyzer=dftracer",
            "analyzer/preset=dlio",
            "analyzer.checkpoint=False",
            f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
            "cluster.processes=False",
            f"hydra.run.dir={tmp_path}",
            f"hydra.runtime.output_dir={tmp_path}",
            "input=zmq",
            f"input.address={bind_address}",
            "view_types=[epoch]",
        ]
    )

    end_event = json.loads(
        '{"id":478,"name":"end","cat":"dftracer","pid":4103080,"tid":4103080,"ts":1753300246416070,"dur":0,"ph":"X","args":{"hhash":"03089b0f8c47cc3d","p_idx":476,"num_events":477,"level":6}}'
    )

    def run_producer():
        context, producer = open_producer(connect_address)
        time.sleep(0.2)
        producer.send_json(end_event)
        producer.close(linger=0)
        context.term()

    proc = multiprocessing.Process(target=run_producer, daemon=True)
    proc.start()

    collected = []
    dfa.analyzer.analyze_zmq(
        address=bind_address,
        view_types=["epoch"],
        output_handler=collected.append,
    )
    assert collected == []

    try:
        dfa.shutdown()
    except Exception:
        pass

    proc.join(timeout=5)

import json
import multiprocessing
import os
import signal
import time
from collections import defaultdict

import pandas as pd
import pytest

from dftracer.analyzer import init_with_hydra
from dftracer.analyzer.streaming.mofka_io import open_producer


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _build_control_events(events, topic_name, control_topic_name):
    events_written_by_pid = defaultdict(int)
    control_events = []
    for event in events:
        pid = event.get("pid")
        if pid is None:
            continue
        pid = int(pid)
        events_written_by_pid[pid] += 1
        event_name = event.get("name")
        if event_name not in {"epoch.start", "epoch.block"}:
            continue
        control_events.append(
            {
                "type": "boundary_event",
                "trace_topic": topic_name,
                "control_topic": control_topic_name,
                "events_written": events_written_by_pid[pid],
                "trigger_event_name": event_name,
                "pid": pid,
                "tid": int(event.get("tid", pid)),
            }
        )
    return control_events


def test_e2e_mofka(
    tmp_path,
    bedrock_mofka_with_control_topic,
    dftracer_ai_logging_posix_events,
):
    group_file, topic_name, control_topic_name = bedrock_mofka_with_control_topic

    dfa = init_with_hydra(
        hydra_overrides=[
            "analyzer=dftracer",
            "analyzer/preset=dlio",
            "analyzer.preset.time_boundary_layer=epoch",
            "analyzer.checkpoint=False",
            f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
            "cluster.processes=False",
            f"hydra.run.dir={tmp_path}",
            f"hydra.runtime.output_dir={tmp_path}",
            "input=mofka",
            f"input.group_file={group_file}",
            f"input.topic_name={topic_name}",
            f"input.control_topic_name={control_topic_name}",
            "output=console",
            "view_types=[epoch]",
            "debug=True",
        ]
    )

    events_raw = dftracer_ai_logging_posix_events[0]
    events_raw.append(
        '{"id":478,"name":"end","cat":"dftracer","pid":4103080,"tid":4103080,"ts":1753300246416070,"dur":0,"ph":"X","args":{"hhash":"03089b0f8c47cc3d","p_idx":476,"num_events":477,"level":6}}'
    )
    events = [json.loads(line) for line in events_raw]
    control_events = _build_control_events(events, topic_name, control_topic_name)

    collected = []

    parent_pid = os.getpid()

    def run_producer():
        trace_driver, trace_producer = open_producer(group_file, topic_name)
        control_driver, control_producer = open_producer(group_file, control_topic_name)
        for event in control_events:
            control_producer.push(event)
        control_producer.flush()
        for event in events:
            trace_producer.push(event)
        trace_producer.flush()
        del control_producer
        del control_driver
        del trace_producer
        del trace_driver
        # Give the analyzer time to drain, then signal shutdown.
        time.sleep(2)
        os.kill(parent_pid, signal.SIGTERM)

    proc = multiprocessing.Process(target=run_producer, daemon=True)
    proc.start()

    dfa.analyzer.analyze_mofka(
        group_file=group_file,
        topic_name=topic_name,
        control_topic_name=control_topic_name,
        view_types=["epoch"],
        extra_columns={"epoch": "Int8"},
        output_handler=collected.append,
    )

    assert len(collected) > 0, "No analysis results emitted from analyze_mofka"
    result = collected[0]

    assert hasattr(result, "flat_views"), "Result missing flat_views"
    assert isinstance(result.flat_views, dict), "flat_views not a dict"
    assert len(result.flat_views) > 0, "flat_views is empty"

    any_df = False
    for k, v in result.flat_views.items():
        assert isinstance(k, tuple), "flat_views keys should be tuples (view_key)"
        assert isinstance(v, pd.DataFrame), f"flat_views[{k}] is not a pandas DataFrame"
        any_df = any_df or (getattr(v, "shape", (0, 0))[0] > 0 and getattr(v, "shape", (0, 0))[1] > 0)

    assert any_df, "No non-empty pandas DataFrame found in flat_views"

    try:
        dfa.shutdown()
    except Exception:
        pass

    proc.join(timeout=5)


def test_e2e_mofka_control_topic_boundaries(
    tmp_path,
    bedrock_mofka_with_control_topic,
    dftracer_ai_logging_posix_events,
    monkeypatch,
):
    group_file, topic_name, control_topic_name = bedrock_mofka_with_control_topic

    dfa = init_with_hydra(
        hydra_overrides=[
            "analyzer=dftracer",
            "analyzer/preset=dlio",
            "analyzer.preset.time_boundary_layer=epoch",
            "analyzer.checkpoint=False",
            f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
            "cluster.processes=False",
            f"hydra.run.dir={tmp_path}",
            f"hydra.runtime.output_dir={tmp_path}",
            "input=mofka",
            f"input.group_file={group_file}",
            f"input.topic_name={topic_name}",
            f"input.control_topic_name={control_topic_name}",
            "output=console",
            "view_types=[epoch]",
            "debug=True",
        ]
    )

    events_raw = dftracer_ai_logging_posix_events[0]
    events_raw.append(
        '{"id":478,"name":"end","cat":"dftracer","pid":4103080,"tid":4103080,"ts":1753300246416070,"dur":0,"ph":"X","args":{"hhash":"03089b0f8c47cc3d","p_idx":476,"num_events":477,"level":6}}'
    )
    events = [json.loads(line) for line in events_raw]
    control_events = _build_control_events(events, topic_name, control_topic_name)

    collected = []

    parent_pid = os.getpid()

    def run_producer():
        trace_driver, trace_producer = open_producer(group_file, topic_name)
        control_driver, control_producer = open_producer(group_file, control_topic_name)
        for event in control_events:
            control_producer.push(event)
        control_producer.flush()
        for event in events:
            trace_producer.push(event)
        trace_producer.flush()
        del control_producer
        del control_driver
        del trace_producer
        del trace_driver
        # Give the analyzer time to drain, then signal shutdown.
        time.sleep(2)
        os.kill(parent_pid, signal.SIGTERM)

    proc = multiprocessing.Process(target=run_producer, daemon=True)
    proc.start()

    monkeypatch.setenv("DFTRACER_MOFKA_CONTROL_TOPIC_NAME", control_topic_name)
    dfa.analyzer.analyze_mofka(
        group_file=group_file,
        topic_name=topic_name,
        view_types=["epoch"],
        extra_columns={"epoch": "Int8"},
        output_handler=collected.append,
    )

    assert len(collected) > 0, "No analysis results emitted from control-topic analyze_mofka"
    result = collected[0]
    assert hasattr(result, "flat_views")
    assert isinstance(result.flat_views, dict)
    assert len(result.flat_views) > 0

    any_df = False
    for k, v in result.flat_views.items():
        assert isinstance(k, tuple)
        assert isinstance(v, pd.DataFrame)
        any_df = any_df or (getattr(v, "shape", (0, 0))[0] > 0 and getattr(v, "shape", (0, 0))[1] > 0)
    assert any_df, "No non-empty pandas DataFrame found in flat_views (control topic path)"

    try:
        dfa.shutdown()
    except Exception:
        pass

    proc.join(timeout=5)

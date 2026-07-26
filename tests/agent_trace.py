"""Synthesize a minimal agent dftracer trace for the ``agent`` preset.

``tests/data`` ships no agent trace, so the seven agent layers cannot be
exercised against a recorded fixture. This builds a small but structurally valid
``.pfw.gz`` covering every agent layer plus interleaved POSIX I/O, which is
enough to check that the preset's layer_defs, derived metrics and token
normalized additional_metrics all resolve end to end.
"""

import gzip
import json

HOST_HASH = "03089b0f8c47cc3d"
HOST_NAME = "agenthost01"
PID = 4242
TID = 4242

# Files the synthetic agent touches, with their dftracer file hashes.
FILES = {
    "aaaa000000000001": "/data/agent/context/notes.md",
    "aaaa000000000002": "/data/agent/output/result.json",
}


def _event(event_id, name, cat, ts, dur, args=None):
    return {
        "id": event_id,
        "name": name,
        "cat": cat,
        "pid": PID,
        "tid": TID,
        "ts": ts,
        "dur": dur,
        "ph": "X",
        "args": {"hhash": HOST_HASH, "level": 2, **(args or {})},
    }


def build_agent_events():
    """Return the ordered list of raw trace records for one agent workflow."""
    records = [
        {
            "id": 1,
            "name": "HH",
            "cat": "dftracer",
            "pid": PID,
            "tid": TID,
            "ph": "M",
            "args": {"hhash": HOST_HASH, "name": HOST_NAME, "value": HOST_HASH},
        }
    ]
    for file_hash, file_name in FILES.items():
        records.append(
            {
                "id": len(records) + 1,
                "name": "FH",
                "cat": "dftracer",
                "pid": PID,
                "tid": TID,
                "ph": "M",
                "args": {"hhash": HOST_HASH, "name": file_name, "value": file_hash},
            }
        )

    base = 1753300000000000
    workflow_id = "wf-0001"
    agent_id = "agent-a"
    next_id = len(records) + 1

    def add(name, cat, ts, dur, args=None):
        nonlocal next_id
        records.append(_event(next_id, name, cat, ts, dur, args))
        next_id += 1

    # One workflow spanning two agent steps.
    add("run", "workflow", base, 4_000_000, {"workflow_id": workflow_id, "agent_id": agent_id})

    for step_index in range(2):
        step_start = base + step_index * 2_000_000
        step_args = {"workflow_id": workflow_id, "agent_id": agent_id, "step": step_index}
        add("plan", "step", step_start, 1_900_000, step_args)

        # LLM call carrying token counts (the agent preset's additional_fields).
        add(
            "call",
            "llm",
            step_start + 50_000,
            600_000,
            {
                **step_args,
                "llm_call_id": f"llm-{step_index}",
                "prompt_tokens": 1200 + step_index,
                "completion_tokens": 340 + step_index,
                "total_tokens": 1540 + 2 * step_index,
            },
        )
        # Tool invocation plus the data operation it performs.
        add(
            "call",
            "tool",
            step_start + 700_000,
            500_000,
            {**step_args, "tool_call_id": f"tool-{step_index}", "tool_name": "read_file"},
        )
        add(
            "load",
            "data",
            step_start + 750_000,
            300_000,
            {**step_args, "operation_kind": "read", "format": "markdown"},
        )
        add("send", "message", step_start + 1_300_000, 20_000, step_args)
        add("evaluate", "judge", step_start + 1_400_000, 200_000, step_args)

        # Interleaved POSIX I/O attributable to the step above.
        add(
            "open",
            "POSIX",
            step_start + 760_000,
            1_000,
            {**step_args, "fhash": "aaaa000000000001"},
        )
        add(
            "read",
            "POSIX",
            step_start + 780_000,
            9_000,
            {**step_args, "fhash": "aaaa000000000001", "ret": 65536, "offset": 0},
        )
        add(
            "write",
            "POSIX",
            step_start + 1_100_000,
            11_000,
            {**step_args, "fhash": "aaaa000000000002", "ret": 4096, "offset": 0},
        )
        add(
            "close",
            "POSIX",
            step_start + 1_200_000,
            1_000,
            {**step_args, "fhash": "aaaa000000000001"},
        )

    return records


def write_agent_trace(path):
    """Write the synthetic agent workflow to ``path`` as a gzipped .pfw."""
    with gzip.open(path, "wt") as handle:
        handle.write("[\n")
        for record in build_agent_events():
            handle.write(json.dumps(record) + "\n")
    return path

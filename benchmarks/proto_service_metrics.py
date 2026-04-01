"""Prototype: parse DFTracer service trace into time_range-aligned system metrics.

Validates the approach for integrating cat="sys" ph="C" events into
DFAnalyzer views before wiring into the analyzer proper.

Usage:
    python benchmarks/proto_service_metrics.py /path/to/dft_service_*.pfw
"""

import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIME_GRANULARITY = 5        # seconds (analysis window)
TIME_RESOLUTION = 1_000_000  # microseconds

BUCKET_WIDTH_US = TIME_GRANULARITY * TIME_RESOLUTION

# CPU metrics we care about for bottleneck correlation
CPU_METRICS = ["user_pct", "system_pct", "iowait_pct", "idle_pct", "irq_pct", "softirq_pct"]

# Memory metrics most relevant to I/O bottleneck analysis
MEMORY_METRICS = ["MemAvailable", "MemFree", "Cached", "Dirty", "Active", "Inactive(anon)"]


# ---------------------------------------------------------------------------
# Step 1: Parse service events
# ---------------------------------------------------------------------------

def parse_service_events(path):
    """Read a .pfw file and return only cat=sys, ph=C events."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                e = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if e.get("ph") == "C" and e.get("cat") == "sys":
                events.append(e)
    return events


def classify_events(events):
    """Split events into aggregate cpu, per-core cpu, and memory."""
    agg_cpu = []      # name == "cpu"
    per_core = defaultdict(list)  # name == "cpu-N"
    memory = []       # name == "memory"

    for e in events:
        name = e["name"]
        if name == "cpu":
            agg_cpu.append(e)
        elif name.startswith("cpu-"):
            per_core[name].append(e)
        elif name == "memory":
            memory.append(e)
    return agg_cpu, per_core, memory


# ---------------------------------------------------------------------------
# Step 2: Compute per-interval deltas (handles cumulative values)
# ---------------------------------------------------------------------------

def compute_interval_rates(events, metrics):
    """Convert cumulative readings to per-interval rates.

    For each consecutive pair of readings, compute the change in each metric.
    If the values are already instantaneous (per-interval), the deltas will
    be tiny and the raw values are more useful — so we return both.

    Returns list of dicts with ts, dt_us, and per-metric delta and raw values.
    """
    sorted_events = sorted(events, key=lambda e: e["ts"])
    intervals = []

    for i in range(1, len(sorted_events)):
        prev, curr = sorted_events[i - 1], sorted_events[i]
        dt_us = curr["ts"] - prev["ts"]
        row = {
            "ts": curr["ts"],
            "dt_us": dt_us,
            "hhash": curr.get("args", {}).get("hhash", ""),
        }
        for m in metrics:
            val_curr = curr["args"].get(m, 0.0)
            val_prev = prev["args"].get(m, 0.0)
            row[f"{m}_raw"] = val_curr
            row[f"{m}_delta"] = val_curr - val_prev
        intervals.append(row)
    return intervals


def detect_value_type(intervals, metrics):
    """Heuristic: are values cumulative-since-boot or per-interval?

    Cumulative values drift very slowly (deltas << 0.01 across intervals).
    Per-interval values change meaningfully between readings.
    """
    for m in metrics:
        deltas = [abs(r[f"{m}_delta"]) for r in intervals]
        raws = [abs(r[f"{m}_raw"]) for r in intervals]
        max_delta = max(deltas) if deltas else 0
        mean_raw = np.mean(raws) if raws else 0
        if mean_raw > 0 and max_delta / mean_raw < 0.001:
            return "cumulative"
    return "instantaneous"


# ---------------------------------------------------------------------------
# Step 3: Bucket into time_range windows and aggregate
# ---------------------------------------------------------------------------

def bucket_aggregate_cpu(events, metrics):
    """Aggregate CPU events into per-time_range rows.

    Uses raw values (mean within bucket) since that's the most robust
    approach whether values are cumulative or instantaneous.
    For cumulative values on an active machine, delta-based rates would
    be more accurate, but require the DFTracer service to emit deltas.
    """
    buckets = defaultdict(list)
    for e in sorted(events, key=lambda e: e["ts"]):
        tr = e["ts"] // BUCKET_WIDTH_US
        buckets[tr].append(e["args"])

    rows = []
    for tr in sorted(buckets):
        samples = buckets[tr]
        row = {"time_range": tr, "n_samples": len(samples)}
        row["hhash"] = samples[0].get("hhash", "")
        for m in metrics:
            vals = [s.get(m, 0.0) for s in samples]
            row[f"sys_cpu_{m}"] = np.mean(vals)
        rows.append(row)
    return rows


def bucket_aggregate_cores(per_core_events, metrics):
    """Aggregate per-core CPU events into cross-core stats per time_range.

    For each time_range, computes mean/max/p95/std across all cores and
    all samples within the bucket.
    """
    # Flatten all per-core events with their core id
    all_events = []
    for core_name, events in per_core_events.items():
        for e in events:
            all_events.append(e)

    buckets = defaultdict(lambda: defaultdict(list))
    for e in all_events:
        tr = e["ts"] // BUCKET_WIDTH_US
        for m in metrics:
            buckets[tr][m].append(e["args"].get(m, 0.0))

    rows = []
    for tr in sorted(buckets):
        row = {"time_range": tr}
        for m in metrics:
            arr = np.array(buckets[tr][m])
            n_cores = len(set(e["name"] for e in all_events if e["ts"] // BUCKET_WIDTH_US == tr))
            row[f"sys_core_{m}_mean"] = arr.mean()
            row[f"sys_core_{m}_max"] = arr.max()
            row[f"sys_core_{m}_p95"] = np.percentile(arr, 95)
            row[f"sys_core_{m}_std"] = arr.std()
            row["n_cores"] = n_cores
        rows.append(row)
    return rows


def bucket_aggregate_memory(events, metrics):
    """Aggregate memory events into per-time_range rows."""
    buckets = defaultdict(list)
    for e in sorted(events, key=lambda e: e["ts"]):
        tr = e["ts"] // BUCKET_WIDTH_US
        buckets[tr].append(e["args"])

    rows = []
    for tr in sorted(buckets):
        samples = buckets[tr]
        row = {"time_range": tr, "n_mem_samples": len(samples)}
        for m in metrics:
            vals = [s.get(m, 0.0) for s in samples]
            row[f"sys_mem_{m}"] = np.mean(vals)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Step 4: Build unified system metrics DataFrame
# ---------------------------------------------------------------------------

def build_system_metrics(path):
    """Full pipeline: parse → classify → aggregate → merge into one table."""
    events = parse_service_events(path)
    agg_cpu, per_core, memory = classify_events(events)

    print(f"Parsed {len(events)} service events")
    print(f"  Aggregate CPU: {len(agg_cpu)} events")
    print(f"  Per-core CPU:  {sum(len(v) for v in per_core.values())} events across {len(per_core)} cores")
    print(f"  Memory:        {len(memory)} events")

    # Detect cumulative vs instantaneous
    if agg_cpu:
        intervals = compute_interval_rates(agg_cpu, CPU_METRICS)
        value_type = detect_value_type(intervals, CPU_METRICS)
        print(f"  Value type (heuristic): {value_type}")

    # Aggregate per time_range
    cpu_rows = bucket_aggregate_cpu(agg_cpu, CPU_METRICS)
    core_rows = bucket_aggregate_cores(per_core, ["iowait_pct", "user_pct", "system_pct"])
    mem_rows = bucket_aggregate_memory(memory, MEMORY_METRICS)

    # Merge into single DataFrame by time_range
    cpu_df = pd.DataFrame(cpu_rows).set_index("time_range") if cpu_rows else pd.DataFrame()
    core_df = pd.DataFrame(core_rows).set_index("time_range") if core_rows else pd.DataFrame()
    mem_df = pd.DataFrame(mem_rows).set_index("time_range") if mem_rows else pd.DataFrame()

    dfs = [df for df in [cpu_df, core_df, mem_df] if not df.empty]
    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")

    return merged.reset_index()


# ---------------------------------------------------------------------------
# Step 5: Demonstrate join with mock flat view
# ---------------------------------------------------------------------------

def demo_flat_view_join(sys_metrics):
    """Show how system metrics would join with a flat view."""

    # Mock flat view: I/O bottleneck scores per (time_range, proc_name)
    time_ranges = sys_metrics["time_range"].values
    mock_flat = pd.DataFrame({
        "time_range": np.repeat(time_ranges, 2),
        "proc_name": [f"app#host1#{pid}#1" for tr in time_ranges for pid in [100, 200]],
        "posix_read_ops_slope_score": np.random.randint(0, 4, len(time_ranges) * 2),
        "posix_data_ops_slope_score": np.random.randint(0, 4, len(time_ranges) * 2),
    })

    # Join: system metrics are per (time_range, host) — broadcast to all procs on that host
    joined = mock_flat.merge(sys_metrics, on="time_range", how="left")

    print("\n=== Mock flat view + system metrics (joined) ===")
    print(joined.to_string(index=False))

    # Example analysis: correlate I/O bottleneck with iowait
    print("\n=== Potential bottleneck correlation ===")
    for _, row in joined.iterrows():
        if row["posix_read_ops_slope_score"] >= 2:
            iowait = row.get("sys_cpu_iowait_pct", 0)
            core_iowait_max = row.get("sys_core_iowait_pct_max", 0)
            print(f"  time_range={int(row['time_range'])} proc={row['proc_name']}: "
                  f"read_score={int(row['posix_read_ops_slope_score'])} "
                  f"iowait={iowait:.2f}% core_max_iowait={core_iowait_max:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <service_trace.pfw>")
        sys.exit(1)

    path = sys.argv[1]

    print(f"=== Processing: {path} ===")
    print(f"    time_granularity={TIME_GRANULARITY}s, time_resolution={TIME_RESOLUTION}us\n")

    sys_metrics = build_system_metrics(path)

    print(f"\n=== System Metrics Table ===")
    print(f"Shape: {sys_metrics.shape}")
    print(f"Columns ({len(sys_metrics.columns)}):")
    for col in sys_metrics.columns:
        print(f"  {col}")
    print()
    print(sys_metrics.to_string(index=False))

    demo_flat_view_join(sys_metrics)

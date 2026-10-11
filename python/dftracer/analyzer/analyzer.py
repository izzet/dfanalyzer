import abc
import dataclasses as dc
from collections import Counter, defaultdict, deque
import dask
import dask.dataframe as dd
import hashlib
import heapq
import itertools as it
import json
import gzip
import zlib
import math
import numpy as np
import os
import pandas as pd
import signal
import socket
import threading
import structlog
import time
from betterset import BetterSet as S
from dask.distributed import fire_and_forget, get_client, wait
from omegaconf import OmegaConf
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Module-level shutdown flag set by SIGTERM handler.
_shutdown_requested = False

HORIZON_SOURCE_FIELDS = (
    "epoch_index",
    "epochs_total",
    "window_index",
    "windows_per_epoch",
)


class _PendingTraceEvents:
    """The drain's carry-forward queue, bucketed per pid.

    WHAT IT IS FOR. A pulled trace batch carries the next window's events as well as this one's, so
    once a rank reaches its `events_written` target the rest of its events are parked here and
    handed to the window that claims them. `_pop_pending_event_for_window` asks for the oldest
    parked event whose pid has NOT yet reached its target, once per iteration of the counted loop.

    WHY IT IS NOT A PLAIN DEQUE. That question used to be answered by walking a single deque end to
    end, moving every over-target event into a side deque and putting all of it back. Two ways that
    is quadratic, and both were measured at 2 nodes:

      the walk returns None       every pull pays O(len) to learn nothing. Five windows of 62 at
                                  every_n=2 ran out the whole 4 s drain grace this way, 3.6 to
                                  3.9 s of it in the walk, at 347 to 1121 pulls.
      the walk returns an event   if the eligible pid's events sit behind thousands of over-target
                                  ones, every single pop walks past all of them. Windows with
                                  ZERO pulls and ZERO decode took 12.1 s to assemble 4147 events,
                                  which is that and nothing else.

    Fixing only the first left the second, so this fixes the shape instead. One deque per pid, each
    item stamped with a sequence number, and selection is a scan over the PIDS rather than over the
    events: at most one bucket per rank plus its workers. Insertion order across pids is preserved
    by the stamp, so the event handed back is the same one the walk would have chosen.
    """

    __slots__ = ("_by_pid", "_len", "_next_seq", "_next_left_seq", "_window_heap", "_window_key", "_sweep_ts")

    def __init__(self, iterable=()):
        self._by_pid = {}
        self._len = 0
        self._next_seq = 0
        self._next_left_seq = -1
        self._window_heap = []
        self._window_key = None
        self._sweep_ts = {}
        for item in iterable:
            self.append(item)

    # -- deque surface the drain already uses ---------------------------------------------
    def __len__(self):
        return self._len

    def __bool__(self):
        return self._len > 0

    def __iter__(self):
        """In insertion order across pids, so a caller can compare against a plain deque."""
        merged = []
        for bucket in self._by_pid.values():
            merged.extend(bucket)
        merged.sort(key=lambda entry: entry[0])
        return iter(item for _seq, item in merged)

    @staticmethod
    def _pid(item):
        return item[2] if isinstance(item, tuple) and len(item) > 2 else None

    def _bucket(self, pid):
        bucket = self._by_pid.get(pid)
        if bucket is None:
            bucket = deque()
            self._by_pid[pid] = bucket
        return bucket

    def append(self, item):
        self._window_key = None
        self._sweep_ts.pop(self._pid(item), None)
        self._bucket(self._pid(item)).append((self._next_seq, item))
        self._next_seq += 1
        self._len += 1

    def appendleft(self, item):
        self._window_key = None
        self._sweep_ts.pop(self._pid(item), None)
        self._bucket(self._pid(item)).appendleft((self._next_left_seq, item))
        self._next_left_seq -= 1
        self._len += 1

    def extend(self, iterable):
        for item in iterable:
            self.append(item)

    def extendleft(self, iterable):
        for item in iterable:
            self.appendleft(item)

    def clear(self):
        self._window_key = None
        self._sweep_ts.clear()
        self._by_pid.clear()
        self._len = 0

    def _take(self, pid, from_left):
        self._window_key = None
        self._sweep_ts.pop(pid, None)
        bucket = self._by_pid[pid]
        entry = bucket.popleft() if from_left else bucket.pop()
        if not bucket:
            del self._by_pid[pid]
        self._len -= 1
        return entry[1]

    def popleft(self):
        if not self._len:
            raise IndexError("pop from an empty _PendingTraceEvents")
        pid = min(self._by_pid, key=lambda key: self._by_pid[key][0][0])
        return self._take(pid, from_left=True)

    def pop(self):
        if not self._len:
            raise IndexError("pop from an empty _PendingTraceEvents")
        pid = max(self._by_pid, key=lambda key: self._by_pid[key][-1][0])
        return self._take(pid, from_left=False)

    # -- what the drain actually wants ----------------------------------------------------
    def pids(self):
        """The distinct pids currently parked. At most one per rank plus its workers."""
        return list(self._by_pid)

    def pop_first_under_target(self, is_over_target, is_ineligible=None):
        """The oldest parked event whose pid is not yet at its target, or None.

        O(pids), not O(events). This is the whole reason the class exists.
        """
        best_pid = None
        best_seq = None
        for pid, bucket in self._by_pid.items():
            if is_over_target(pid):
                continue
            if is_ineligible is not None and is_ineligible(bucket[0][1]):
                continue
            seq = bucket[0][0]
            if best_seq is None or seq < best_seq:
                best_pid, best_seq = pid, seq
        if best_pid is None:
            return None
        return self._take(best_pid, from_left=True)

    def sweep_for_window(self, targets, seen, boundary_ts_us):
        """Partition carry-forward buckets once; return selected rows in order.

        Already-satisfied rank buckets cannot contribute any rows. Worker
        timestamps need not be ordered, so inspect every worker row instead
        of inferring a timestamp prefix. Avoid repeatedly finding the global
        oldest PID for rows that will all be carried back unchanged.
        """
        self._window_key = None
        selected = []
        for pid, bucket in list(self._by_pid.items()):
            count = seen.get(pid, 0)
            target = targets.get(pid)
            if target is not None and count >= target:
                continue
            if target is None:
                # Worker rows have no count target. Retained future rows can
                # vastly outnumber this window's rows. Cache their timestamps
                # and partition in C, still checking EVERY timestamp rather
                # than assuming any ordering. Object dtype preserves Python
                # integer comparisons, including values outside int64.
                if boundary_ts_us:
                    cached = self._sweep_ts.get(pid)
                    timestamps = cached[1] if cached is not None and cached[0] is bucket else np.fromiter(
                        (item[1] or 0 for _, item in bucket), dtype=object, count=len(bucket)
                    )
                    future = (timestamps != 0) & (timestamps > boundary_ts_us)
                    selected.extend(it.compress(bucket, ~future))
                    retained = deque(it.compress(bucket, future))
                else:
                    selected.extend(bucket)
                    retained = deque()
                if retained:
                    self._by_pid[pid] = retained
                    self._sweep_ts[pid] = (retained, timestamps[future])
                else:
                    del self._by_pid[pid]
                    self._sweep_ts.pop(pid, None)
                continue
            self._sweep_ts.pop(pid, None)
            retained = deque()
            for seq, item in bucket:
                ts = item[1]
                over_target = target is not None and count >= target
                past_boundary = bool(boundary_ts_us and ts and ts > boundary_ts_us)
                if over_target or past_boundary:
                    retained.append((seq, item))
                else:
                    selected.append((seq, item))
                    if pid is not None:
                        count += 1
            if retained:
                self._by_pid[pid] = retained
            else:
                del self._by_pid[pid]
        self._len -= len(selected)
        selected.sort(key=lambda entry: entry[0])
        return [item for _, item in selected]

    def pop_for_window(self, targets, seen, boundary_ts_us):
        """Select the same eligible oldest event with a heap of bucket heads.

        Within one window counts only increase. A rank whose target is met
        cannot become eligible again; a worker's timestamp fence is fixed.
        Rebuild on queue insertion or a new window. General deque operations
        invalidate the cursor, so they retain their existing contract.
        """
        def eligible(pid, item):
            if pid in targets:
                return seen.get(pid, 0) < targets[pid]
            return not (boundary_ts_us and item[1] and item[1] > boundary_ts_us)

        key = (targets, seen, boundary_ts_us)
        if (self._window_key is None or self._window_key[0] is not targets
                or self._window_key[1] is not seen or self._window_key[2] != boundary_ts_us):
            self._window_heap = [
                (bucket[0][0], pid) for pid, bucket in self._by_pid.items()
                if eligible(pid, bucket[0][1])
            ]
            heapq.heapify(self._window_heap)
            self._window_key = key
        while self._window_heap:
            _, pid = self._window_heap[0]
            # The legacy selector uses None as its no-candidate sentinel.
            # Retain that behavior for pid-less parked records too.
            if pid is None and eligible(pid, self._by_pid[pid][0][1]):
                return None
            heapq.heappop(self._window_heap)
            bucket = self._by_pid[pid]
            if not eligible(pid, bucket[0][1]):
                continue
            _, item = bucket.popleft()
            self._sweep_ts.pop(pid, None)
            self._len -= 1
            if bucket:
                if eligible(pid, bucket[0][1]):
                    heapq.heappush(self._window_heap, (bucket[0][0], pid))
            else:
                del self._by_pid[pid]
            return item
        return None


_TRACE_FRAMING_LINES = ("[", "]")


def split_trace_batch_lines(text: str) -> Tuple[List[str], int]:
    """Split a newline-delimited trace batch into (payload lines, framing-line count).

    The SHM writer's per-pid `events_written` drain target counts every newline in the
    payloads it publishes (shm_ring_writer.cpp, "COUNT EVENTS, NOT RECORDS").  DFTracer's
    array framing line (`[`, written once at the head of a process's first flush) is such a
    newline but carries no event, so a consumer that counts only parsed events is one short
    of the target for every writing pid, and the window can never close.  Framing lines are
    returned separately so the caller can charge them to the pid that wrote the batch.
    """
    lines: List[str] = []
    framing = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line in _TRACE_FRAMING_LINES:
            framing += 1
            continue
        lines.append(line)
    return lines, framing


def drain_settle_ms(targets_satisfied: bool, timeout_ms: int, override: Optional[int] = None) -> int:
    """How long the drain's sweep may block once the counted loop has finished.

    The sweep drains what the consumer can hand over right now. Blocking for it can only end two
    ways: a batch arrives whose first event is past the window boundary and is pushed straight back
    onto pending, or nothing arrives. Neither adds an event to this window, so when every rank's
    target is already met the wait is dead time paid once per window. When the counted loop instead
    ran out of grace with targets unmet, records really are still in flight and waiting is correct.
    """
    if override is not None:
        return max(0, override)
    return 0 if targets_satisfied else max(0, timeout_ms)


def _drained_window_span_ms(events: List[dict]) -> Optional[float]:
    """Wall-clock span of the events a window drained, in milliseconds.

    Used for the release check that per-window analysis is faster than the window it describes
    (`T3-0-SHM-LOOP-CORONA.md` section 4j condition 4). Derived from the events themselves rather
    than from the boundary timestamps so it stays correct when a window closes on a timeout.
    """
    first = None
    last = None
    for event in events:
        ts = event.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        end = ts + (event.get("dur") or 0)
        if first is None or ts < first:
            first = ts
        if last is None or end > last:
            last = end
    if first is None or last is None:
        return None
    return round((last - first) / 1000.0, 1)


def _lane_occupancy() -> Optional[dict]:
    """Ring lane fill and drop counters, when the liveflow transport is the one in use.

    Section 4j found 37.6% of trace and 98.6% of control records dropped once the analyzer fell
    behind, and none of it was visible until the run was over. Sampling at window close puts the
    drop counters next to the analysis time that causes them.
    """
    try:
        from .streaming import transport
    except Exception:
        return None
    stats = getattr(transport, "lane_occupancy", None)
    if stats is None:
        return None
    try:
        return stats()
    except Exception:
        return None


from .liveflow_v1 import STEP_EVIDENCE_COUNTERS, STEP_EVIDENCE_FIELDS, window_step_evidence  # noqa: E402


def window_horizon_metrics(events: List[dict], control_event: Optional[dict] = None) -> Dict[str, int]:
    """Return the declared and remaining horizon for one completed window.

    New writers may copy the fields into boundary metadata.  Existing writers
    retain them only on the duration-bearing window.stop trace record, so both
    locations are accepted.  No value is synthesized when the four-field
    contract is incomplete.
    """
    candidates = []
    if isinstance(control_event, dict):
        candidates.append(control_event)
    candidates.extend(
        event for event in events
        if isinstance(event, dict) and event.get("name") == "window.stop"
    )

    step_fields = ("configured_total_steps", "completed_steps", "configured_window_steps")
    step_sources = []
    for candidate in candidates:
        try:
            values = {name: int(candidate[name]) for name in step_fields}
        except (KeyError, TypeError, ValueError):
            continue
        if (values["configured_total_steps"] >= values["completed_steps"] >= 0
                and values["configured_window_steps"] >= 1):
            step_sources.append(values)
    if step_sources:
        first_steps = step_sources[0]
        if any(value != first_steps for value in step_sources[1:]):
            logger.warning("window.horizon.step_mismatch", candidates=step_sources)
            return {}
        remaining = first_steps["configured_total_steps"] - first_steps["completed_steps"]
        ready = [int(candidate["planner_history_ready"]) for candidate in candidates
                 if "planner_history_ready" in candidate]
        history = {"planner_history_ready": ready[0]} if ready and len(set(ready)) == 1 else {}
        # The applied (physical) vector: every `physical_<knob>` the writer snapshotted plus the
        # step it took effect at. Knob names are the workload's (halo3d: read_threads/decode_threads;
        # StorMer: num_workers/prefetch_factor/bucket_cap_mb), so the field set is read from the
        # events, not fixed here. Ranks of one node must agree exactly; otherwise nothing is
        # passed (a discordant vector is not evidence of any configuration).
        vectors = []
        for c in candidates:
            if "physical_applied_step" not in c:
                continue
            try:
                vectors.append(tuple(sorted((k, int(v)) for k, v in c.items()
                                            if isinstance(k, str) and k.startswith("physical_"))))
            except (TypeError, ValueError):
                continue
        physical = dict(vectors[0]) if vectors and len(set(vectors)) == 1 else {}
        # Tail of the per-step wall in the window (microseconds -> seconds is the planner's job).
        # Each rank reports its own p90; the node's value is the worst rank's, the critical path.
        tails = []
        for c in candidates:
            try:
                tails.append(int(c["step_wall_p90_us"]))
            except (KeyError, TypeError, ValueError):
                continue
        tail = {"step_wall_p90_us": max(tails)} if tails else {}
        tail.update(window_step_evidence(candidates))
        return dict(first_steps, **history, **physical, **tail, remaining_steps=remaining,
                    windows_remaining=remaining / first_steps["configured_window_steps"])

    complete = []
    for candidate in candidates:
        try:
            values = {name: int(candidate[name]) for name in HORIZON_SOURCE_FIELDS}
        except (KeyError, TypeError, ValueError):
            continue
        if values["epoch_index"] < 1 or values["epochs_total"] < 1:
            continue
        if values["window_index"] < 1 or values["windows_per_epoch"] < 1:
            continue
        complete.append(values)

    if not complete:
        return {}
    first = complete[0]
    if any(values != first for values in complete[1:]):
        logger.warning("window.horizon.mismatch", candidates=complete)
        return {}

    # `window_index` is the native control-window identity, while
    # `epochs_total * windows_per_epoch` is the declared size of the run at
    # this cadence.  A tuple beyond that bound is internally inconsistent.
    # Clamping it to zero would make a malformed source look like a genuine
    # exhausted run and silently convert a source-contract failure into a
    # planner decline.  Keep it absent instead: no horizon evidence, no
    # cost-aware action.
    declared_windows = first["epochs_total"] * first["windows_per_epoch"]
    if first["window_index"] > declared_windows:
        logger.warning("window.horizon.impossible", values=first,
                       declared_windows=declared_windows)
        return {}

    metrics = dict(first)
    metrics["epochs_remaining"] = max(first["epochs_total"] - first["epoch_index"], 0)
    metrics["windows_remaining"] = max(
        first["epochs_total"] * first["windows_per_epoch"] - first["window_index"],
        0,
    )
    return metrics


def attach_window_horizon(result, metrics: Dict[str, int]) -> None:
    """Attach horizon passthrough metrics to flat views and every emitted fact."""
    if not metrics:
        return
    for flat_view in result.flat_views.values():
        for name, value in metrics.items():
            flat_view[name] = value
    for _view_key, fact in result.iter_analysis_facts():
        fact.evidence.setdefault("metrics", {}).update(metrics)


def _ack_event(event) -> None:
    """Acknowledge a Mofka event -- disabled by default. See LIVEFLOW-STATUS sec 60.

    On diaspora-stream-api 0.5.1, acking an event that belongs to a multi-event batch
    permanently stops delivery to that consumer, and leaves the group in a state where
    peer processes hang on exit. Measured with liveflow_validation/repro_consumer_stall.sh:
    ack on -> 1 of 100 events received; ack off -> 100 of 100.

    LiveFlow loses nothing by not acking: topics are created per run with `--type memory`
    partitions, so no consumer ever resumes a previous offset.

    Set LIVEFLOW_MOFKA_ACK=1 to restore the old behaviour.
    """
    if os.environ.get("LIVEFLOW_MOFKA_ACK", "0") == "1":
        event.acknowledge()


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def install_shutdown_handler():
    """Install SIGTERM handler so the analyzer exits gracefully.

    No-op off the main thread. `signal.signal` raises ValueError anywhere else, and under the
    Option C node agent (LIVEFLOW-ARCHITECTURE.md sec 6) this analyzer runs as ONE THREAD beside
    the diagnoser and optimizer rather than owning its own process. Signal disposition is
    process-wide, so a stage thread was never the right place to set it: the agent installs its
    own handler on the main thread and shuts the stages down through it.

    Silently skipping is correct rather than merely convenient. The handler exists so a
    standalone analyzer process exits cleanly on SIGTERM; when the analyzer is not the process,
    that responsibility belongs to whoever is.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    signal.signal(signal.SIGTERM, _sigterm_handler)

from .analysis_utils import (
    build_view_rename_map,
    derive_call_stats,
    fix_dtypes,
    fix_hlm_dtypes,
    fix_std_cols,
    set_file_dir,
    set_file_pattern,
    set_size_bins,
    set_unique_counts,
    split_duration_records_vectorized,
)
from .config import CHECKPOINT_VIEWS, HASH_CHECKPOINT_NAMES, AnalyzerPresetConfig, FactsConfig
from .constants import (
    COL_CATEGORY,
    COL_COUNT,
    COL_FILE_NAME,
    COL_FUNC_NAME,
    COL_HOST_NAME,
    COL_PROC_NAME,
    COL_TIME_END,
    COL_TIME_RANGE,
    COL_TIME_START,
    VIEW_TYPES,
    Layer,
)
from .metrics import (
    attach_window_integrity_metrics,
    attach_input_step_max_sum_metrics,
    attach_input_job_step_max_sum_metrics,
    attach_comm_exposed_median_metrics,
    attach_comm_total_median_metrics,
    find_layer_time_metrics,
    set_cross_layer_metrics,
    set_main_metrics,
    set_quantile_metrics,
    set_view_metrics,
)
from .fact_engine import FactEngine
from .types import (
    AnalysisResult,
    ReadTraceResult,
    RawStats,
    ViewKey,
    ViewMetricBoundaries,
    ViewType,
    Views,
)
from .utils.collection_utils import is_set_like_series
from .utils.dask_agg import quantile_stats, unique_set, unique_set_flatten
from .utils.expr_utils import extract_numerator_and_denominators
from .utils.file_utils import ensure_dir
from .utils.json_encoders import NpEncoder
from .utils.log_utils import console_block, log_block
from .utils.pandas_agg import unique_set_flatten_pd, unique_set_pd
from .utils.pandas_utils import flatten_column_names, to_nullable_numeric
from .streaming.window_buffer import WindowTracker, WindowBuffer


CHECKPOINT_FLAT_VIEW = "_flat_view"
CHECKPOINT_HLM = "_hlm"
CHECKPOINT_MAIN_VIEW = "_main_view"
CHECKPOINT_RAW_STATS = "_raw_stats"
CHECKPOINT_VIEW = "_view"
HLM_AGG = {
    "time": "sum",
    "count": "sum",
    "size": "sum",
}
PARTITION_SIZE = "128MB"
VIEW_PERMUTATIONS = False

DataFrameType = Union[dd.DataFrame, pd.DataFrame]

logger = structlog.get_logger()


class UnresolvedTraceMetadataError(RuntimeError):
    """Trace metadata (host/file hashes) did not resolve, so every view would be empty."""


def _raw_stats_as_dict(raw_stats) -> dict:
    """Normalise RawStats (dataclass) or a plain dict to a dict, for the guard below."""
    if dc.is_dataclass(raw_stats) and not isinstance(raw_stats, type):
        return dc.asdict(raw_stats)
    return dict(raw_stats) if isinstance(raw_stats, dict) else {}


def _assert_metadata_resolved(raw_stats_data: dict) -> None:
    """FAIL LOUDLY when metadata resolution collapsed, instead of emitting schema-valid zeros.

    THE FAILURE THIS CATCHES (measured 2026-08-28). On a 16-node/64-rank StorMer corpus, an offline
    run produced `unique_host_count=0` while parsing perfectly: identical `total_event_count`
    (1,213,709), identical `unique_process_count` (192), identical `job_time_s` to a known-good run
    on the SAME traces. `get_unique_host_count()` is `traces["host_hash"].nunique()`, which returns 0
    only when the column is ENTIRELY NA -- so host metadata had silently failed to attach. Every
    host-involving groupby then yields zero groups and all views collapse to 0.0.

    The run exited 0 and wrote a schema-valid metrics.json full of zeros. That is the dangerous
    shape: an analysis that reports nothing is indistinguishable from a workload that did nothing,
    and this project has already mistaken one for the other repeatedly (a dead env var, a stale
    regex matching zero rows, a `_noop` optimizer context). Reporting `Total Nodes 0` against a
    64-rank job is not a result, it is a broken read.

    Known trigger class, not yet root-caused: a dftracer-utils version newer than the pinned 0.0.5
    (0.0.12 renamed Reader->TraceReader and flattened the `args` dict). Shims exist for both
    (`dftracer.py:12-18`, `_unflatten_args`), so the exact defeating mechanism is still open. This
    guard is deliberately independent of that root cause -- it asserts the OUTCOME, so it keeps
    working whatever the next incompatibility turns out to be.

    Set DFANALYZER_ALLOW_UNRESOLVED_METADATA=1 to downgrade to a warning (e.g. a genuinely
    single-host corpus with metadata intentionally stripped).
    """
    def _concrete(value):
        """Return an int, or None if the value is lazy/absent.

        MUST NOT use truthiness or `or` on these: at this call site the values may still be lazy
        dask Scalars, and `bool(dask_scalar)` RAISES TypeError ("cannot be converted to a boolean
        value"). An earlier version of this guard did `raw_stats_data.get(...) or 0` and crashed
        every offline run at `read_stats`. Skipping lazily-valued calls is correct: the assertion
        is a safety net, not a computation trigger, and forcing `.compute()` here would change the
        analyzer's execution shape just to run a check.
        """
        if value is None or hasattr(value, "compute") or hasattr(value, "dask"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    events = _concrete(raw_stats_data.get("total_event_count"))
    if events is None or events <= 0:
        return  # lazy, absent, or genuinely no events -- all different problems, none ours

    broken = []
    for name in ("unique_host_count", "unique_process_count"):
        value = _concrete(raw_stats_data.get(name))
        if value is not None and value <= 0:
            broken.append(name)
    if not broken:
        return

    detail = ", ".join("%s=0" % n for n in broken)
    if os.environ.get("DFANALYZER_ALLOW_UNRESOLVED_METADATA", "0") == "1":
        logger.warning("analyzer.metadata.unresolved", broken=detail, total_event_count=events,
                       note="continuing because DFANALYZER_ALLOW_UNRESOLVED_METADATA=1")
        return
    raise UnresolvedTraceMetadataError(
        "trace metadata did not resolve (%s) despite %d parsed events -- every view would be empty "
        "and the output would be schema-valid zeros.\n"
        "  Known cause (2026-08-28, still OPEN): host metadata does not propagate from the few "
        "records that carry `args.hhash` (metadata/HH records) to the vast majority that do not. "
        "Measured on a 64-rank corpus: 49 of 18,462 records carried an hhash; the rest stayed NA, "
        "so `traces['host_hash'].nunique()` returned 0.\n"
        "  NOT the cause, though both were suspected and eliminated: (a) a dftracer-utils version "
        "mismatch -- `Reader` is used ONLY by load_indexed_gzip_files, the GZIP path, so plain .pfw "
        "corpora never touch it; (b) pandas `astype(str)` on NaN -- that was MASKING this defect by "
        "turning NA into the literal string 'nan' (which is why older runs reported "
        "unique_host_count=1 on a 16-node job), and is now fixed to `astype(\"string\")`.\n"
        "  Set DFANALYZER_ALLOW_UNRESOLVED_METADATA=1 to downgrade this to a warning if a zero "
        "really is expected (e.g. a corpus with metadata deliberately stripped)."
        % (detail, events)
    )


def _replace_zeros_preserving_stall_time(frame, preserve=False, by_function=False):
    """Keep literal measured stall zeros before min/mean aggregation.

    Missing input and undefined ratios stay null. Dropping zero early biases
    mixed zero/nonzero groups even if publication restores all-zero groups.
    """
    result = frame.replace(0, pd.NA)
    if not preserve:
        return result
    count_cols = [c for c in frame.columns if
                  (c[0] if isinstance(c, tuple) else c) in ("count", "count_sum")]
    if not count_cols:
        return result
    counts = pd.to_numeric(frame[count_cols[0]], errors="coerce")
    observed = counts.notna() & np.isfinite(counts) & (counts > 0)
    if by_function:
        if "func_name" not in frame.index.names:
            return result
        observed &= frame.index.get_level_values("func_name").isin(("stall.storage", "stall.decode"))
    time_stats = {"time", "time_sq", "time_sum", "time_call_min", "time_call_max",
                  "time_call_mean", "time_call_std", "time_proc_min", "time_proc_max",
                  "time_proc_mean", "time_proc_std"}
    for col in frame.columns:
        metric = col[0] if isinstance(col, tuple) else col
        if metric in time_stats:
            result[col] = result[col].mask(observed & frame[col].eq(0), 0.0)
    return result


class Analyzer(abc.ABC):
    def __init__(
        self,
        preset: AnalyzerPresetConfig,
        checkpoint: bool = True,
        checkpoint_dir: str = "",
        debug: bool = False,
        facts_config: Optional[Union[FactsConfig, Dict[str, Any]]] = None,
        profile_distribution: str = "uniform",
        profile_time_granularity: float = 5,
        quantile_stats: bool = False,
        time_approximate: bool = True,
        time_granularity: float = 1,
        time_resolution: float = 1e6,
        time_sliced: bool = False,
        verbose: bool = False,
    ):
        """Initializes the Analyzer instance.

        Args:
            preset: The configuration preset for the analyzer.
            checkpoint: Whether to enable checkpointing of intermediate results.
            checkpoint_dir: Directory to store checkpoint data.
            debug: Whether to enable debug mode.
            profile_distribution: Strategy for distributing profile measures
                when expanding buckets ('uniform' or 'weighted').
            profile_time_granularity: The time granularity of profile buckets
                emitted by the tracer, in seconds.
            time_approximate: Whether to use approximate time for I/O operations.
            time_granularity: The time granularity for analysis, in seconds.
            time_resolution: The time resolution for analysis, in microseconds.
            time_sliced: Whether to slice time ranges for analysis.
            verbose: Whether to enable verbose logging.
        """
        if checkpoint:
            assert checkpoint_dir != "", "Checkpoint directory must be defined"
        if profile_distribution not in ("uniform", "weighted"):
            raise ValueError(f"profile_distribution must be 'uniform' or 'weighted', got '{profile_distribution}'")

        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_tasks = []
        self.dask_client = get_client()
        self.debug = debug
        self.profile_distribution = profile_distribution
        self.profile_time_granularity = profile_time_granularity
        self.quantile_stats = quantile_stats
        self.layers = list(preset.layer_defs.keys())
        self.facts_config = self._build_facts_config(facts_config)
        self.fact_engine = None
        if self.facts_config.enabled:
            self.fact_engine = FactEngine.from_rule_file(
                self.facts_config.rule_file,
                strict_time_semantics=self.facts_config.strict_time_semantics,
                allow_mixed_time_aggregates=self.facts_config.allow_mixed_time_aggregates,
            )
        if preset.logical_views is None:
            self.logical_views = {}
        elif isinstance(preset.logical_views, dict):
            self.logical_views = preset.logical_views
        else:
            self.logical_views = dict(OmegaConf.to_object(preset.logical_views))  # type: ignore
        if self.fact_engine is not None and getattr(preset, "name", None) == "liveflow-v1":
            self.fact_engine.liveflow_declaration = dict(preset.liveflow_families)
        self.preset = preset
        self.time_approximate = time_approximate
        self.time_granularity = time_granularity
        self.time_resolution = time_resolution
        self.time_sliced = time_sliced
        self.verbose = verbose
        # Set only around an online completed control window.  Offline analysis
        # deliberately has no fabricated window/rank identity.
        self._debug_window_provenance = None
        self._fact_window_context = None
        ensure_dir(self.checkpoint_dir)

    def _debug_artifact_suffix(self) -> str:
        provenance = self._debug_window_provenance or {}
        window_index = provenance.get("window_index")
        if window_index is None:
            return ""
        return f"_window_{int(window_index):06d}"

    def _write_debug_flat_view(self, flat_view: pd.DataFrame, view_key: ViewKey) -> None:
        """Write a debug view with its index and online-window provenance intact.

        Flat-view indices are the view's join keys (for example epoch or
        process).  Dropping them and overwriting a fixed filename made an online
        snapshot impossible to associate with its completed control window.
        These artifacts are debug/provenance evidence only; this does not alter
        the emitted result, facts, or optimizer inputs.
        """
        debug_dir = getattr(self, "debug_output_dir", os.getcwd())
        os.makedirs(debug_dir, exist_ok=True)

        # Preserve every index level under reserved names, avoiding collisions
        # with metrics that happen to share an index-level label.
        index_values = flat_view.index.to_frame(index=False).reset_index(drop=True)
        index_values.columns = [f"__dfanalyzer_index_{i}_{name or 'unnamed'}"
                                for i, name in enumerate(flat_view.index.names)]
        export = pd.concat([index_values, flat_view.reset_index(drop=True)], axis=1)

        provenance = self._debug_window_provenance or {}
        export.insert(0, "__dfanalyzer_view_key", json.dumps(list(view_key)))
        if provenance:
            export.insert(0, "__liveflow_window_index", provenance["window_index"])
            export.insert(1, "__liveflow_rank_pids", json.dumps(provenance["rank_pids"]))
            export.insert(2, "__liveflow_boundary_ts_ns_by_pid",
                          json.dumps(provenance["boundary_ts_ns_by_pid"], sort_keys=True))

        name = f"flat_view_{'_'.join(view_key)}{self._debug_artifact_suffix()}.csv"
        export.to_csv(os.path.join(debug_dir, name), index=False)

    def _write_debug_window_integrity_ledger(self, traces: pd.DataFrame) -> None:
        """Export raw rows that can explain a window-integrity decision.

        This is debug provenance, not an alternate aggregation: it preserves the
        normalized rows passed to the analyzer and marks precisely which rows
        the window integrity helper is allowed to count.  It is intentionally
        emitted only for a completed control window.
        """
        if not self.debug or traces is None or not isinstance(traces, pd.DataFrame):
            return
        provenance = self._debug_window_provenance or {}
        window_index = provenance.get("window_index")
        if window_index is None:
            return
        debug_dir = getattr(self, "debug_output_dir", os.getcwd())
        os.makedirs(debug_dir, exist_ok=True)
        columns = [
            column for column in (
                "window", "pid", "proc_name", "time_start", "time_end",
                "name", "func_name", "cat", "epoch", "step",
            ) if column in traces.columns
        ]
        export = traces.loc[:, columns].copy()
        rank_pids = {int(pid) for pid in provenance.get("rank_pids", [])}
        if "pid" in export.columns:
            numeric_pid = pd.to_numeric(export["pid"], errors="coerce")
            is_rank_pid = numeric_pid.isin(rank_pids)
            export["__liveflow_is_rank_pid"] = is_rank_pid
        else:
            is_rank_pid = pd.Series(False, index=traces.index)
            export["__liveflow_is_rank_pid"] = is_rank_pid
        needed = {"window", "epoch", "step"}
        if needed.issubset(traces.columns):
            numeric_epoch = pd.to_numeric(traces["epoch"], errors="coerce")
            numeric_step = pd.to_numeric(traces["step"], errors="coerce")
            export["__liveflow_integrity_contributes"] = (
                is_rank_pid & traces["window"].notna()
                & numeric_epoch.notna() & numeric_step.notna()
            )
        else:
            export["__liveflow_integrity_contributes"] = False
        export.insert(0, "__liveflow_analysis_window", int(window_index))
        name = f"window_integrity_ledger{self._debug_artifact_suffix()}.csv"
        export.to_csv(os.path.join(debug_dir, name), index=False)

    def _write_debug_fact_envelope(self, result: AnalysisResult) -> None:
        """Preserve the exact emitted fact envelope for a completed debug window.

        The CSV views explain aggregation and the integrity ledger explains the
        gate, but neither retains the selected worker PID carried by a fact.
        This is debug provenance only; transport still serializes the same
        ``AnalysisResult.to_fact_envelope()`` payload.
        """
        if not self.debug or not result.analysis_facts:
            return
        debug_dir = getattr(self, "debug_output_dir", os.getcwd())
        os.makedirs(debug_dir, exist_ok=True)
        envelope = result.to_fact_envelope().to_dict()
        name = f"analysis_facts{self._debug_artifact_suffix()}.json"
        with open(os.path.join(debug_dir, name), "w") as handle:
            json.dump(envelope, handle, cls=NpEncoder, indent=2, sort_keys=True)

    @staticmethod
    def _build_facts_config(facts_config: Optional[Union[FactsConfig, Dict[str, Any]]]) -> FactsConfig:
        if facts_config is None:
            return FactsConfig()
        if isinstance(facts_config, FactsConfig):
            return facts_config
        if isinstance(facts_config, dict):
            return FactsConfig(**facts_config)
        if OmegaConf.is_config(facts_config):
            config_obj = OmegaConf.to_object(facts_config)
            if isinstance(config_obj, FactsConfig):
                return config_obj
            if isinstance(config_obj, dict):
                return FactsConfig(**config_obj)
            if dc.is_dataclass(config_obj):
                return FactsConfig(**dc.asdict(config_obj))
            raise TypeError(f"Unsupported OmegaConf facts object type: {type(config_obj)}")
        raise TypeError(f"Unsupported facts_config type: {type(facts_config)}")

    def _evaluate_analysis_facts(
        self,
        flat_views: Dict[ViewKey, pd.DataFrame],
        raw_stats: Any,
    ) -> Dict[ViewKey, List[Any]]:
        # Always-on (2026-09-28, follow-up to the facts.evaluate.view_keys diagnostic): the
        # v4.5.7 rerun showed that line NEVER fires, which by elimination (both AnalysisResult
        # construction paths that reach here execute without raising) means one of the two
        # early returns below is the one firing silently. Only the second is logged (DEBUG);
        # the first never was. Log both conditions unconditionally, plus id(self.fact_engine),
        # so a comparison against the id logged where FastPathPlan.build() consumes
        # self.fact_engine (fastpath.py's analyzer.fastpath.enabled line) settles directly
        # whether self.fact_engine is None here or has become a *different* object.
        logger.info(
            "facts.evaluate_analysis_facts.entry",
            fact_engine_is_none=(self.fact_engine is None),
            fact_engine_id=(id(self.fact_engine) if self.fact_engine is not None else None),
            emit_analysis_facts=self.facts_config.emit_analysis_facts,
            analyzer_id=id(self),
        )
        if self.fact_engine is None:
            return {}
        if not self.facts_config.emit_analysis_facts:
            logger.debug("facts.emit_analysis_facts.disabled")
            return {}
        fact_context = dict(self._fact_window_context or {})
        # A proc_name slope is meaningful only relative to peer processes;
        # the optimizer must nevertheless receive the completed window's one
        # coherent vector, not whichever worker happened to be selected.
        window_view = flat_views.get(("window",))
        if fact_context and window_view is not None and len(window_view) == 1:
            row = window_view.iloc[0]
            control_names = (
                "domain_coherent", "domain_procs_active",
                "window_wall_time_proc_crit", "domain_input_time_proc_crit",
                "domain_decode_time_proc_crit", "domain_comm_time_proc_crit",
                "domain_compute_time_proc_crit", "residual_time_proc_crit",
                "epoch_nunique", "step_nunique", "step_min", "step_max",
            )
            # Optional measured cross-rank skew attribution (metrics.SKEW_COLUMNS) and the typical
            # rank's own vector, which the planner needs to price the skew link.  Optional: a
            # producer without them simply omits them, and rules requiring them fail closed.
            from .fact_engine import LOOP_EXPOSURE_WINDOW_METRICS, SKEW_WINDOW_METRICS
            fact_context["window_metrics"] = {
                name: row[name]
                for name in control_names + SKEW_WINDOW_METRICS + LOOP_EXPOSURE_WINDOW_METRICS
                if name in row.index
            }
        result = self.fact_engine.evaluate(
            flat_views=flat_views,
            raw_stats=raw_stats,
            context=fact_context,
        )
        # 2026-09-28 (judge batch 5 G4, orchestrator direction): patch 13's unconditional
        # logger.info() as the FIRST statement inside FactEngine.evaluate() never fired in the
        # v4.5.8.1 1N run even though this call is provably reached (fact_engine_is_none=false,
        # emit_analysis_facts=true, fact_engine_id matching the id logged at construction) and no
        # exception was raised (analysis_facts=0 is a clean return, not a crash). Rather than
        # trust another log line to explain a log line that already failed to appear, persist the
        # same information via plain, unbuffered file I/O -- immune to any structlog/logging
        # level, handler, or cache-on-first-use configuration this process may have accumulated
        # across the analyzer/diagnoser/optimizer stages' independent configure_logging() calls.
        try:
            _diag_dir = getattr(self, "debug_output_dir", None)
            if _diag_dir:
                _diag_path = os.path.join(_diag_dir, "fact_engine_diagnostics.jsonl")
                _std_logger = __import__("logging").getLogger("dftracer.analyzer.fact_engine")
                with open(_diag_path, "a") as _f:
                    _f.write(json.dumps({
                        "view_keys": [list(_k) for _k in flat_views.keys()],
                        "view_row_counts": {str(_k): len(_v) for _k, _v in flat_views.items()},
                        "rules_by_view_keys": list(self.fact_engine.rules_by_view.keys()),
                        "facts_returned_counts": {str(_k): len(_v) for _k, _v in result.items()},
                        "total_facts": sum(len(_v) for _v in result.values()),
                        # 2026-09-28 (orchestrator follow-up): views/rules align (source_view
                        # "proc_name" is present in flat_views with real rows) yet facts_returned
                        # is still empty every window -- ruling out a view-config/source_view
                        # mismatch. Both rules declare rank_thread_only: true, which requires
                        # non-empty context["row_pid_by_index"] and context["worker_pids"] (see
                        # fact_engine.py's early skip checks). An ordinary StorMer SHM launch
                        # (this diagnostic included) does not activate libdftracer_preload.so
                        # inside forked DataLoader workers (STORMER-WORKER-POSIX-GATE.md), so
                        # worker PIDs may never appear distinct from rank PIDs in the traces this
                        # window saw. Record both sizes directly to confirm or refute that without
                        # another round of guessing.
                        "context_row_pid_by_index_size": len(fact_context.get("row_pid_by_index", {}) or {}),
                        "context_worker_pids_size": len(fact_context.get("worker_pids", ()) or ()),
                        "context_rank_pids_size": len(fact_context.get("rank_pids", ()) or ()),
                        "fact_engine_logger_level": _std_logger.level,
                        "fact_engine_logger_effective_level": _std_logger.getEffectiveLevel(),
                        "fact_engine_logger_propagate": _std_logger.propagate,
                        "fact_engine_logger_disabled": _std_logger.disabled,
                        "fact_engine_logger_handler_count": len(_std_logger.handlers),
                        "root_logger_handler_count": len(__import__("logging").getLogger().handlers),
                    }) + "\n")
                    _f.flush()
                    os.fsync(_f.fileno())
        except Exception:
            pass  # diagnostic-only: never let this persistence path break the analysis pipeline
        return result

    def _materialize_output_artifacts(
        self,
        flat_views: Dict[ViewKey, pd.DataFrame],
        analysis_facts: Dict[ViewKey, List[Any]],
    ) -> Tuple[Dict[ViewKey, pd.DataFrame], Dict[ViewKey, List[Any]]]:
        output_flat_views = flat_views if self.facts_config.emit_flat_views else {}
        output_analysis_facts = analysis_facts if self.facts_config.emit_analysis_facts else {}
        return output_flat_views, output_analysis_facts

    def analyze_file(
        self,
        path: str,
        view_types: List[ViewType],
        exclude_characteristics: List[str] = [],
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        logical_view_types: bool = False,
        metric_boundaries: ViewMetricBoundaries = {},
    ) -> AnalysisResult:
        """Analyzes I/O trace data to identify performance bottlenecks.

        This method orchestrates the entire analysis process, including reading
        trace data, computing various metrics and views, evaluating these views
        to detect bottlenecks, and applying rules to characterize them.

        Args:
            path: Path to the I/O trace file or directory.
            exclude_characteristics: A list of I/O characteristics to exclude.
            logical_view_types: Whether to compute views based on logical relationships.
            view_types: A list of view types to compute (e.g., 'file_name', 'proc_name').

        Returns:
            An AnalysisResult object containing the analysis results.
        """
        proc_view_types = self.ensure_proc_view_type(view_types=view_types)
        hlm_checkpoint_name = self.get_hlm_checkpoint_name(view_types=proc_view_types)
        read_result = None
        profiles = None
        traces = None
        raw_stats = None
        with console_block("Read trace & stats"):
            if not self.checkpoint or not self.has_checkpoint(name=hlm_checkpoint_name):
                # Read trace & stats
                with log_block("read_trace"):
                    read_result = self.read_trace(
                        trace_path=path,
                        extra_columns=extra_columns,
                        extra_columns_fn=extra_columns_fn,
                    )
                    traces = read_result.traces
                    profiles = read_result.profiles
                if profiles is not None:
                    with log_block("validate_and_expand_profiles"):
                        ptg = read_result.profile_time_granularity or self.profile_time_granularity
                        profiles = self._validate_and_expand_profiles(
                            profiles=profiles,
                            profile_time_granularity=ptg,
                        )
                        read_result.profiles = profiles
                with log_block("read_stats"):
                    raw_stats = self.read_stats(traces=traces, profiles=profiles)
                with log_block("postread_trace"):
                    traces = self.postread_trace(
                        traces=traces,
                        view_types=proc_view_types,
                    )
                with log_block("set_size_bins"):
                    traces = traces.map_partitions(set_size_bins)
                if self.time_sliced:
                    with log_block("split_duration_records_vectorized"):
                        traces = traces.map_partitions(
                            split_duration_records_vectorized,
                            time_granularity=self.time_granularity,
                            time_resolution=self.time_resolution,
                        )
                if read_result is not None:
                    read_result.traces = traces
            else:
                # Restore stats
                with log_block("restore_raw_stats"):
                    raw_stats = self.restore_extra_data(
                        name=self.get_stats_checkpoint_name(),
                        fallback=lambda: None,
                    )

        # Compute high-level metrics
        is_dask = isinstance(traces, dd.DataFrame)
        hlm = None
        trace_hlm = None
        profile_hlm = None
        with console_block("Compute high-level metrics"):
            if profiles is None:
                with log_block("compute_high_level_metrics"):
                    hlm = self.compute_high_level_metrics(
                        checkpoint_name=hlm_checkpoint_name,
                        traces=traces,
                        view_types=proc_view_types,
                    )
                if is_dask:
                    with log_block("persist"):
                        (hlm, raw_stats) = dask.persist(hlm, raw_stats)
                    with log_block("wait"):
                        wait([hlm, raw_stats])
            else:
                with log_block("compute_trace_hlm"):
                    trace_hlm = self.compute_high_level_metrics(
                        checkpoint_name=self.get_trace_hlm_checkpoint_name(view_types=proc_view_types),
                        traces=traces,
                        view_types=proc_view_types,
                    )
                with log_block("compute_profile_hlm"):
                    profile_hlm = self.compute_profile_hlm(
                        checkpoint_name=self.get_profile_hlm_checkpoint_name(view_types=proc_view_types),
                        profiles=profiles,
                        view_types=proc_view_types,
                    )
                hlm = None
                with log_block("persist"):
                    (trace_hlm, profile_hlm, raw_stats) = dask.persist(trace_hlm, profile_hlm, raw_stats)
                with log_block("wait"):
                    wait([trace_hlm, profile_hlm, raw_stats])

        # Compute system metrics (small table, safe to materialize)
        system_metrics = None
        if read_result is not None and read_result.system_metrics is not None:
            with log_block("compute_system_metrics"):
                system_metrics = read_result.system_metrics.compute()

        # Analyze HLM
        result = self._analyze_hlm(
            hlm=hlm,
            is_dask=is_dask,
            logical_view_types=logical_view_types,
            metric_boundaries=metric_boundaries,
            profile_hlm=profile_hlm,
            proc_view_types=proc_view_types,
            raw_stats=raw_stats,
            system_metrics=system_metrics,
            trace_hlm=trace_hlm,
        )

        # Attach correct traces & view types
        if read_result is not None:
            result._read_result = read_result
        result.view_types = view_types

        return result

    def analyze_zmq(
        self,
        address: str,
        view_types: List[ViewType],
        exclude_characteristics: List[str] = [],
        logical_view_types: bool = False,
        metric_boundaries: ViewMetricBoundaries = {},
        epoch_start_name: str = "epoch.start",
        epoch_end_name: str = "epoch.block",
        process_key: str = "pid",
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        output_handler: Optional[Callable[[AnalysisResult], None]] = None,
    ) -> None:
        from .streaming.zmq_io import open_consumer

        context, consumer = open_consumer(address)
        logger.debug("ZMQ consumer started", address=address)

        # Buffer for unpacking newline-delimited JSON batches from ZMQ messages.
        pending_events: List[dict] = []

        def pull_batch_event():
            while not pending_events:
                data = consumer.recv()
                text = data.decode("utf-8", errors="replace")
                for line in text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        pending_events.append(event)
            return pending_events.pop(0)

        try:
            self._analyze_stream(
                pull_event=pull_batch_event,
                view_types=view_types,
                exclude_characteristics=exclude_characteristics,
                logical_view_types=logical_view_types,
                metric_boundaries=metric_boundaries,
                epoch_start_name=epoch_start_name,
                epoch_end_name=epoch_end_name,
                process_key=process_key,
                extra_columns=extra_columns,
                extra_columns_fn=extra_columns_fn,
                output_handler=output_handler,
                stream_name="zmq",
            )
        finally:
            consumer.close(linger=0)
            context.term()

    def _analyze_stream(
        self,
        pull_event: Callable[[], Optional[dict]],
        view_types: List[ViewType],
        exclude_characteristics: List[str] = [],
        logical_view_types: bool = False,
        metric_boundaries: ViewMetricBoundaries = {},
        epoch_start_name: str = "epoch.start",
        epoch_end_name: str = "epoch.block",
        process_key: str = "pid",
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        output_handler: Optional[Callable[[AnalysisResult], None]] = None,
        stream_name: str = "stream",
    ) -> None:
        install_shutdown_handler()
        proc_view_types = self.ensure_proc_view_type(view_types=view_types)
        window_start_name = epoch_start_name
        window_end_name = epoch_end_name
        buffer = WindowBuffer(
            window_start_name=window_start_name,
            window_end_name=window_end_name,
            process_key=process_key,
        )
        output_handler = output_handler or (lambda result: None)

        while not _shutdown_requested:
            event = pull_event()
            if event is None:
                if _shutdown_requested:
                    break
                raise RuntimeError(f"{stream_name} consumer returned no event")
            if not isinstance(event, dict):
                raise ValueError(f"Invalid event type from {stream_name}: {type(event)}")
            logger.debug(f"{stream_name}.raw_event", name=event.get("name"), ph=event.get("ph"))

            # Remove internal window labels from extra_columns as they are handled
            # by the WindowBuffer.
            normalized_extra_columns = extra_columns.copy() if extra_columns else None
            if normalized_extra_columns:
                normalized_extra_columns.pop("epoch", None)
                normalized_extra_columns.pop("step", None)
                normalized_extra_columns.pop("window", None)

            normalized_event = self.normalize_trace_event(
                event=event,
                extra_columns=normalized_extra_columns,
                extra_columns_fn=extra_columns_fn,
            )
            logger.debug(f"{stream_name}.normalized_event", name=normalized_event.get("name"))

            window_events = buffer.push(normalized_event)
            if window_events:
                logger.debug(f"{stream_name}.window_emitted", count=len(window_events))
            if not window_events:
                continue

            traces, profiles, system_metrics = self.handle_stream_events(
                events=window_events,
                view_types=proc_view_types,
                extra_columns=extra_columns,
            )
            result = self._analyze_trace(
                traces=traces,
                proc_view_types=proc_view_types,
                logical_view_types=logical_view_types,
                raw_stats={},
                metric_boundaries=metric_boundaries,
                profiles=profiles,
                system_metrics=system_metrics,
            )
            logger.debug(f"{stream_name}.analysis_complete", flat_views=len(result.flat_views))
            output_handler(result)

    def _analyze_mofka_with_control(
        self,
        group_file: str,
        topic_name: str,
        control_topic_name: str,
        view_types: List[ViewType],
        exclude_characteristics: List[str] = [],
        logical_view_types: bool = False,
        metric_boundaries: ViewMetricBoundaries = {},
        trace_drain_grace_ms: int = 5000,
        process_key: str = "pid",
        trace_consumer_name: Optional[str] = None,
        control_consumer_name: Optional[str] = None,
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        output_handler: Optional[Callable[[AnalysisResult], None]] = None,
        num_ranks: int = 1,
        worker_roster_dir: Optional[str] = None,
        worker_roster_prefix: Optional[str] = None,
        worker_roster_expected_workers: Optional[int] = None,
        worker_roster_schema: Optional[str] = None,
    ) -> None:
        from .streaming.transport import open_consumer

        trace_driver, trace_consumer = open_consumer(
            group_file,
            topic_name,
            consumer_name=trace_consumer_name or "dftracer-analyzer",
            use_progress_thread=False,
        )
        control_driver, control_consumer = open_consumer(
            group_file,
            control_topic_name,
            consumer_name=control_consumer_name or "dftracer-analyzer-control",
            use_progress_thread=False,
        )
        logger.debug(
            "Mofka consumers started",
            trace_topic=topic_name,
            control_topic=control_topic_name,
        )

        continuous_trace_consumer = None

        install_shutdown_handler()
        proc_view_types = self.ensure_proc_view_type(view_types=view_types)
        output_handler = output_handler or (lambda result: None)

        normalized_extra_columns = extra_columns.copy() if extra_columns else None
        if normalized_extra_columns:
            normalized_extra_columns.pop("epoch", None)
            normalized_extra_columns.pop("step", None)
            normalized_extra_columns.pop("window", None)

        # Window-based analysis: the Window class handles cadence gating.
        # The analyzer processes every window.start / window.stop pair.
        window_tracker = WindowTracker(
            num_ranks=num_ranks,
            require_explicit_start=True,
        )
        selected_trace_drain_grace_ms = max(trace_drain_grace_ms, 0)

        control_wait_timeout_ms = 1000
        trace_drain_timeout_ms = 100
        # HOW LONG THE SWEEP MAY BLOCK ONCE EVERY RANK'S TARGET IS ALREADY MET.
        #
        # The sweep's job, in its own comment below, is "drain whatever the consumer can hand over
        # right now". It was blocking for up to trace_drain_timeout_ms to do it, and that wait can
        # only ever end one of two ways: a batch arrives whose first event is past the boundary and
        # is immediately pushed back onto pending, or nothing arrives and the loop breaks. Neither
        # outcome adds an event to this window. So whenever the counted loop finished because every
        # target was met -- which is the normal case, since the writer publishes each trace record
        # BEFORE the boundary that counts it -- the wait is dead time, paid once per window.
        #
        # When the counted loop instead ran out of grace with targets UNMET, records really are
        # still in flight and waiting is the right thing, so the timeout stays in force there.
        # LIVEFLOW_DRAIN_SETTLE_MS forces a value either way, which is what the ablation arm uses.
        drain_settle_ms_override = os.environ.get("LIVEFLOW_DRAIN_SETTLE_MS")
        try:
            drain_settle_ms_override = (
                int(drain_settle_ms_override) if drain_settle_ms_override is not None else None
            )
        except ValueError:
            drain_settle_ms_override = None
        # pending_trace_events deque holds buffered events from batch splits
        # (used by both target_counts and timestamp-based drain paths)

        def _parse_control_int(control_event: dict, key: str) -> Optional[int]:
            value = control_event.get(key)
            if value is None or value == "":
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def _normalize_single_trace(
            trace_event: dict,
        ) -> Tuple[Optional[dict], Optional[int], Optional[int]]:
            """Normalize a single trace event dict and return (event, ts, pid)."""
            normalized_event = self.normalize_trace_event(
                event=trace_event,
                extra_columns=normalized_extra_columns,
                extra_columns_fn=extra_columns_fn,
            )
            # The writer's boundary metadata historically kept only epoch and
            # step.  Preserve the horizon contract from the window trace
            # record so the completed-window path can forward it even before
            # every writer implementation learns the new fields.
            args = trace_event.get("args", {})
            if isinstance(normalized_event, dict) and isinstance(args, dict):
                for field in (*HORIZON_SOURCE_FIELDS, "configured_total_steps",
                              "completed_steps", "configured_window_steps", "planner_history_ready",
                              "step_wall_p90_us", *STEP_EVIDENCE_FIELDS):
                    if field in args:
                        normalized_event[field] = args[field]
                for field in args:
                    if isinstance(field, str) and field.startswith("physical_"):
                        normalized_event[field] = args[field]
            event_ts = None
            event_pid = None
            try:
                event_ts = int(trace_event.get("ts", 0))
            except (TypeError, ValueError):
                pass
            raw_pid = trace_event.get(process_key)
            if raw_pid is None and isinstance(normalized_event, dict):
                raw_pid = normalized_event.get(process_key)
            try:
                if raw_pid is not None:
                    event_pid = int(raw_pid)
            except (TypeError, ValueError):
                event_pid = None
            return normalized_event, event_ts, event_pid

        def _normalize_trace_event(
            mofka_event,
        ) -> List[Tuple[Optional[dict], Optional[int], Optional[int]]]:
            """Unpack a Mofka trace event into one or more normalized events.

            Supports both batched format (newline-delimited JSON in DataView)
            and legacy single-event format (JSON in metadata).
            """
            if isinstance(mofka_event, PreparedTraceRecord):
                return mofka_event.normalized()
            results = []
            metadata = mofka_event.metadata

            if isinstance(metadata, dict) and metadata.get("type") == "batch":
                # Batched format: events are in DataView as newline-delimited JSON
                data = mofka_event.data
                if isinstance(data, list):
                    data = b"".join(data)
                if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                    # dftracer gzips the batch unless DFTRACER_TRACE_COMPRESSION=0. Decoding a
                    # gzip stream as UTF-8 with errors="replace" does not raise -- it yields
                    # replacement characters, every json.loads below hits JSONDecodeError, and the
                    # `continue` swallows it. The result is a window that drains for its full grace
                    # period and closes EMPTY, with a ring that is provably full (151,787 events)
                    # and not one log line saying anything was discarded. Detect the magic and
                    # decompress rather than trusting callers to unset the env var.
                    if data[:2] == b"\x1f\x8b":
                        try:
                            data = gzip.decompress(bytes(data))
                        except (OSError, EOFError, zlib.error) as exc:
                            logger.warning("trace.batch.gunzip_failed", error=str(exc),
                                           nbytes=len(data))
                            return results
                    text = data.decode("utf-8", errors="replace")
                    batch_lines, framing_lines = split_trace_batch_lines(text)
                    for line in batch_lines:
                        try:
                            trace_event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(trace_event, dict):
                            results.append(_normalize_single_trace(trace_event))
                    if framing_lines and results:
                        # One ring record has one writer pid.  Charge the framing lines the
                        # writer counted to it, as event-less placeholders, so `seen` and the
                        # writer's target count the same unit.
                        _, first_ts, first_pid = results[0]
                        results[:0] = [(None, first_ts, first_pid)] * framing_lines
            elif isinstance(metadata, dict):
                # Legacy single-event format: event JSON in metadata
                results.append(_normalize_single_trace(metadata))

            if hasattr(mofka_event, "acknowledge"):
                _ack_event(mofka_event)
            return results

        from .streaming.continuous import BufferedTraceConsumer, PreparedTraceRecord
        if os.environ.get("LIVEFLOW_TRANSPORT") == "liveflow":
            continuous_trace_consumer = BufferedTraceConsumer(
                trace_consumer, prepare=_normalize_trace_event,
                idle_wait_ms=trace_drain_timeout_ms,
            )
            trace_consumer = continuous_trace_consumer

        def _append_trace_event_to_window(
            normalized_event: Optional[dict],
            sink_events: List[dict],
            *,
            window_index: Optional[int] = None,
            start_counts: Optional[Dict[int, int]] = None,
        ) -> None:
            if not isinstance(normalized_event, dict):
                return
            # METADATA IS NOT WINDOW-SCOPED, and treating it as if it were is what emptied the
            # reader_posix layer for the whole of this project's streaming history.
            #
            # dftracer emits an FH record once, when a file is first opened. For DLIO that is
            # during loader construction, BEFORE the first window.start, so `current_window` is
            # None for every one of them and the `return` below dropped all 504 of a lane's file
            # hashes. Every pread afterwards then carried a `file_hash` that resolved to nothing,
            # `file_name` stayed NA, `_fix_file_posix_category` could not tag a single row, and
            # `reader_posix` was empty in every window of every run. Fed the same events with the
            # FH records present, the identical frame code resolves all 2001 preads of a sample and
            # categorises them `posix_reader_lustre`, so nothing downstream was ever wrong.
            #
            # A hash-to-name mapping belongs to the RUN, not to a window, so it is kept aside and
            # replayed into every window. Deduplicated by (type, hash, name): the tables hold one
            # row per file, host and string, bounded by the corpus rather than by the run.
            _etype = normalized_event.get("type")
            if _etype is not None and int(_etype) != 0:      # 0 == TYPE_EVENT
                stream_metadata_events.setdefault(
                    (int(_etype), normalized_event.get("hash"), normalized_event.get("name")),
                    dict(normalized_event),
                )
                return
            event_pid = normalized_event.get(process_key)
            if event_pid is None:
                logger.warning("mofka.trace_event.missing_pid", name=normalized_event.get("name"))
                return

            event_pid = int(event_pid)
            if start_counts and event_pid in start_counts:
                if trace_events_seen_by_pid.get(event_pid, 0) <= start_counts[event_pid]:
                    return
            current_window = window_index
            if current_window is None:
                current_window = window_tracker.current_window(event_pid)
            if current_window is None:
                return
            event_with_window = self._attach_stream_window(
                normalized_event,
                current_window,
                native_window="window" in proc_view_types,
            )
            sink_events.append(event_with_window)

        trace_events_seen_by_pid = defaultdict(int)
        pending_trace_events = _PendingTraceEvents()
        # Hash tables seen so far in this run, replayed into every window drain. See the note in
        # _append_trace_event_to_window for why they cannot be window-scoped.
        stream_metadata_events: Dict[tuple, dict] = {}

        def _targets_satisfied(target_counts: Dict[int, int]) -> bool:
            if not target_counts:
                return False
            return all(trace_events_seen_by_pid.get(pid, 0) >= target for pid, target in target_counts.items())

        def _event_exceeds_target(event_pid: Optional[int], target_counts: Dict[int, int]) -> bool:
            if event_pid is None:
                return False
            if event_pid not in target_counts:
                return False
            return trace_events_seen_by_pid.get(event_pid, 0) >= target_counts[event_pid]

        def _record_trace_event(
            normalized_event: Optional[dict],
            event_pid: Optional[int],
            sink_events: List[dict],
            *,
            window_index: Optional[int] = None,
            start_counts: Optional[Dict[int, int]] = None,
        ) -> None:
            if event_pid is not None:
                trace_events_seen_by_pid[event_pid] += 1
            _append_trace_event_to_window(
                normalized_event,
                sink_events,
                window_index=window_index,
                start_counts=start_counts,
            )

        def _pop_pending_event_for_window(
            target_counts: Dict[int, int], boundary_ts_us: int
        ):
            """The oldest parked event whose pid is still under its target, or None.

            The queue answers this in O(pids). It used to be a walk over the events, which is
            what made a window of 4147 parked events take 12.1 s to assemble with zero pulls and
            zero decode.
            """
            return pending_trace_events.pop_for_window(
                target_counts, trace_events_seen_by_pid, boundary_ts_us
            )

        trace_snapshot = None
        last_completed_boundary = None
        accepted_stop_controls = {}
        try:
            if continuous_trace_consumer is not None:
                continuous_trace_consumer.start()
            control_future = control_consumer.pull()
            trace_future = trace_consumer.pull()
            eos_seen = False
            while not _shutdown_requested:
                # Control boundaries still define analysis windows. The SHM
                # drain thread evacuates records while control/analysis waits.
                control_mofka_event = control_future.wait(timeout_ms=control_wait_timeout_ms)
                if control_mofka_event is None:
                    if not eos_seen:
                        continue
                    # EOS is producer completion, not an ordered boundary: the
                    # Python EOS lane is drained before the C++ rank lanes.
                    # Finish every already-committed rank boundary before
                    # accepting the quiet control stream as terminal.
                    if window_tracker.pending_by_window:
                        known_pids = (set(window_tracker.active_window_by_pid)
                                      | set(window_tracker.completed_window_by_pid))
                        incomplete = [
                            (window, sorted(
                                pid for pid in known_pids
                                if window_tracker.completed_window_by_pid.get(pid, 0) < window
                            ))
                            for window in sorted(window_tracker.pending_by_window)
                        ]
                        raise RuntimeError(
                            "incomplete window at liveflow.eos: "
                            + "; ".join(
                                f"window {window} missing rank stops {pids}"
                                for window, pids in incomplete
                            )
                        )
                    # First account for events already read beyond the final
                    # boundary. They are tail diagnostics, never fact evidence.
                    final_window = (
                        last_completed_boundary.window_index
                        if last_completed_boundary is not None else None
                    )
                    final_targets = (
                        last_completed_boundary.events_written_by_pid
                        if last_completed_boundary is not None else {}
                    )
                    final_stops = {
                        pid: stop for pid, stop in accepted_stop_controls.items()
                        if stop["window"] == final_window
                    }
                    logger.info(
                        "mofka.stream.eos_tail_boundary",
                        final_window=final_window,
                        final_targets=final_targets,
                        final_stop_controls=final_stops,
                    )
                    eos_records = 0
                    def _record_eos_tail(normalized, ts_us, pid, *, source, lane=None,
                                         sequence=None, batch_index=None):
                        nonlocal eos_records
                        eos_records += 1
                        logger.info(
                            "mofka.stream.eos_tail_event",
                            ordinal=eos_records,
                            source=source,
                            lane=lane,
                            sequence=sequence,
                            batch_index=batch_index,
                            pid=pid,
                            ts_us=ts_us,
                            final_window=final_window,
                            final_target=final_targets.get(pid),
                            final_stop_control=final_stops.get(pid),
                            trace_event=normalized,
                        )

                    while pending_trace_events:
                        normalized, ts_us, pid = pending_trace_events.popleft()
                        _record_eos_tail(normalized, ts_us, pid, source="pending")
                    eos_deadline = time.monotonic() + (selected_trace_drain_grace_ms / 1000.0)
                    while True:
                        remaining_ms = int(max(0.0, (eos_deadline - time.monotonic()) * 1000.0))
                        if remaining_ms <= 0:
                            break
                        trace_mofka_event = trace_future.wait(
                            timeout_ms=min(trace_drain_timeout_ms, remaining_ms)
                        )
                        if trace_mofka_event is None:
                            continue
                        for batch_index, (normalized, ts_us, pid) in enumerate(
                            _normalize_trace_event(trace_mofka_event)
                        ):
                            _record_eos_tail(
                                normalized, ts_us, pid, source="ring",
                                lane=getattr(trace_mofka_event, "lane", None),
                                sequence=getattr(trace_mofka_event, "seq", None),
                                batch_index=batch_index,
                            )
                        trace_future = trace_consumer.pull()
                        eos_deadline = time.monotonic() + (selected_trace_drain_grace_ms / 1000.0)
                    logger.info("mofka.stream.eos_drained", unassociated_trace_events=eos_records)
                    return

                control_future = control_consumer.pull()
                control_event = control_mofka_event.metadata
                if not isinstance(control_event, dict) or control_event.get("type") != "boundary_event":
                    if hasattr(control_mofka_event, "acknowledge"):
                        _ack_event(control_mofka_event)
                    continue

                # The launcher publishes this only after every application rank has
                # exited. SHM lanes have no global order: EOS can be yielded
                # before already-committed rank stops in higher-numbered lanes.
                # Defer finalization until those controls have passed through
                # the ordinary window assembly path.
                if control_event.get("trigger_event_name") == "liveflow.eos":
                    if hasattr(control_mofka_event, "acknowledge"):
                        _ack_event(control_mofka_event)
                    eos_seen = True
                    continue

                try:
                    pid = int(control_event["pid"])
                    events_written = int(control_event["events_written"])
                    trigger_name = str(control_event.get("trigger_event_name", ""))
                except (KeyError, TypeError, ValueError):
                    logger.warning("mofka.control_boundary.invalid", metadata=control_event)
                    if hasattr(control_mofka_event, "acknowledge"):
                        _ack_event(control_mofka_event)
                    continue

                if hasattr(control_mofka_event, "acknowledge"):
                    _ack_event(control_mofka_event)

                if trigger_name == "window.start":
                    started_window = window_tracker.observe_start_boundary(
                        pid,
                        events_written=events_written,
                    )
                    logger.info(
                        "mofka.window.start",
                        pid=pid,
                        window=started_window,
                        boundary_seq=events_written,
                    )
                    continue
                close_reason = None
                if trigger_name == "window.stop":
                    close_reason = "window_stop"
                if close_reason is None:
                    logger.debug(
                        "mofka.control_boundary.ignored",
                        pid=pid,
                        trigger=trigger_name,
                        events_written=events_written,
                    )
                    continue

                stop_start_us = _parse_control_int(control_event, "trace_ts_us")
                stop_duration_us = _parse_control_int(control_event, "duration_us")
                if stop_start_us is not None and stop_start_us > 0:
                    # A stop control timestamp marks the beginning of its window.
                    # Include trace rows emitted during its valid elapsed duration;
                    # legacy controls without one retain their start-time boundary.
                    boundary_ts_us = stop_start_us + (
                        stop_duration_us
                        if stop_duration_us is not None and stop_duration_us >= 0
                        else 0
                    )
                    boundary_ts_ns = boundary_ts_us * 1000
                else:
                    boundary_ts_ns = int(control_event.get("ts_unix_ns", 0))
                stop_window = window_tracker.current_window(pid)
                previously_completed = window_tracker.completed_window_by_pid.get(pid)
                completed_windows = window_tracker.observe_end_boundary(
                    pid=pid,
                    boundary_ts_ns=boundary_ts_ns,
                    events_written=events_written,
                    discard=bool(_parse_control_int(control_event, "discard") or 0),
                )
                if (
                    stop_window is not None
                    and previously_completed != stop_window
                    and window_tracker.completed_window_by_pid.get(pid) == stop_window
                ):
                    stop_end_us = (
                        stop_start_us + stop_duration_us
                        if stop_start_us is not None and stop_start_us >= 0
                        and stop_duration_us is not None and stop_duration_us >= 0
                        else None
                    )
                    accepted_stop_controls[pid] = {
                        "window": stop_window,
                        "events_written": events_written,
                        "trace_start_ts_us": stop_start_us,
                        "duration_us": stop_duration_us,
                        "trace_end_ts_us": stop_end_us,
                        "ts_unix_ns": _parse_control_int(control_event, "ts_unix_ns"),
                        "flush": bool(_parse_control_int(control_event, "flush") or 0),
                        "discard": bool(_parse_control_int(control_event, "discard") or 0),
                    }
                    logger.info(
                        "mofka.window.stop",
                        pid=pid,
                        window=stop_window,
                        boundary_seq=events_written,
                    )

                for completed_window in completed_windows:
                    last_completed_boundary = completed_window
                    _discard_values = list(completed_window.discard_by_pid.values())
                    _all_discarded = (
                        len(_discard_values) == num_ranks and all(_discard_values)
                    )
                    _mixed_discard = bool(_discard_values) and not (
                        all(_discard_values) or not any(_discard_values)
                    )
                    _window_wall_start = time.perf_counter()
                    # Where the drain's time goes, split into the four things it can be doing.
                    # `drain_elapsed_ms` alone cannot say whether the analyzer was waiting for a
                    # rank that had not written yet, waiting out a poll timeout for a batch that
                    # was never coming, or decoding JSON it already had.
                    _dstat = {
                        "counted_ms": 0.0, "sweep_ms": 0.0, "carry_ms": 0.0, "decode_ms": 0.0,
                        "polls_data": 0, "polls_empty": 0, "empty_wait_ms": 0.0,
                        "settle_ms": -1,
                    }
                    logger.info(
                        "mofka.window.block",
                        window=completed_window.window_index,
                        ranks_received=completed_window.ranks_received,
                        num_ranks=num_ranks,
                        boundary_ts_ns=completed_window.boundary_ts_ns,
                        close_reason=close_reason,
                        trigger=trigger_name,
                        step=_parse_control_int(control_event, "step"),
                        epoch=_parse_control_int(control_event, "epoch"),
                    )

                    # Drain trace events up to this window boundary's timestamp.
                    pulled_events = []
                    target_counts = {
                        int(boundary_pid): int(boundary_count)
                        for boundary_pid, boundary_count in completed_window.events_written_by_pid.items()
                        if boundary_count is not None and int(boundary_count) > 0
                    }
                    boundary_ts_us = (
                        completed_window.boundary_ts_ns // 1000
                        if completed_window.boundary_ts_ns
                        else 0
                    )

                    if target_counts:
                        # SHM rank boundaries follow the trace writes they count.
                        # Capture each lane's committed head once so unrelated
                        # later arrivals cannot extend this window's work.
                        trace_snapshot = (
                            trace_consumer.snapshot_committed()
                            if hasattr(trace_consumer, "snapshot_committed") else None
                        )
                        start_counts = {
                            int(boundary_pid): int(boundary_count)
                            for boundary_pid, boundary_count in completed_window.start_events_written_by_pid.items()
                            if boundary_count is not None and int(boundary_count) > 0
                        }
                        # The grace is an unavailable-data wait budget, not a limit on
                        # decoding records that are already committed in SHM.  In
                        # particular the first window can contain a large backlog.
                        idle_wait_ms = 0.0
                        while not _targets_satisfied(target_counts):
                            if _shutdown_requested:
                                raise RuntimeError("shutdown during trace target drain")
                            pending_event = _pop_pending_event_for_window(
                                target_counts, boundary_ts_us
                            )
                            if pending_event is not None:
                                normalized_event, event_ts, event_pid = pending_event
                                _record_trace_event(
                                    normalized_event,
                                    event_pid,
                                    pulled_events,
                                    window_index=completed_window.window_index,
                                    start_counts=start_counts,
                                )
                                continue

                            # Check the consumer's existing buffer and committed ring
                            # records before charging any wait budget.  Processing time
                            # cannot turn those records into an incomplete window.
                            _t0 = time.perf_counter()
                            trace_mofka_event = (
                                trace_snapshot.next() if trace_snapshot is not None
                                else trace_future.wait(timeout_ms=0)
                            )
                            _waited = (time.perf_counter() - _t0) * 1000.0
                            _dstat["counted_ms"] += _waited
                            if trace_mofka_event is None:
                                if trace_snapshot is not None:
                                    break
                                remaining_ms = selected_trace_drain_grace_ms - idle_wait_ms
                                if remaining_ms <= 0:
                                    break
                                poll_ms = min(trace_drain_timeout_ms, max(1, int(remaining_ms)))
                                _t0 = time.perf_counter()
                                trace_mofka_event = trace_future.wait(timeout_ms=poll_ms)
                                _waited = (time.perf_counter() - _t0) * 1000.0
                                _dstat["counted_ms"] += _waited
                                if trace_mofka_event is None:
                                    idle_wait_ms += _waited
                                    _dstat["polls_empty"] += 1
                                    _dstat["empty_wait_ms"] += _waited
                                    continue
                            _dstat["polls_data"] += 1
                            _t0 = time.perf_counter()
                            unpacked = _normalize_trace_event(trace_mofka_event)
                            _dstat["decode_ms"] += (time.perf_counter() - _t0) * 1000.0
                            if trace_snapshot is None:
                                trace_future = trace_consumer.pull()
                            for normalized_event, event_ts, event_pid in unpacked:
                                if (
                                    _event_exceeds_target(event_pid, target_counts)
                                    or (
                                        event_pid not in target_counts
                                        and boundary_ts_us and event_ts
                                        and event_ts > boundary_ts_us
                                    )
                                ):
                                    pending_trace_events.append((normalized_event, event_ts, event_pid))
                                    continue
                                _record_trace_event(
                                    normalized_event,
                                    event_pid,
                                    pulled_events,
                                    window_index=completed_window.window_index,
                                    start_counts=start_counts,
                                )

                        # SWEEP EVENTS THE TARGETS CANNOT ACCOUNT FOR.
                        #
                        # `target_counts` is built from the boundary's events_written_by_pid, and a
                        # boundary is only ever emitted by the process that calls window.start/stop
                        # -- the training loop. Forked DataLoader workers never call it, so their
                        # pids can NEVER appear in target_counts. The loop above therefore stops the
                        # instant the main pid's count is reached, which it already is, and every
                        # worker event stays queued in pending_trace_events and is carried forward
                        # forever. Measured on a 1-node DLIO run: windows closed with event_count
                        # 1-3 (FileStorage.get_uri, opendir, __xstat64), seen_counts EXACTLY equal
                        # to target_counts, drain_elapsed 0.1-1.9 ms -- while the ring held 151,787
                        # events, 99,944 of them POSIX reads. analysis_facts was 0 every window,
                        # because the rules key on read latency and not one read ever entered a
                        # window. The pipeline looked alive and was measuring almost nothing.
                        #
                        # The timestamp branch below already drains pending up to the boundary; this
                        # is that same sweep, which the counted branch was missing.
                        #
                        # Sweeping `pending` alone is not enough: worker batches that have not been
                        # PULLED yet are still sitting in the ring, and the counted loop above stops
                        # pulling the moment the main pid's target is met. So first drain whatever
                        # the consumer can hand over right now, up to the boundary timestamp.
                        # `break` on an empty pull rather than `continue`: the records are already
                        # buffered by the time the boundary arrives, so there is nothing to wait
                        # for, and waiting out the full grace on every window would add ~30 s each.
                        settle_ms = drain_settle_ms(
                            targets_satisfied=_targets_satisfied(target_counts),
                            timeout_ms=trace_drain_timeout_ms,
                            override=drain_settle_ms_override,
                        )
                        _dstat["settle_ms"] = settle_ms
                        drain_deadline = time.monotonic() + (settle_ms / 1000.0)
                        past_boundary = False
                        while trace_snapshot is not None or not past_boundary:
                            if _shutdown_requested:
                                raise RuntimeError("shutdown during trace timestamp sweep")
                            if trace_snapshot is not None and trace_snapshot.remaining == 0:
                                break
                            remaining_ms = int(
                                max(
                                    0.0,
                                    min(
                                        float(settle_ms),
                                        (drain_deadline - time.monotonic()) * 1000.0,
                                    ),
                                )
                            )
                            # A zero settle still DRAINS: shm_io.Consumer._next reads every lane
                            # into its buffer before it consults the deadline, so a non-blocking
                            # pull returns everything already in the ring and only then None.
                            # Breaking on remaining_ms <= 0 would skip that drain entirely.
                            if trace_snapshot is None and settle_ms > 0 and remaining_ms <= 0:
                                break
                            _t0 = time.perf_counter()
                            trace_mofka_event = (
                                trace_snapshot.next() if trace_snapshot is not None
                                else trace_future.wait(timeout_ms=remaining_ms)
                            )
                            _waited = (time.perf_counter() - _t0) * 1000.0
                            _dstat["sweep_ms"] += _waited
                            if trace_mofka_event is None:
                                _dstat["polls_empty"] += 1
                                _dstat["empty_wait_ms"] += _waited
                                break
                            _dstat["polls_data"] += 1
                            _t0 = time.perf_counter()
                            unpacked = _normalize_trace_event(trace_mofka_event)
                            _dstat["decode_ms"] += (time.perf_counter() - _t0) * 1000.0
                            if trace_snapshot is None:
                                trace_future = trace_consumer.pull()
                            for normalized_event, event_ts, event_pid in unpacked:
                                if (
                                    _event_exceeds_target(event_pid, target_counts)
                                    or (boundary_ts_us and event_ts and event_ts > boundary_ts_us)
                                ):
                                    pending_trace_events.append(
                                        (normalized_event, event_ts, event_pid)
                                    )
                                    past_boundary = True
                                    continue
                                _record_trace_event(
                                    normalized_event,
                                    event_pid,
                                    pulled_events,
                                    window_index=completed_window.window_index,
                                    start_counts=start_counts,
                                )

                        if trace_snapshot is not None:
                            trace_snapshot.close()
                            trace_snapshot = None

                        _t0 = time.perf_counter()
                        for normalized_event, event_ts, event_pid in pending_trace_events.sweep_for_window(
                            target_counts, trace_events_seen_by_pid, boundary_ts_us
                        ):
                            _record_trace_event(
                                normalized_event,
                                event_pid,
                                pulled_events,
                                window_index=completed_window.window_index,
                                start_counts=start_counts,
                            )
                        _dstat["carry_ms"] = (time.perf_counter() - _t0) * 1000.0
                    else:
                        can_drain_trace_stream = True
                        drain_deadline = (
                            time.monotonic()
                            + (selected_trace_drain_grace_ms / 1000.0)
                        )

                        # Drain any buffered events from previous batch splits.
                        while pending_trace_events and can_drain_trace_stream:
                            normalized_event, event_ts, _pid = pending_trace_events.popleft()
                            if boundary_ts_us and event_ts and event_ts > boundary_ts_us:
                                pending_trace_events.appendleft((normalized_event, event_ts, _pid))
                                can_drain_trace_stream = False
                            else:
                                _append_trace_event_to_window(
                                    normalized_event,
                                    pulled_events,
                                    window_index=completed_window.window_index,
                                )

                        if can_drain_trace_stream:
                            past_boundary = False
                            while not past_boundary:
                                remaining_ms = int(
                                    max(
                                        0.0,
                                        min(
                                            float(trace_drain_timeout_ms),
                                            (drain_deadline - time.monotonic()) * 1000.0,
                                        ),
                                    )
                                )
                                if remaining_ms <= 0:
                                    break
                                trace_mofka_event = trace_future.wait(timeout_ms=remaining_ms)
                                if trace_mofka_event is None:
                                    continue
                                unpacked = _normalize_trace_event(trace_mofka_event)
                                trace_future = trace_consumer.pull()
                                for normalized_event, event_ts, event_pid in unpacked:
                                    # Buffer events past the window boundary for the next window.
                                    if boundary_ts_us and event_ts and event_ts > boundary_ts_us:
                                        pending_trace_events.append((normalized_event, event_ts, event_pid))
                                        past_boundary = True
                                        continue
                                    _append_trace_event_to_window(
                                        normalized_event,
                                        pulled_events,
                                        window_index=completed_window.window_index,
                                    )

                    if target_counts and not _targets_satisfied(target_counts):
                        missing_counts = {
                            pid: {"seen": trace_events_seen_by_pid.get(pid, 0), "target": target}
                            for pid, target in target_counts.items()
                            if trace_events_seen_by_pid.get(pid, 0) < target
                        }
                        raise RuntimeError(
                            "incomplete trace targets for window "
                            f"{completed_window.window_index}: {missing_counts}"
                        )

                    if not pulled_events:
                        logger.warning(
                            "mofka.window.empty",
                            window=completed_window.window_index,
                            close_reason=close_reason,
                            trigger=trigger_name,
                            target_counts=target_counts,
                            boundary_ts_us=boundary_ts_us,
                        )
                        continue

                    # A boundary opened before next(loader) can discover end-of-input.  It must
                    # be drained and completed above so the next epoch retains a valid tracker
                    # identity, but it contains no completed workload step.  Never turn it (or a
                    # mixed-rank report) into analysis, a fact, an optimizer observation, or a
                    # horizon input.  The explicit application flag is authoritative; incidental
                    # trace activity and zero/positive wall time are not eligibility heuristics.
                    if _all_discarded or _mixed_discard:
                        logger.info(
                            "mofka.window.control_excluded",
                            window=completed_window.window_index,
                            reason=("all_ranks_discarded" if _all_discarded
                                    else "mixed_rank_discard"),
                            discard_by_pid=completed_window.discard_by_pid,
                            event_count=len(pulled_events),
                        )
                        continue

                    # Log event category breakdown for debugging
                    cat_counts = Counter(e.get("cat", "?") for e in pulled_events)
                    name_sample = Counter(e.get("name", "?") for e in pulled_events)
                    _drain_elapsed_ms = round((time.perf_counter() - _window_wall_start) * 1000, 1)
                    # True skew at window close, on the application's own clock: the spread of
                    # the ranks' own window.stop timestamps. This is the part of the wait that is
                    # the workload's and cannot be engineered away from here.
                    _by_pid = completed_window.boundary_ts_ns_by_pid or {}
                    _skew_ms = (
                        round((max(_by_pid.values()) - min(_by_pid.values())) / 1e6, 2)
                        if len(_by_pid) > 1 else None
                    )
                    logger.info(
                        "mofka.window.drain_summary",
                        window=completed_window.window_index,
                        drain_elapsed_ms=_drain_elapsed_ms,
                        rank_skew_ms=_skew_ms,
                        counted_ms=round(_dstat["counted_ms"], 1),
                        sweep_ms=round(_dstat["sweep_ms"], 1),
                        carry_ms=round(_dstat["carry_ms"], 1),
                        decode_ms=round(_dstat["decode_ms"], 1),
                        polls_data=_dstat["polls_data"],
                        polls_empty=_dstat["polls_empty"],
                        empty_wait_ms=round(_dstat["empty_wait_ms"], 1),
                        settle_ms=_dstat["settle_ms"],
                        event_count=len(pulled_events),
                        target_counts=target_counts,
                        seen_counts={pid: trace_events_seen_by_pid.get(pid, 0) for pid in sorted(target_counts)},
                        cat_counts=dict(cat_counts),
                        top_names=dict(name_sample.most_common(10)),
                    )

                    # Replay the run's hash tables into this window. They are not events and
                    # carry no window, but without them every file_hash in the window resolves to
                    # nothing. Prepended rather than appended so the frame's first rows establish
                    # the string dtypes for the merge keys.
                    traces, profiles, system_metrics = self.handle_stream_events(
                        events=(list(stream_metadata_events.values()) + pulled_events
                                if stream_metadata_events else pulled_events),
                        view_types=proc_view_types,
                        extra_columns=extra_columns,
                    )
                    _traces_bytes = int(traces.memory_usage(deep=True).sum()) if hasattr(traces, "memory_usage") else -1
                    logger.debug(
                        "mofka.window.traces_memory",
                        window=completed_window.window_index,
                        traces_rows=len(traces),
                        traces_bytes=_traces_bytes,
                        traces_mb=round(_traces_bytes / 1024 / 1024, 1),
                    )
                    # This is provenance for the streaming epoch contract, not a repair.  The
                    # control tracker assigned this completed window's identity before the frame
                    # was standardised; record whether that identity survived into the analyzer
                    # frame and whether the duration-bearing wall events retained it.  A later
                    # domain/slope result is invalid if either count is zero, and the compact
                    # counts let a smoke distinguish an emitter failure from a dataframe-path
                    # loss without dumping workload traces into the agent log.
                    if {"epoch", "cat", "func_name"}.issubset(traces.columns):
                        _epoch_present = int(traces["epoch"].notna().sum())
                        _wall_mask = (
                            (traces["cat"] == "window")
                            & (traces["func_name"] == "window.stop")
                        )
                        _wall_total = int(_wall_mask.sum())
                        _wall_epoch_present = int(
                            (traces.loc[_wall_mask, "epoch"].notna()).sum()
                        )
                    else:
                        _epoch_present = _wall_total = _wall_epoch_present = 0
                    logger.info(
                        "mofka.window.epoch_provenance",
                        window=completed_window.window_index,
                        traces_rows=len(traces),
                        epoch_rows=_epoch_present,
                        wall_rows=_wall_total,
                        wall_epoch_rows=_wall_epoch_present,
                    )
                    # `_analyze_trace()` writes debug flat views internally.  Make the
                    # completed control-window identity available before entering it so those
                    # artifacts remain joinable to the ring/control evidence after the run.
                    # Do not infer a rank from a row: the control tracker is the authority for
                    # the complete rank set and boundary times.
                    self._debug_window_provenance = {
                        "window_index": completed_window.window_index,
                        "rank_pids": sorted(completed_window.boundary_ts_ns_by_pid or {}),
                        "boundary_ts_ns_by_pid": completed_window.boundary_ts_ns_by_pid or {},
                    }
                    self._write_debug_window_integrity_ledger(traces)
                    # Fact rules receive the completed control-window identity
                    # and PID membership directly from the stream, not from the
                    # debug CSV.  A proc_name maps only when its trace rows
                    # unambiguously name one numeric PID; ambiguous/missing
                    # mappings are deliberately omitted and worker-only rules
                    # then fail closed.
                    row_pid_by_index = {}
                    trace_pids = set()
                    if {"proc_name", process_key}.issubset(traces.columns):
                        for proc_name, group in traces[["proc_name", process_key]].dropna().groupby("proc_name"):
                            pids = {int(pid) for pid in group[process_key]}
                            trace_pids.update(pids)
                            if len(pids) == 1:
                                row_pid_by_index[proc_name] = pids.pop()
                    rank_pids = {int(pid) for pid in (completed_window.boundary_ts_ns_by_pid or {})}
                    from .fastpath import (
                        completed_window_stall_storage_provenance,
                        completed_window_worker_roster_provenance,
                    )
                    roster_requested = any(value is not None for value in (
                        worker_roster_dir, worker_roster_prefix,
                        worker_roster_expected_workers,
                    ))
                    roster_provenance = None
                    if roster_requested:
                        roster_provenance = completed_window_worker_roster_provenance(
                            traces,
                            row_pid_by_index,
                            roster_dir=worker_roster_dir,
                            roster_prefix=worker_roster_prefix,
                            expected_workers=worker_roster_expected_workers,
                            expected_host=socket.gethostname(),
                            expected_ranks=num_ranks,
                            expected_rank_pids=rank_pids,
                            pid_column=process_key,
                            # None keeps the DLIO default; an unknown name fails closed.
                            **({"roster_schema": worker_roster_schema}
                               if worker_roster_schema is not None else {}),
                        )
                        worker_pids = (
                            roster_provenance["worker_roster_pids"]
                            if roster_provenance["worker_roster_coverage"] == 1.0 else []
                        )
                    else:
                        worker_pids = sorted(trace_pids - rank_pids)
                    self._fact_window_context = {
                        "native_window": completed_window.window_index,
                        "native_host": socket.gethostname(),
                        "rank_pids": sorted(rank_pids),
                        "worker_pids": worker_pids,
                        "row_pid_by_index": row_pid_by_index,
                        "completed_window_provenance": completed_window_stall_storage_provenance(
                            traces, rank_pids, pid_column=process_key),
                    }
                    if hasattr(self.fact_engine, "liveflow_declaration"):
                        from .liveflow_v1 import window_trace_integrity
                        dropped_records = None
                        if os.environ.get("LIVEFLOW_TRANSPORT") == "liveflow":
                            from liveflow_transport.shm_io import ring_totals
                            totals = ring_totals().get(topic_name)
                            # An unavailable SHM counter is unknown integrity.
                            dropped_records = totals[1] if totals is not None else -1
                        integrity = window_trace_integrity(
                            traces, rank_pids,
                            window_horizon_metrics(pulled_events, control_event),
                            dropped_records=dropped_records,
                            global_steps=self.fact_engine.liveflow_declaration["workload"] == "halo3d",
                        )
                        self._fact_window_context["trace_integrity"] = integrity
                        logger.info("window.trace_integrity", window=completed_window.window_index,
                                    **integrity)
                    if roster_provenance is not None:
                        self._fact_window_context["completed_window_worker_roster_provenance"] = roster_provenance
                    try:
                        # The fast path serves the threshold-free *-signal rule sets directly
                        # from the traces frame and declines anything else, so a rule set it
                        # cannot serve is slow, never wrong.  Section 4j: the full path costs
                        # 8.7 s for an 82 ms window at 32 nodes, which is what made the agent
                        # sample 1 window in 350 and the ring lanes overflow behind it.
                        result = self._analyze_trace_fast(
                            traces=traces,
                            proc_view_types=proc_view_types,
                            metric_boundaries=metric_boundaries,
                            raw_stats={},
                        )
                        _fastpath_used = result is not None
                        if result is None:
                            result = self._analyze_trace(
                                traces=traces,
                                proc_view_types=proc_view_types,
                                logical_view_types=logical_view_types,
                                raw_stats={},
                                metric_boundaries=metric_boundaries,
                                profiles=profiles,
                                system_metrics=system_metrics,
                            )
                        result.window_index = completed_window.window_index
                        horizon_metrics = window_horizon_metrics(
                            pulled_events,
                            control_event=control_event,
                        )
                        if horizon_metrics:
                            attach_window_horizon(result, horizon_metrics)
                            logger.info(
                                "window.horizon.published",
                                window=completed_window.window_index,
                                **horizon_metrics,
                            )
                        else:
                            logger.warning(
                                "window.horizon.missing",
                                window=completed_window.window_index,
                                required_fields=HORIZON_SOURCE_FIELDS,
                            )
                    except KeyError as exc:
                        trace_columns = list(traces.columns) if hasattr(traces, "columns") else []
                        logger.warning(
                            "mofka.window.analysis_skipped_missing_column",
                            window=completed_window.window_index,
                            missing_column=str(exc),
                            columns=trace_columns,
                            event_count=len(pulled_events),
                            target_counts=target_counts,
                        )
                        continue
                    finally:
                        # A later offline call must not inherit a stale online-window identity.
                        self._debug_window_provenance = None
                        self._fact_window_context = None
                    _analysis_elapsed_ms = round((time.perf_counter() - _window_wall_start) * 1000, 1)
                    # The release rule of section 4j: analysis_elapsed_ms must be below the
                    # window's own wall-clock span, else the agent is sampling rather than
                    # observing. Report both on the same line so the check needs no join.
                    _window_span_ms = _drained_window_span_ms(pulled_events)
                    logger.info(
                        "mofka.window.analysis_complete",
                        window=completed_window.window_index,
                        analysis_elapsed_ms=_analysis_elapsed_ms,
                        drain_elapsed_ms=_drain_elapsed_ms,
                        window_span_ms=_window_span_ms,
                        fastpath=_fastpath_used,
                        event_count=len(pulled_events),
                        num_ranks=num_ranks,
                        lane_occupancy=_lane_occupancy(),
                        flat_views=len(result.flat_views),
                        analysis_facts=len(result.analysis_facts),
                    )
                    _publish_start = time.perf_counter()
                    output_handler(result)
                    _publish_elapsed_ms = round((time.perf_counter() - _publish_start) * 1000, 1)
                    _e2e_elapsed_ms = round((time.perf_counter() - _window_wall_start) * 1000, 1)
                    logger.info(
                        "mofka.window.publish_complete",
                        window=completed_window.window_index,
                        publish_elapsed_ms=_publish_elapsed_ms,
                        e2e_elapsed_ms=_e2e_elapsed_ms,
                    )

        finally:
            if trace_snapshot is not None:
                trace_snapshot.close()
            if continuous_trace_consumer is not None:
                continuous_trace_consumer.close()
            logger.info("mofka.control_stream.stop", reason="sigterm")
            del control_consumer
            del control_driver
            del trace_consumer
            del trace_driver

    def analyze_mofka(
        self,
        group_file: str,
        topic_name: str,
        view_types: List[ViewType],
        exclude_characteristics: List[str] = [],
        logical_view_types: bool = False,
        metric_boundaries: ViewMetricBoundaries = {},
        trace_drain_grace_ms: int = 5000,
        process_key: str = "pid",
        control_topic_name: Optional[str] = None,
        trace_consumer_name: Optional[str] = None,
        control_consumer_name: Optional[str] = None,
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        output_handler: Optional[Callable[[AnalysisResult], None]] = None,
        num_ranks: int = 1,
        worker_roster_dir: Optional[str] = None,
        worker_roster_prefix: Optional[str] = None,
        worker_roster_expected_workers: Optional[int] = None,
        worker_roster_schema: Optional[str] = None,
    ) -> None:
        resolved_control_topic = (
            control_topic_name
            if control_topic_name is not None
            else os.getenv("DFTRACER_MOFKA_CONTROL_TOPIC_NAME", "")
        )
        if not resolved_control_topic:
            resolved_control_topic = "control_events"

        logger.debug(
            "Mofka analyzer window mode",
            trace_topic=topic_name,
            control_topic=resolved_control_topic,
            num_ranks=num_ranks,
        )
        self._analyze_mofka_with_control(
            group_file=group_file,
            topic_name=topic_name,
            control_topic_name=resolved_control_topic,
            view_types=view_types,
            exclude_characteristics=exclude_characteristics,
            logical_view_types=logical_view_types,
            metric_boundaries=metric_boundaries,
            trace_drain_grace_ms=trace_drain_grace_ms,
            process_key=process_key,
            trace_consumer_name=trace_consumer_name,
            control_consumer_name=control_consumer_name,
            extra_columns=extra_columns,
            extra_columns_fn=extra_columns_fn,
            output_handler=output_handler,
            num_ranks=num_ranks,
            worker_roster_dir=worker_roster_dir,
            worker_roster_prefix=worker_roster_prefix,
            worker_roster_expected_workers=worker_roster_expected_workers,
            worker_roster_schema=worker_roster_schema,
        )

    def analyze_mofka_global(
        self,
        group_file: str,
        input_topic: str,
        view_types: List[ViewType],
        num_nodes: int,
        metric_boundaries: ViewMetricBoundaries = {},
        logical_view_types: bool = False,
        output_handler: Optional[Callable[["AnalysisResult"], None]] = None,
        idle_timeout_sec: int = 300,
        pull_timeout_ms: int = 500,
        consumer_name: Optional[str] = None,
    ) -> None:
        """Global analyzer: consumes HLMs from per-node analyzers, concatenates, and analyzes.

        Each per-node analyzer publishes its HLM (parquet) to the input_topic.
        This method buffers HLMs by window_index, waits for all num_nodes to
        report, concatenates them, and runs _analyze_hlm with the given
        view_types (typically [host_hash] for cross-node analysis).
        """
        import io as _io

        from .streaming.transport import open_consumer

        install_shutdown_handler()

        driver, consumer = open_consumer(
            group_file, input_topic, consumer_name, use_progress_thread=True,
        )

        # Buffer: window_index -> {source_node: hlm_df}
        window_buffer: Dict[int, Dict[str, pd.DataFrame]] = {}

        future = consumer.pull()

        logger.info(
            "global.analyzer.start",
            input_topic=input_topic,
            num_nodes=num_nodes,
            view_types=view_types,
        )

        try:
            while not _shutdown_requested:
                event = future.wait(timeout_ms=pull_timeout_ms)
                if event is None:
                    continue

                metadata = event.metadata
                data = event.data
                if isinstance(data, list):
                    data = b"".join(data)

                window_index = metadata.get("window_index", -1)
                source_node = metadata.get("source_node", "unknown")

                hlm_df = pd.read_parquet(_io.BytesIO(data))
                # Restore the index from the sender's metadata.
                hlm_index_names = metadata.get("hlm_index_names", [])
                if hlm_index_names:
                    valid_idx = [c for c in hlm_index_names if c in hlm_df.columns]
                    if valid_idx:
                        hlm_df = hlm_df.set_index(valid_idx)
                        if "host_hash" in hlm_df.index.names:
                            annotated = hlm_df.reset_index()
                            annotated["host_hash"] = source_node
                            hlm_df = annotated.set_index(valid_idx)

                logger.info(
                    "global.hlm.received",
                    window_index=window_index,
                    source_node=source_node,
                    hlm_rows=len(hlm_df),
                    hlm_index=hlm_index_names,
                )

                _ack_event(event)
                future = consumer.pull()

                if window_index not in window_buffer:
                    window_buffer[window_index] = {}
                window_buffer[window_index][source_node] = hlm_df

                if len(window_buffer[window_index]) >= num_nodes:
                    nodes_hlms = window_buffer.pop(window_index)
                    merged_hlm = pd.concat(list(nodes_hlms.values()))

                    logger.info(
                        "global.window.complete",
                        window_index=window_index,
                        nodes=sorted(nodes_hlms.keys()),
                        merged_rows=len(merged_hlm),
                        columns=list(merged_hlm.columns),
                    )

                    # Dump merged HLM to file for debugging
                    try:
                        hlm_path = os.path.join(
                            self.checkpoint_dir or ".",
                            f"global_hlm_window_{window_index}.parquet",
                        )
                        merged_hlm.to_parquet(hlm_path)
                        logger.debug("global.hlm.saved", path=hlm_path)
                    except Exception as exc:
                        logger.warning("global.hlm.save_failed", error=str(exc))

                    try:
                        result = self._analyze_hlm(
                            hlm=merged_hlm.reset_index(),
                            proc_view_types=view_types,
                            metric_boundaries=metric_boundaries,
                            raw_stats={},
                            logical_view_types=logical_view_types,
                            is_dask=False,
                        )
                        result.view_types = view_types
                        result.window_index = window_index

                        logger.info(
                            "global.window.analysis_complete",
                            window_index=window_index,
                            flat_views=len(result.flat_views),
                            analysis_facts=len(result.analysis_facts),
                        )

                        if output_handler:
                            output_handler(result)

                    except Exception as exc:
                        logger.warning(
                            "global.window.analysis_failed",
                            window_index=window_index,
                            error=str(exc),
                            exc_info=True,
                        )
        finally:
            logger.info("global.analyzer.shutdown",
                        pending_windows=list(window_buffer.keys()))

    def read_stats(self, traces: dd.DataFrame, profiles: Optional[dd.DataFrame] = None) -> RawStats:
        """Computes and restores raw statistics from the trace data.

        Calculates job time and total event count from the traces.
        It attempts to restore these stats from a checkpoint if available,
        otherwise computes them and checkpoints the result.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            A RawStats dictionary containing 'job_time', 'time_granularity',
            and 'total_count'.
        """
        job_time = self.get_job_time(traces)
        trace_event_count = self.get_total_event_count(traces)
        profile_event_count = self.get_profile_event_count(profiles)
        total_event_count = trace_event_count + profile_event_count
        unique_file_count = self.get_unique_file_count(traces)
        unique_host_count = self.get_unique_host_count(traces)
        unique_process_count = self.get_unique_process_count(traces)
        raw_stats_data = self.restore_extra_data(
            name=self.get_stats_checkpoint_name(),
            fallback=lambda: dict(
                job_time=job_time,
                time_granularity=self.time_granularity,
                time_resolution=self.time_resolution,
                trace_event_count=trace_event_count,
                profile_event_count=profile_event_count,
                total_event_count=total_event_count,
                unique_file_count=unique_file_count,
                unique_host_count=unique_host_count,
                unique_process_count=unique_process_count,
            ),
        )
        if "trace_event_count" not in raw_stats_data:
            raw_stats_data["trace_event_count"] = raw_stats_data.get("total_event_count", 0)
        if "profile_event_count" not in raw_stats_data:
            raw_stats_data["profile_event_count"] = 0
        if "total_event_count" not in raw_stats_data:
            raw_stats_data["total_event_count"] = (
                raw_stats_data["trace_event_count"] + raw_stats_data["profile_event_count"]
            )
        raw_stats = RawStats(**raw_stats_data)
        _assert_metadata_resolved(raw_stats_data)
        return raw_stats

    @abc.abstractmethod
    def read_trace(
        self,
        trace_path: str,
        extra_columns: Optional[Dict[str, str]],
        extra_columns_fn: Optional[Callable[[dict], dict]],
    ) -> ReadTraceResult:
        """Reads I/O trace data from the specified path.

        This is an abstract method that must be implemented by subclasses
        to handle specific trace formats.

        Args:
            trace_path: Path to the I/O trace file or directory.

        Returns:
            A ReadTraceResult containing the parsed I/O trace data and any
            additional native profile streams for the analyzer.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def read_zmq(
        self,
        trace_address: str,
        extra_columns: Optional[Dict[str, str]],
        extra_columns_fn: Optional[Callable[[dict], dict]],
    ):
        raise RuntimeError("read_zmq is deprecated. Use analyze_zmq with output_handler.")

    def postread_trace(self, traces: dd.DataFrame, view_types: List[ViewType]) -> dd.DataFrame:
        """Performs any post-processing on the raw trace data.

        This method can be overridden by subclasses to perform additional
        transformations or filtering on the trace data after it has been read.
        By default, it returns the traces unmodified.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            A Dask DataFrame with any post-processing applied.
        """
        return traces

    def postread_zmq(
        self,
        trace_stream,
        view_types: List[ViewType],
        extra_columns: Optional[Dict[str, str]],
        extra_columns_fn: Optional[Callable[[dict], dict]],
    ):
        return trace_stream

    def normalize_trace_event(
        self,
        event: dict,
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
    ) -> dict:
        return event

    @staticmethod
    def _attach_stream_window(
        normalized_event: dict,
        current_window: int,
        *,
        native_window: bool,
    ) -> dict:
        """Attach control-window membership without corrupting workload metadata.

        The historical live grouping is keyed by ``epoch`` and uses the
        completed control-window number as a compatibility alias for epoch and
        step.  A native ``window`` view preserves genuine epoch/step values
        parsed from the trace instead.
        """
        event = dict(normalized_event)
        event["window"] = current_window
        if not native_window:
            event["epoch"] = current_window
            event["step"] = current_window
        return event

    @staticmethod
    def assign_window_epochs(traces: pd.DataFrame) -> pd.DataFrame:
        """Give epoch-less rows the epoch of the window they fall inside, by TIMESTAMP RANGE.

        THE PROBLEM. The high-level views group by every view type at once and pandas drops a row
        whose key is NA in any of them, so a row carrying no `epoch` is absent from EVERY view. The
        rows that carry no epoch are exactly the ones that matter for storage: POSIX reads are
        emitted by the tracer's syscall interception on reader threads and forked DataLoader
        workers, which never call `dft_ai.update(epoch=...)`, so `reader_posix_*` came out NA and
        the planner had no measurement of what the storage layer was doing. That is
        ANALYZER-FASTPATH.md section 4, and it is the metric the barrier-attribution rule (T3-0
        section 4m, user decision option 1) needs in order to compare a node's read bandwidth
        against its measured Lustre ceiling.

        THE FIX, and why it is a range and not a tag. The application cannot tag a syscall it does
        not make. But every window is bracketed by `window.start` and `window.stop` events which DO
        carry the epoch, on the same pid, and the `window.stop` event is duration-bearing, so its
        [time_start, time_end] IS the window's span on the application's own clock. A row of that
        pid whose start falls inside that span belongs to that window, whatever thread emitted it.

        DELIBERATELY DIFFERENT FROM `_set_epochs`, which exists on the batch path and cannot be
        reused here. That one begins `df["epoch"] = pd.NA` and rebuilds every epoch from the
        `pipeline`/`epoch*` REGIONS. Those regions are emitted at epoch start and stop, so a single
        window drain contains none of them, and running it here would blank the epoch on every row
        instead of filling the missing ones. This never overwrites an epoch the application set: it
        only fills where the value is absent, so an application tag always wins.

        No-op, at the cost of two column checks, when there are no window rows or nothing is
        missing.
        """
        if traces is None or getattr(traces, "empty", True):
            return traces
        cols = traces.columns
        needed = (COL_CATEGORY, COL_FUNC_NAME, COL_TIME_START, COL_TIME_END, "pid", "epoch")
        if any(c not in cols for c in needed):
            return traces

        missing = traces["epoch"].isna()
        if not missing.any():
            return traces

        stops = traces[
            (traces[COL_CATEGORY] == "window")
            & (traces[COL_FUNC_NAME] == "window.stop")
            & traces["epoch"].notna()
        ]
        if stops.empty:
            return traces

        # At most one row per rank per window, so this loop is over ranks-per-node and not over
        # events. Vectorised inside, which is what keeps it inside the fast path's budget.
        time_start = traces[COL_TIME_START]
        pids = traces["pid"]
        filled = 0
        for pid, start, end, epoch in zip(
            stops["pid"], stops[COL_TIME_START], stops[COL_TIME_END], stops["epoch"]
        ):
            mask = missing & (pids == pid) & (time_start >= start) & (time_start <= end)
            if mask.any():
                traces.loc[mask, "epoch"] = epoch
                missing = missing & ~mask
                filled += int(mask.sum())
        if filled:
            logger.debug("analyzer.window.epoch_assigned", rows=filled, windows=len(stops))
        return traces

    def handle_stream_events(
        self,
        events: List[dict],
        view_types: List[ViewType],
        extra_columns: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Split and standardize stream events into (traces, profiles, system_metrics)."""
        traces = pd.DataFrame(events)
        traces = self.postread_trace(traces=traces, view_types=view_types)
        # Before either analysis path sees the frame, so the fast path and the full path get the
        # same rows and neither has to know about this.
        traces = self.assign_window_epochs(traces)
        return traces, None, None

    def compute_job_time(self, traces: dd.DataFrame) -> float:
        """Computes the total job execution time from the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data,
                    expected to have 'tstart' and 'tend' columns.

        Returns:
            The total job time as a float.
        """
        return traces[COL_TIME_END].max() - traces[COL_TIME_START].min()

    def compute_total_count(self, traces: dd.DataFrame) -> int:
        """Computes the total number of I/O events in the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            The total count of I/O events as an integer.
        """
        return traces.index.count().persist()

    def compute_high_level_metrics(
        self,
        traces: DataFrameType,
        view_types: List[ViewType],
        partition_size: str = PARTITION_SIZE,
        checkpoint_name: Optional[str] = None,
    ) -> DataFrameType:
        """Computes high-level metrics by aggregating trace data.

        Groups the trace data by the specified view types and extra columns
        (io_cat, acc_pat, func_id) and aggregates metrics like time, count, and size.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.
            view_types: A list of column names to group by for aggregation.
            partition_size: The desired partition size for the resulting Dask DataFrame.

        Returns:
            A Dask DataFrame containing the computed high-level metrics.
        """
        checkpoint_name = checkpoint_name or self.get_hlm_checkpoint_name(view_types)
        return self.restore_view(
            name=checkpoint_name,
            fallback=lambda: self._compute_high_level_metrics(
                partition_size=partition_size,
                traces=traces,
                view_types=view_types,
            ),
        )

    def compute_profile_hlm(
        self,
        profiles: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str = PARTITION_SIZE,
        checkpoint_name: Optional[str] = None,
    ) -> dd.DataFrame:
        checkpoint_name = checkpoint_name or self.get_profile_hlm_checkpoint_name(view_types)
        return self.restore_view(
            name=checkpoint_name,
            fallback=lambda: self._compute_profile_hlm(
                partition_size=partition_size,
                profiles=profiles,
                view_types=view_types,
            ),
        )

    def reconcile_hlm(
        self,
        trace_hlm: dd.DataFrame,        
        profile_hlm: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str = PARTITION_SIZE,
        layer: Optional[Layer] = None,
    ) -> dd.DataFrame:
        hlm_groupby = list(dict.fromkeys(view_types + list(self.preset.hlm_fields)))
        trace_hlm = trace_hlm.reset_index()
        profile_hlm = profile_hlm.reset_index()
        for col in hlm_groupby:
            if col in trace_hlm.columns and col in profile_hlm.columns:
                profile_dtype = profile_hlm._meta[col].dtype
                if trace_hlm._meta[col].dtype != profile_dtype:
                    trace_hlm[col] = trace_hlm[col].astype(profile_dtype)
        trace_keys = trace_hlm[hlm_groupby].drop_duplicates().assign(_trace_present=1)
        profile_hlm = profile_hlm.merge(trace_keys, how="left", on=hlm_groupby).persist()
        overlap_count = int(profile_hlm["_trace_present"].fillna(0).sum().compute())
        if overlap_count > 0:
            logger.warning("Hybrid reconcile found overlapping HLM keys", layer=layer, overlap_count=overlap_count)
        else:
            logger.info("Hybrid reconcile found no overlapping HLM keys", layer=layer, overlap_count=overlap_count)
        profile_only = profile_hlm[profile_hlm["_trace_present"].isna()].drop(columns=["_trace_present"])
        combined_hlm = dd.concat([trace_hlm, profile_only], interleave_partitions=True)
        bin_cols = [col for col in combined_hlm.columns if "_bin_" in col]
        view_types_diff = list(set(VIEW_TYPES).difference(view_types))
        hlm_agg = dict(HLM_AGG)
        hlm_agg.update({col: "sum" for col in bin_cols})
        for col in view_types_diff:
            if col in combined_hlm.columns:
                hlm_agg[col] = unique_set_flatten()
        hlm = (
            combined_hlm.groupby(hlm_groupby)
            .agg(hlm_agg, split_out=max(1, math.ceil(math.sqrt(combined_hlm.npartitions))))
            .persist()
            .repartition(partition_size=partition_size)
            .replace(0, pd.NA)
        )
        if bin_cols:
            hlm[bin_cols] = hlm[bin_cols].astype("Int32")
        return hlm.persist()

    def compute_main_view(
        self,
        layer: Layer,
        hlm: DataFrameType,
        view_types: List[ViewType],
        partition_size: str = PARTITION_SIZE,
    ) -> DataFrameType:
        """Computes the main aggregated view from high-level metrics.

        This method takes the high-level metrics, sets derived columns,
        and then groups by the specified view_types to create a primary
        aggregated view of the I/O performance data.

        Args:
            hlm: A Dask DataFrame containing high-level metrics.
            view_types: A list of view types to group by for the main view.
            partition_size: The desired partition size for the resulting Dask DataFrame.

        Returns:
            A Dask DataFrame representing the main aggregated view.
        """
        return self.restore_view(
            name=self.get_checkpoint_name(CHECKPOINT_MAIN_VIEW, str(layer), *sorted(view_types)),
            fallback=lambda: self._compute_main_view(
                hlm=hlm,
                layer=layer,
                partition_size=partition_size,
                view_types=view_types,
            ),
        )

    def compute_views(
        self,
        layer: Layer,
        main_view: DataFrameType,
        view_types: List[ViewType],
    ) -> Views:
        """Computes multifaceted views for each specified metric.

        Iterates through all permutations of view_types for each metric,
        generating different "perspectives" on the data. Each perspective
        is a ViewResult, containing the filtered data and critical items.

        Args:
            main_view: The main aggregated Dask DataFrame.
            metrics: A list of metrics to compute views for.
            metric_boundaries: A dictionary of precomputed metric boundaries.
            view_types: A list of base view types to permute for creating views.

        Returns:
            A dictionary where keys are metrics and values are dictionaries
            mapping ViewKey to ViewResult.
        """
        views = {}
        for view_key in self.view_permutations(view_types=view_types):
            view_type = view_key[-1]
            parent_view_key = view_key[:-1]
            parent_records = main_view
            for parent_view_type in parent_view_key:
                parent_records = parent_records.query(
                    f"{parent_view_type} in @indices",
                    local_dict={"indices": views[(parent_view_type,)].index},
                )
            views[view_key] = self.compute_view(
                layer=layer,
                records=parent_records,
                view_key=view_key,
                view_type=view_type,
                view_types=view_types,
            )
        return views

    def compute_logical_views(
        self,
        layer: Layer,
        main_view: dd.DataFrame,
        views: Dict[ViewKey, dd.DataFrame],
        view_types: List[ViewType],
    ):
        """Computes views based on predefined logical relationships in the data.

        This method extends the existing view_results by adding new views
        derived from logical columns (e.g., file directory from file name).

        Args:
            main_view: The main aggregated Dask DataFrame.
            metric_boundaries: A dictionary of precomputed metric boundaries.
            metrics: A list of metrics to compute logical views for.
            view_results: The existing dictionary of computed views to be updated.
            view_types: A list of base view types available in the main_view.

        Returns:
            The updated view_results dictionary including the computed logical views.
        """
        logical_views = {}
        for parent_view_type in self.logical_views:
            parent_view_key = (parent_view_type,)
            if parent_view_key not in views:
                continue
            for view_type in self.logical_views[parent_view_type]:
                view_key = (parent_view_type, view_type)
                parent_records = main_view
                for parent_view_type in parent_view_key:
                    parent_records = parent_records.query(
                        f"{parent_view_type} in @indices",
                        local_dict={"indices": views[(parent_view_type,)].index},
                    )
                view_condition = self.logical_views[parent_view_type][view_type]
                if view_condition is None:
                    if view_type == "file_dir":
                        parent_records = parent_records.map_partitions(set_file_dir)
                    elif view_type == "file_pattern":
                        parent_records = parent_records.map_partitions(set_file_pattern)
                    else:
                        raise ValueError("XXX")
                else:
                    parent_records = parent_records.eval(f"{view_type} = {view_condition}")
                logical_views[view_key] = self.compute_view(
                    layer=layer,
                    records=parent_records,
                    view_key=view_key,
                    view_type=view_type,
                    view_types=view_types,
                )
        return logical_views

    def compute_time_boundaries(self, flat_views: Dict[ViewKey, pd.DataFrame]) -> ViewMetricBoundaries:
        """Computes time boundaries for each metric in the flat views.

        Args:
            flat_views: A dictionary of flat views keyed by their view types.

        Returns:
            A dictionary of time boundaries keyed by their view types.
        """
        time_boundaries = {}
        for view_key in flat_views:
            view_cols = flat_views[view_key].columns
            view_type = view_key[-1]
            time_layer = self.preset.time_boundary_layer
            time_metric = "time_sum" if self.is_view_process_based(view_key) else "time_proc_max"
            with log_block("calculate_time_boundary", view_key=view_key):
                if self.time_sliced and view_type == COL_TIME_RANGE:
                    time_boundary = self.time_granularity
                else:
                    time_boundary = flat_views[view_key][f"{time_layer}_{time_metric}"].sum()
                time_boundaries[view_type] = time_boundaries.get(view_type, {})
                for layer in self.preset.layer_defs:
                    layer_time_metrics = find_layer_time_metrics(list(view_cols), layer, time_metric)
                    for layer_time_metric in layer_time_metrics:
                        time_boundaries[view_type][layer_time_metric] = time_boundary
        return time_boundaries

    def compute_view(
        self,
        layer: Layer,
        view_key: ViewKey,
        view_type: str,
        view_types: List[ViewType],
        records: dd.DataFrame,
    ) -> dd.DataFrame:
        """Computes a single view based on the provided parameters.

        This involves restoring a view from a checkpoint or computing it.

        Args:
            metrics: The list of all metrics being analyzed.
            metric: The specific metric for this view.
            metric_boundary: The precomputed boundary for the current metric.
            records: The Dask DataFrame (parent records) to compute the view from.
            view_key: The key identifying this specific view.
            view_type: The primary dimension/column for this view.

        Returns:
            A ViewResult object containing the computed view, critical items,
            and filtered records.
        """
        return self.restore_view(
            name=self.get_checkpoint_name(CHECKPOINT_VIEW, str(layer), *list(view_key)),
            fallback=lambda: self._compute_view(
                layer=layer,
                records=records,
                view_key=view_key,
                view_type=view_type,
                view_types=view_types,
            ),
            read_from_disk=False,
            write_to_disk=CHECKPOINT_VIEWS,
        )

    def get_checkpoint_name(self, *args) -> str:
        """Generates a standardized name for a checkpoint.

        Joins the provided arguments with underscores. If HASH_CHECKPOINT_NAMES
        is True, it returns an MD5 hash of the name.

        Args:
            *args: String components to form the checkpoint name.

        Returns:
            A string representing the checkpoint name.
        """
        args = list(args) + [str(int(self.time_granularity))]
        checkpoint_name = "_".join(args)
        if HASH_CHECKPOINT_NAMES:
            return hashlib.md5(checkpoint_name.encode("utf-8")).hexdigest()
        return checkpoint_name

    def get_checkpoint_path(self, name: str) -> str:
        """Constructs the full path for a given checkpoint name.

        Args:
            name: The name of the checkpoint.

        Returns:
            The absolute path to the checkpoint directory/file.
        """
        return f"{self.checkpoint_dir}/{name}"

    def get_hlm_checkpoint_name(self, view_types: List[ViewType]) -> str:
        return self.get_checkpoint_name(CHECKPOINT_HLM, *sorted(view_types))

    def get_profile_hlm_checkpoint_name(self, view_types: List[ViewType]) -> str:
        return self.get_checkpoint_name(CHECKPOINT_HLM, "profile", *sorted(view_types))

    def get_trace_hlm_checkpoint_name(self, view_types: List[ViewType]) -> str:
        return self.get_checkpoint_name(CHECKPOINT_HLM, "trace", *sorted(view_types))

    def get_job_time(self, traces: dd.DataFrame) -> float:
        """Computes the total job execution time from the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data,
                    expected to have 'tstart' and 'tend' columns.

        Returns:
            The total job time as a float.
        """
        return traces[COL_TIME_END].max() - traces[COL_TIME_START].min()

    def ensure_proc_view_type(self, view_types: List[ViewType]) -> List[ViewType]:
        """Ensures that COL_PROC_NAME is always included in the list of view types.

        Args:
            view_types: A list of view types to be used for analysis.

        Returns:
            A sorted list of view types that always includes COL_PROC_NAME.
        """
        return list(sorted(set(view_types).union({COL_PROC_NAME})))

    def get_stats_checkpoint_name(self):
        return self.get_checkpoint_name(CHECKPOINT_RAW_STATS)

    def get_total_event_count(self, traces: dd.DataFrame) -> int:
        """Computes the total number of I/O events in the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            The total count of I/O events as an integer.
        """
        return traces.index.count().persist()

    def get_profile_event_count(self, profiles: Optional[dd.DataFrame]):
        if profiles is None:
            return 0
        if COL_COUNT in profiles.columns:
            return profiles[COL_COUNT].fillna(0).sum().persist()
        return profiles.index.count().persist()

    def get_unique_host_count(self, traces: dd.DataFrame):
        """Computes the total number of unique hosts accessed in the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            The total count of unique hosts accessed as an integer.
        """
        return traces[COL_HOST_NAME].nunique()

    def get_unique_file_count(self, traces: dd.DataFrame):
        """Computes the total number of unique files accessed in the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            The total count of unique files accessed as an integer.
        """
        return traces[COL_FILE_NAME].nunique()

    def get_unique_process_count(self, traces: dd.DataFrame):
        """Computes the total number of unique processes accessed in the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            The total count of unique processes accessed as an integer.
        """
        return traces[COL_PROC_NAME].nunique()

    def has_checkpoint(self, name: str):
        """Checks if a checkpoint with the given name exists.

        A checkpoint is considered to exist if its `_metadata` file is present.

        Args:
            name: The name of the checkpoint.

        Returns:
            True if the checkpoint exists, False otherwise.
        """
        checkpoint_path = self.get_checkpoint_path(name=name)
        return os.path.exists(f"{checkpoint_path}/_metadata")

    def is_logical_view_of(self, view_key: ViewKey, parent_view_type: ViewType) -> bool:
        if len(view_key) == 2:
            return view_key[1] in self.logical_views[parent_view_type]
        return False

    def is_view_process_based(self, view_key: ViewKey) -> bool:
        view_type = view_key[-1]
        is_proc_view = view_type == COL_PROC_NAME
        is_logical_proc_view = self.is_logical_view_of(view_key, COL_PROC_NAME)
        return is_proc_view or is_logical_proc_view

    def restore_extra_data(self, name: str, fallback: Callable[[], dict], force=False, persist=False) -> dict:
        """Restores extra (non-DataFrame) data from a JSON checkpoint.

        If checkpointing is enabled and the checkpoint file exists (unless 'force'
        is True), it loads the data from the JSON file. Otherwise, it calls the
        'fallback' function to compute the data and then stores it asynchronously.

        Args:
            name: The name of the checkpoint.
            fallback: A callable function that returns the data if not found or forced.
            force: If True, forces recomputation even if a checkpoint exists.
            persist: (Currently unused in the method body, but part of signature)

        Returns:
            A dictionary containing the restored or computed data.
        """
        if self.checkpoint:
            data_path = f"{self.get_checkpoint_path(name=name)}.json"
            if force or not os.path.exists(data_path):
                data = fallback()
                fire_and_forget(
                    self.dask_client.submit(
                        self.store_extra_data,
                        data=self.dask_client.submit(dask.compute, data),
                        data_path=data_path,
                    )
                )
                return data
            with open(data_path, "r") as f:
                return json.load(f)
        return fallback()

    def restore_flat_views(self, view_keys: List[ViewKey]) -> Dict[ViewKey, pd.DataFrame]:
        restored_flat_views = {}
        for view_key in view_keys:
            flat_view_checkpoint_name = self.get_checkpoint_name(CHECKPOINT_FLAT_VIEW, *list(view_key))
            flat_view_checkpoint_path = self.get_checkpoint_path(name=flat_view_checkpoint_name)
            if self.has_checkpoint(name=flat_view_checkpoint_name):
                restored_flat_views[view_key] = pd.read_parquet(f"{flat_view_checkpoint_path}.parquet")
        return restored_flat_views

    def restore_view(
        self,
        name: str,
        fallback: Callable[[], dd.DataFrame],
        force=False,
        write_to_disk=True,
        read_from_disk=False,
    ) -> dd.DataFrame:
        """Restores a Dask DataFrame view from a Parquet checkpoint.

        If checkpointing is enabled and the checkpoint exists (unless 'force' is True),
        it reads the DataFrame from the Parquet store. Otherwise, it calls the
        'fallback' function to compute the DataFrame. If 'write_to_disk' is True,
        the computed DataFrame is then stored as a checkpoint.

        Args:
            name: The name of the checkpoint.
            fallback: A callable function that returns the DataFrame if not found or forced.
            force: If True, forces recomputation even if a checkpoint exists.
            write_to_disk: If True, saves the computed view to disk if it was recomputed.

        Returns:
            A Dask DataFrame representing the restored or computed view.
        """
        if self.checkpoint:
            view_path = self.get_checkpoint_path(name=name)
            if force or not self.has_checkpoint(name=name):
                with log_block("restore_view_fallback_build", name=name):
                    view = fallback()
                if not write_to_disk:
                    return view
                with log_block("restore_view_schedule_store_view", name=name):
                    checkpoint_task = self.dask_client.compute(self.store_view(name=name, view=view), sync=False)
                    self.checkpoint_tasks.append(checkpoint_task)
                if not read_from_disk:
                    return view
                self.dask_client.cancel(checkpoint_task)
            with log_block("restore_view_read_parquet_metadata", name=name):
                return dd.read_parquet(view_path)
        with log_block("restore_view_fallback_build_no_ckpt", name=name):
            return fallback()

    @staticmethod
    def set_layer_metrics(
        hlm: pd.DataFrame,
        derived_metrics: Dict[str, str],
        size_derived_metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        # Create an explicit copy to avoid SettingWithCopyWarning
        hlm = hlm.copy()
        hlm_columns = list(hlm.columns)
        size_derived_metric_set = set(size_derived_metrics or [])
        is_size_col = {col: (col == "size" or "size_bin" in col) for col in hlm_columns}
        col_kinds = {}
        numeric_cols = {}

        for col in hlm_columns:
            series = hlm[col]
            if is_size_col[col] or pd.api.types.is_numeric_dtype(series.dtype):
                col_kinds[col] = "numeric"
                numeric_cols[col] = to_nullable_numeric(series)
            elif pd.api.types.is_string_dtype(series.dtype):
                col_kinds[col] = "string"
            elif is_set_like_series(series):
                col_kinds[col] = "set_like"
            else:
                raise TypeError(
                    f"Unsupported data type '{series.dtype}' for column '{col}'. "
                    "Developer must add explicit handling for this data type in set_layer_metrics."
                )

        # Build derived columns in-memory and append once to avoid repeated fragmentation.
        derived_cols: Dict[str, pd.Series] = {}
        for metric, condition in derived_metrics.items():
            metric_mask = hlm.eval(condition)
            is_size_metric = metric in size_derived_metric_set
            for col in hlm_columns:
                if not is_size_metric and is_size_col[col]:
                    continue
                metric_col = f"{metric}_{col}"
                if col_kinds[col] in {"string", "set_like"}:
                    # unique_set_flatten skips None for set-like columns downstream.
                    derived_cols[metric_col] = hlm[col].where(metric_mask, None)
                else:
                    derived_cols[metric_col] = numeric_cols[col].where(metric_mask)

        if derived_cols:
            hlm = pd.concat([hlm, pd.DataFrame(derived_cols, index=hlm.index)], axis=1)
        return hlm

    @staticmethod
    def store_extra_data(data: Tuple[Dict], data_path: str):
        """Saves extra (non-DataFrame) data to a JSON file.

        This static method is typically used by Dask workers to persist data.

        Args:
            data: A tuple containing a single dictionary of data to be saved.
            data_path: The full path to the JSON file where data will be stored.
        """
        with open(data_path, "w") as f:
            return json.dump(data[0], f, cls=NpEncoder)

    def store_flat_views(self, flat_views: Dict[ViewKey, pd.DataFrame]):
        for view_key in flat_views:
            flat_view_checkpoint_name = self.get_checkpoint_name(CHECKPOINT_FLAT_VIEW, *list(view_key))
            flat_view_checkpoint_path = self.get_checkpoint_path(name=flat_view_checkpoint_name)
            if self.has_checkpoint(name=flat_view_checkpoint_name):
                continue
            # Save local pandas flat views directly to avoid shipping large payloads
            # through Dask task graphs.
            self._save_flat_view(
                view=flat_views[view_key],
                view_path=flat_view_checkpoint_path,
            )
        return []

    def store_view(self, name: str, view: dd.DataFrame, partition_size="64MB"):
        """Stores a Dask DataFrame view to a Parquet checkpoint.

        The view DataFrame is repartitioned and then written to a subdirectory
        named `name` within the `checkpoint_dir`.

        Args:
            name: The name of the checkpoint.
            view: The Dask DataFrame to store.
            compute: Whether to compute the DataFrame before writing (Dask default is True).
            partition_size: The desired partition size for the output Parquet files.

        Returns:
            The result of the Dask `to_parquet` operation.
        """
        for col in view.columns:
            if view.dtypes[col].name == "object":
                view[col] = view[col].astype(str)
        if view.npartitions > 1:
            view = view.repartition(partition_size=partition_size)
        return view.to_parquet(
            self.get_checkpoint_path(name=name),
            compute=False,
            write_metadata_file=True,
        )

    def validate_time_granularity(self, hlm: dd.DataFrame, view_types: List[ViewType]):
        if "io_time" in hlm.columns:
            max_io_time = hlm.groupby(view_types)["io_time"].sum().max().compute()
            if max_io_time > self.time_granularity:
                raise ValueError(
                    f"The max 'io_time' exceeds the 'time_granularity' '{self.time_granularity}'. "
                    f"Please adjust the 'time_granularity' to '{int(2 * max_io_time)}' and rerun the analyzer."
                )

    @staticmethod
    def view_permutations(view_types: List[ViewType]):
        """Generates all permutations of view_types for creating multifaceted views.

        For a list of view_types [vt1, vt2, vt3], it will generate permutations
        of length 1, 2, and 3, e.g., (vt1,), (vt2,), (vt1, vt2), (vt2, vt1), ...

        Args:
            view_types: A list of ViewType elements.

        Returns:
            An iterator yielding tuples, where each tuple is a permutation of view_types.
        """

        if not VIEW_PERMUTATIONS:
            return it.permutations(view_types, 1)

        def _iter_permutations(r: int):
            return it.permutations(view_types, r + 1)

        return it.chain.from_iterable(map(_iter_permutations, range(len(view_types))))

    def _analyze_hlm(
        self,
        hlm: Optional[DataFrameType],
        proc_view_types: List[ViewType],
        metric_boundaries: ViewMetricBoundaries,
        raw_stats: RawStats,
        logical_view_types: bool,
        trace_hlm: Optional[DataFrameType] = None,
        profile_hlm: Optional[DataFrameType] = None,
        system_metrics: Optional[pd.DataFrame] = None,
        layer_main_views: Optional[Dict[Layer, DataFrameType]] = None,
        is_dask: bool = True,
        source_traces: Optional[pd.DataFrame] = None,
        source_control_pids: Optional[List[int]] = None,
    ) -> AnalysisResult:
        # Compute layers & views
        with console_block("Compute views"):
            with log_block("create_layers_and_views_tasks"):
                hlms = {}
                main_views = {}
                main_indexes = {}
                views = {}
                view_keys = set()
                for layer, layer_condition in self.preset.layer_defs.items():
                    layer_hlm = None
                    if layer_main_views is not None and layer in layer_main_views:
                        layer_main_view = layer_main_views[layer]
                    else:
                        if hlm is not None:
                            layer_hlm = hlm.copy()
                            if layer_condition:
                                layer_hlm = hlm.query(layer_condition)
                        else:
                            if trace_hlm is None and profile_hlm is None:
                                raise ValueError(
                                    "Either hlm or at least one of trace_hlm/profile_hlm must be provided "
                                    "when layer_main_views is not supplied"
                                )
                            layer_trace_hlm = trace_hlm
                            layer_profile_hlm = profile_hlm
                            if layer_condition:
                                if layer_trace_hlm is not None:
                                    layer_trace_hlm = layer_trace_hlm.query(layer_condition)
                                if layer_profile_hlm is not None:
                                    layer_profile_hlm = layer_profile_hlm.query(layer_condition)
                            if layer_trace_hlm is None:
                                layer_hlm = layer_profile_hlm
                            elif layer_profile_hlm is None:
                                layer_hlm = layer_trace_hlm
                            else:
                                layer_hlm = self.reconcile_hlm(
                                    layer=layer,
                                    profile_hlm=layer_profile_hlm,
                                    trace_hlm=layer_trace_hlm,
                                    view_types=proc_view_types,
                                )
                        layer_main_view = self.compute_main_view(
                            layer=layer,
                            hlm=layer_hlm,
                            view_types=proc_view_types,
                        )
                    layer_main_index = layer_main_view.index.to_frame().reset_index(drop=True)
                    layer_views = self.compute_views(
                        layer=layer,
                        main_view=layer_main_view,
                        view_types=proc_view_types,
                    )
                    if logical_view_types:
                        layer_logical_views = self.compute_logical_views(
                            layer=layer,
                            main_view=layer_main_view,
                            views=layer_views,
                            view_types=proc_view_types,
                        )
                        layer_views.update(layer_logical_views)
                    hlms[layer] = layer_hlm
                    main_views[layer] = layer_main_view
                    main_indexes[layer] = layer_main_index
                    views[layer] = layer_views
                    view_keys.update(layer_views.keys())

        if is_dask:
            with log_block("compute_views_and_raw_stats"):
                (views, raw_stats) = dask.compute(views, raw_stats)
            # ASSERT HERE, not at read_stats. This is the first point where raw_stats hold concrete
            # ints; at construction time they are lazy dask Scalars and the guard silently skipped
            # every one of them, so it never fired on the very corpus that motivated it.
            _assert_metadata_resolved(_raw_stats_as_dict(raw_stats))

        # Restore checkpointed flat views if available
        checkpointed_flat_views = {}
        if self.checkpoint:
            with log_block("restore_flat_view_checkpoints"):
                checkpointed_flat_views.update(self.restore_flat_views(view_keys=list(view_keys)))

        # Process views to create flat views
        with console_block("Process views"):
            flat_views = {}
            for layer in views:
                for view_key in views[layer]:
                    if view_key in checkpointed_flat_views:
                        flat_views[view_key] = checkpointed_flat_views[view_key]
                        continue
                    with log_block("merge_flat_view", view_key=view_key):
                        view = views[layer][view_key].copy()
                        view.columns = view.columns.map(lambda col: layer.lower() + "_" + col)
                        if view_key in flat_views:
                            flat_views[view_key] = flat_views[view_key].merge(
                                view,
                                how="outer",
                                left_index=True,
                                right_index=True,
                            )
                        else:
                            flat_views[view_key] = view
                    try:
                        df = flat_views[view_key]
                        mem_bytes = int(df.memory_usage(deep=True).sum()) if hasattr(df, "memory_usage") else -1
                        logger.debug(
                            "Flat view created",
                            view_key=view_key,
                            shape=getattr(df, "shape", None),
                            mem_bytes=mem_bytes,
                        )
                    except Exception:
                        pass

            # Compute time boundaries for flat views
            with log_block("compute_time_boundaries"):
                metric_boundaries.update(self.compute_time_boundaries(flat_views))
                if self.debug:
                    debug_dir = getattr(self, "debug_output_dir", os.getcwd())
                    os.makedirs(debug_dir, exist_ok=True)
                    with open(os.path.join(
                        debug_dir, f"metric_boundaries{self._debug_artifact_suffix()}.json"
                    ), "w") as f:
                        json.dump(
                            metric_boundaries,
                            f,
                            cls=NpEncoder,
                            indent=4,
                        )

            # Process flat views
            with log_block("process_flat_views"):
                for view_key in flat_views:
                    # Process flat views to compute metrics and scores
                    flat_views[view_key] = self._process_flat_view(
                        flat_view=flat_views[view_key],
                        view_key=view_key,
                        metric_boundaries=metric_boundaries,
                        system_metrics=system_metrics,
                    )
                    if self.debug:
                        self._write_debug_flat_view(
                            flat_view=flat_views[view_key],
                            view_key=view_key,
                        )

            # HLM-only callers have no original event frame.  Preserve missing
            # integrity fields in that case rather than fabricating metadata;
            # trace-backed streaming callers pass their concrete input below.
            flat_views = attach_window_integrity_metrics(
                flat_views,
                source_traces,
                control_pids=source_control_pids,
            )
            flat_views = attach_input_step_max_sum_metrics(flat_views, source_traces)
            flat_views = attach_input_job_step_max_sum_metrics(flat_views, source_traces)
            flat_views = attach_comm_exposed_median_metrics(flat_views, source_traces)
            flat_views = attach_comm_total_median_metrics(flat_views, source_traces)
            if self.debug and ("window",) in flat_views:
                self._write_debug_flat_view(flat_view=flat_views[("window",)], view_key=("window",))

        # Checkpoint flat views if enabled
        if self.checkpoint:
            with log_block("write_flat_view_checkpoints"):
                self.checkpoint_tasks.extend(self.store_flat_views(flat_views=flat_views))

        # Wait for all checkpoint tasks
        if self.checkpoint:
            with log_block("wait_for_checkpoints"):
                wait(self.checkpoint_tasks)

        # Debug-log full flat view data and key HLM epoch metrics
        for vk, vdf in flat_views.items():
            logger.debug(
                "flat_view.publish",
                view_key=list(vk),
                columns=list(vdf.columns),
                num_rows=len(vdf),
                data=vdf.reset_index().to_dict(orient="records"),
            )
            # Log key epoch-level HLM metrics when available
            if vk[-1] == "epoch":
                hlm_sample = {}
                for col_name in ("compute_time_proc_max", "fetch_iter_time_proc_max", "reader_posix_time_proc_max"):
                    for c in vdf.columns:
                        if c.endswith(col_name):
                            try:
                                hlm_sample[c] = float(vdf[c].iloc[0]) if len(vdf) > 0 else None
                            except Exception:
                                pass
                if hlm_sample:
                    logger.debug("hlm.epoch_metrics", **hlm_sample)

        with log_block("evaluate_fact_rules"):
            analysis_facts = self._evaluate_analysis_facts(
                flat_views=flat_views,
                raw_stats=raw_stats,
            )

        # Debug-log all analysis facts before publishing
        if analysis_facts:
            all_facts_list = []
            for vk, facts_list in analysis_facts.items():
                for f in facts_list:
                    fact_dict = dc.asdict(f) if dc.is_dataclass(f) else f
                    all_facts_list.append(fact_dict)
            logger.debug(
                "analysis_facts.publish",
                fact_count=len(all_facts_list),
                facts=all_facts_list,
            )

        output_flat_views, output_analysis_facts = self._materialize_output_artifacts(
            flat_views=flat_views,
            analysis_facts=analysis_facts,
        )

        result = AnalysisResult(
            _hlms=hlms,
            _main_views=main_views,
            _metric_boundaries=metric_boundaries,
            additional_metrics={
                view_type: list(metrics.keys())
                for view_type, metrics in (self.preset.additional_metrics or {}).items()
            },  # type: ignore
            analysis_facts=output_analysis_facts,
            checkpoint_dir=self.checkpoint_dir,
            flat_views=output_flat_views,
            layers=self.layers,
            raw_stats=raw_stats,
            view_types=proc_view_types,
            views=views,
        )
        self._write_debug_fact_envelope(result)
        return result

    # ------------------------------------------------------------------ live fast path
    def _fastpath_mode(self) -> str:
        """`auto` (default), `1`/`on` to force, `0`/`off` to disable.

        `auto` uses the fast path only when the loaded rule set is one it can serve exactly
        (see fastpath.FastPathPlan.build); `1` makes an unservable rule set an error rather than
        a silent fall back to the 6-second path, which is what a scale run wants.
        """
        value = os.environ.get("LIVEFLOW_ANALYZER_FASTPATH", "auto").strip().lower()
        if value in ("0", "off", "false", "no"):
            return "off"
        if value in ("1", "on", "true", "yes", "force"):
            return "force"
        return "auto"

    def _get_fastpath_plan(self, proc_view_types: List[ViewType]):
        """Resolve the plan once per analyzer, not once per window."""
        mode = self._fastpath_mode()
        if mode == "off":
            return None
        cached = getattr(self, "_fastpath_plan_cache", None)
        key = tuple(proc_view_types)
        if cached is not None and cached[0] == key:
            return cached[1]
        from .fastpath import FastPathPlan

        plan = FastPathPlan.build(
            preset=self.preset,
            view_types=list(proc_view_types),
            layers=self.layers,
            fact_engine=self.fact_engine,
        )
        if plan is None and mode == "force":
            raise RuntimeError(
                "LIVEFLOW_ANALYZER_FASTPATH=1 but the loaded rule set needs metrics the fast "
                "path does not produce; see the analyzer.fastpath.declined log line."
            )
        self._fastpath_plan_cache = (key, plan)
        return plan

    def _analyze_trace_fast(
        self,
        traces: DataFrameType,
        proc_view_types: List[ViewType],
        metric_boundaries: ViewMetricBoundaries,
        raw_stats: RawStats,
    ) -> Optional[AnalysisResult]:
        """Per-window analysis restricted to what the *-signal rule sets read.

        Returns None when the fast path cannot serve the loaded rules, so the caller runs the
        full path.  Never returns a partial answer.
        """
        plan = self._get_fastpath_plan(proc_view_types)
        if plan is None:
            return None
        flat_views = plan.flat_views(
            traces,
            control_pids=(getattr(self, "_debug_window_provenance", None) or {}).get("rank_pids", []),
        )
        if not flat_views:
            return None
        # The fast path is an analysis optimization, not a provenance opt-out.
        # The normal path writes these per-completed-window CSVs while processing
        # flat views.  Keep the same artifacts here so a signal-rule smoke can
        # pass the established SHM validator and remain auditable after the
        # node agent composes the diagnoser in the same process.
        if self.debug:
            for view_key, flat_view in flat_views.items():
                self._write_debug_flat_view(flat_view=flat_view, view_key=view_key)
        analysis_facts = self._evaluate_analysis_facts(flat_views=flat_views, raw_stats=raw_stats)
        output_flat_views, output_analysis_facts = self._materialize_output_artifacts(
            flat_views=flat_views,
            analysis_facts=analysis_facts,
        )
        result = AnalysisResult(
            _hlms={},
            _main_views={},
            _metric_boundaries=metric_boundaries,
            additional_metrics={},
            analysis_facts=output_analysis_facts,
            checkpoint_dir=self.checkpoint_dir,
            flat_views=output_flat_views,
            layers=self.layers,
            raw_stats=raw_stats,
            view_types=list(proc_view_types),
            views={},
        )
        self._write_debug_fact_envelope(result)
        return result

    def _analyze_trace(
        self,
        traces: DataFrameType,
        proc_view_types: List[ViewType],
        logical_view_types: bool,
        raw_stats: RawStats,
        metric_boundaries: ViewMetricBoundaries,
        profiles: Optional[DataFrameType] = None,
        system_metrics: Optional[pd.DataFrame] = None,
    ):
        is_dask = isinstance(traces, dd.DataFrame)
        hlm_checkpoint_name = self.get_hlm_checkpoint_name(view_types=proc_view_types)

        # Compute high-level metrics
        with console_block("Compute high-level metrics"):
            with log_block("compute_high_level_metrics"):
                hlm = self.compute_high_level_metrics(
                    checkpoint_name=hlm_checkpoint_name,
                    traces=traces,
                    view_types=proc_view_types,
                )
            profile_hlm = None
            if profiles is not None and not profiles.empty:
                with log_block("compute_profile_hlm"):
                    profile_hlm = self.compute_profile_hlm(
                        profiles=profiles,
                        view_types=proc_view_types,
                    )

            if is_dask:
                with log_block("persist"):
                    (hlm, raw_stats) = dask.persist(hlm, raw_stats)
                with log_block("wait"):
                    wait([hlm, raw_stats])

        # Analyze HLM — when profiles exist, pass hlm as trace_hlm so
        # _analyze_hlm enters the reconcile path (matching the batch path).
        if profile_hlm is not None:
            result = self._analyze_hlm(
                hlm=None,
                trace_hlm=hlm,
                is_dask=is_dask,
                logical_view_types=logical_view_types,
                metric_boundaries=metric_boundaries,
                proc_view_types=proc_view_types,
                raw_stats=raw_stats,
                profile_hlm=profile_hlm,
                system_metrics=system_metrics,
                source_traces=traces if isinstance(traces, pd.DataFrame) else None,
                source_control_pids=(getattr(self, "_debug_window_provenance", None) or {}).get("rank_pids", []),
            )
        else:
            result = self._analyze_hlm(
                hlm=hlm,
                is_dask=is_dask,
                logical_view_types=logical_view_types,
                metric_boundaries=metric_boundaries,
                proc_view_types=proc_view_types,
                raw_stats=raw_stats,
                system_metrics=system_metrics,
                source_traces=traces if isinstance(traces, pd.DataFrame) else None,
                source_control_pids=(getattr(self, "_debug_window_provenance", None) or {}).get("rank_pids", []),
            )

        # Attach correct traces & view types
        result._read_result = ReadTraceResult(traces=traces)
        result.view_types = proc_view_types
        result._pre_layer_hlm = hlm.compute() if isinstance(hlm, dd.DataFrame) else hlm

        return result

    def _compute_high_level_metrics(
        self,
        traces: DataFrameType,
        view_types: list,
        partition_size: str,
    ) -> DataFrameType:
        # Add layer columns
        hlm_groupby = list(dict.fromkeys(view_types + list(self.preset.hlm_fields)))
        # Build agg_dict
        bin_cols = [col for col in traces.columns if "_bin_" in col]
        view_types_diff = list(set(VIEW_TYPES).difference(view_types))

        # Pre-compute alias columns for per-call statistics
        traces = traces.assign(
            time_sq=traces["time"] ** 2,
            size_sq=traces["size"] ** 2,
            time_call_min=traces["time"],
            time_call_max=traces["time"],
            size_call_min=traces["size"],
            size_call_max=traces["size"],
        )
        hlm_agg = dict(HLM_AGG)
        hlm_agg.update({col: "sum" for col in bin_cols})
        # Per-call support columns
        hlm_agg["time_sq"] = "sum"
        hlm_agg["size_sq"] = "sum"
        hlm_agg["time_call_min"] = "min"
        hlm_agg["time_call_max"] = "max"
        hlm_agg["size_call_min"] = "min"
        hlm_agg["size_call_max"] = "max"

        if isinstance(traces, dd.DataFrame):
            hlm_agg.update({col: unique_set() for col in view_types_diff})
            hlm = (
                traces.groupby(hlm_groupby)
                .agg(hlm_agg, split_out=math.ceil(math.sqrt(traces.npartitions)))
                .persist()
                .repartition(partition_size=partition_size)
                .map_partitions(_replace_zeros_preserving_stall_time, preserve=self.preset.name == "liveflow-v1", by_function=True)
            )
        else:
            hlm_agg.update({col: unique_set_pd for col in view_types_diff})
            hlm = traces.groupby(hlm_groupby).agg(hlm_agg)
            hlm = _replace_zeros_preserving_stall_time(hlm, preserve=self.preset.name == "liveflow-v1", by_function=True)
            hlm = fix_hlm_dtypes(hlm)

        hlm[bin_cols] = hlm[bin_cols].astype("Int32")

        return hlm

    def _compute_profile_hlm(
        self,
        profiles: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str,
    ) -> dd.DataFrame:
        profiles = profiles.map_partitions(set_size_bins)
        return self._compute_high_level_metrics(
            partition_size=partition_size,
            traces=profiles,
            view_types=view_types,
        )

    def _validate_and_expand_profiles(
        self,
        profiles: dd.DataFrame,
        profile_time_granularity: float,
    ) -> dd.DataFrame:
        profile_ratio = self.time_granularity / profile_time_granularity
        inverse_ratio = profile_time_granularity / self.time_granularity
        is_coarser_aligned = math.isclose(profile_ratio, round(profile_ratio), rel_tol=0, abs_tol=1e-9)
        is_finer_aligned = math.isclose(inverse_ratio, round(inverse_ratio), rel_tol=0, abs_tol=1e-9)
        if not is_coarser_aligned and not is_finer_aligned:
            raise ValueError(
                f"Analysis granularity ({self.time_granularity}s) must evenly divide into "
                f"or be an integer multiple of the profile bucket width "
                f"({profile_time_granularity}s)"
            )
        if self.time_granularity < profile_time_granularity:
            expansion_factor = int(round(inverse_ratio))
            profiles = profiles.map_partitions(
                self._expand_profile_buckets,
                expansion_factor=expansion_factor,
                time_resolution=self.time_resolution,
                time_granularity=self.time_granularity,
                distribution=self.profile_distribution,
                meta=profiles._meta,
            )
        return profiles

    @staticmethod
    def _expand_profile_buckets(
        df: pd.DataFrame,
        expansion_factor: int,
        time_resolution: float,
        time_granularity: float,
        distribution: str,
    ) -> pd.DataFrame:
        """Expand each profile row into ``expansion_factor`` sub-bucket rows.

        When the analysis granularity is finer than the profile bucket width
        (e.g. 1s analysis vs 5s profiles, expansion_factor=5), each canonical
        profile row is replicated into *expansion_factor* rows with adjusted
        ``time_range``, ``time_start``, and ``time_end``.

        Measures (``count``, ``time``, ``size``) are distributed across
        sub-buckets using the chosen *distribution* strategy:

        * ``"uniform"`` — divide evenly.
        * ``"weighted"`` — use ``time_min`` / ``time_max`` to shape the
          distribution so that sub-buckets with higher estimated per-event
          duration receive proportionally more of the aggregate ``time``.
          ``count`` and ``size`` remain uniformly split.
        """
        if df.empty:
            return df

        sub_granularity_us = int(time_granularity * time_resolution)
        output_columns = list(df.columns)

        # Repeat every row expansion_factor times
        expanded = df.loc[df.index.repeat(expansion_factor)].copy()
        sub_idx = np.tile(np.arange(expansion_factor), len(df))

        # Recompute time_start / time_end / time_range for each sub-bucket
        base_start = np.repeat(df["time_start"].values, expansion_factor)
        expanded["time_start"] = pd.array(
            (base_start + sub_idx * sub_granularity_us).astype(np.int64), dtype="Int64"
        )
        expanded["time_end"] = pd.array(
            (expanded["time_start"].to_numpy(dtype=np.int64, na_value=0) + sub_granularity_us),
            dtype="Int64",
        )
        expanded["time_range"] = pd.array(
            expanded["time_start"].to_numpy(dtype=np.int64, na_value=0) // sub_granularity_us,
            dtype="Int64",
        )

        # Distribute measures
        if distribution == "weighted":
            t_min = np.repeat(df["time_min"].fillna(0).values, expansion_factor)
            t_max = np.repeat(df["time_max"].fillna(0).values, expansion_factor)
            if expansion_factor > 1:
                ramp = t_min + (t_max - t_min) * sub_idx / (expansion_factor - 1)
            else:
                ramp = t_min
            ramp_sums = np.repeat(
                ramp.reshape(-1, expansion_factor).sum(axis=1),
                expansion_factor,
            )
            weights = np.where(ramp_sums > 0, ramp / ramp_sums, 1.0 / expansion_factor)
            total_time = np.repeat(df["time"].values, expansion_factor)
            expanded["time"] = (total_time * weights).astype("float64")
        else:
            expanded["time"] = (
                np.repeat(df["time"].values, expansion_factor) / expansion_factor
            ).astype("float64")

        # count and size always split uniformly (they are discrete totals)
        expanded["count"] = (
            np.repeat(df["count"].fillna(0).values, expansion_factor) / expansion_factor
        )
        base_count = expanded["count"].values
        floor_count = np.floor(base_count).astype(np.int64)
        remainders = np.repeat(
            (df["count"].fillna(0).values % expansion_factor).astype(np.int64),
            expansion_factor,
        )
        floor_count = np.where(sub_idx < remainders, floor_count + 1, floor_count)
        expanded["count"] = pd.array(floor_count, dtype="Int64")

        orig_size = df["size"].values
        has_size = pd.notna(orig_size)
        rep_size = np.repeat(
            np.where(has_size, orig_size.astype(float), 0.0), expansion_factor
        )
        rep_has_size = np.repeat(has_size, expansion_factor)
        size_vals = np.where(rep_has_size, (rep_size / expansion_factor).astype(np.int64), 0)
        size_mask = ~rep_has_size
        expanded["size"] = pd.array(size_vals, dtype="Int64")
        expanded.loc[size_mask, "size"] = pd.NA

        # stat columns carry through unchanged (they are per-event bounds)
        expanded.reset_index(drop=True, inplace=True)
        return expanded[output_columns]

    def _compute_main_view(
        self,
        layer: Layer,
        hlm: DataFrameType,
        view_types: List[ViewType],
        partition_size: str,
    ) -> DataFrameType:
        is_dask = isinstance(hlm, dd.DataFrame)
        with log_block("drop_and_set_metrics", layer=layer):
            size_layers = {configured_layer.lower() for configured_layer in (self.preset.size_layers or [])}
            keep_size_for_layer = layer.lower() in size_layers
            if not keep_size_for_layer:
                size_cols = [col for col in hlm.columns if col.startswith("size")]
                hlm = hlm.drop(columns=size_cols)  # type: ignore
                if "file_name" in hlm.columns:
                    hlm = hlm.drop(columns=["file_name"])  # type: ignore
            layer_derived_metrics = self.preset.derived_metrics.get(layer.lower(), {})
            layer_size_derived_metrics = (self.preset.size_derived_metrics or {}).get(layer.lower(), [])
            if is_dask:
                hlm = hlm.map_partitions(
                    self.set_layer_metrics,
                    derived_metrics=layer_derived_metrics,
                    size_derived_metrics=layer_size_derived_metrics,
                )
            else:
                hlm = self.set_layer_metrics(
                    hlm=hlm,
                    derived_metrics=layer_derived_metrics,
                    size_derived_metrics=layer_size_derived_metrics,
                )

        with log_block("build_agg_dict", layer=layer):
            # Build agg dict
            view_types_diff = set(VIEW_TYPES).difference(view_types)
            main_view_agg = {}
            for col in hlm.columns:
                if any(map(col.endswith, view_types_diff)):
                    if is_dask:
                        main_view_agg[col] = unique_set_flatten()
                    else:
                        main_view_agg[col] = unique_set_flatten_pd
                elif col not in self.preset.hlm_fields:
                    if col.endswith("_call_min"):
                        main_view_agg[col] = "min"
                    elif col.endswith("_call_max"):
                        main_view_agg[col] = "max"
                    else:
                        main_view_agg[col] = "sum"

        with log_block("compute_main_view", layer=layer):
            if is_dask:
                main_view = (
                    hlm.groupby(list(view_types))
                    .agg(main_view_agg, split_out=hlm.npartitions)
                    .map_partitions(set_main_metrics)
                    .map_partitions(_replace_zeros_preserving_stall_time,
                                    preserve=self.preset.name == "liveflow-v1" and layer in ("stall_storage", "stall_decode"))
                    .map_partitions(fix_dtypes)
                    .persist()
                )
            else:
                main_view = hlm.groupby(list(view_types)).agg(main_view_agg)
                main_view = _replace_zeros_preserving_stall_time(set_main_metrics(main_view),
                    preserve=self.preset.name == "liveflow-v1" and layer in ("stall_storage", "stall_decode"))
                main_view = fix_dtypes(main_view)

        return main_view

    def _compute_view(
        self,
        layer: Layer,
        records: DataFrameType,
        view_key: ViewKey,
        view_type: str,
        view_types: List[ViewType],
    ) -> DataFrameType:
        is_dask = isinstance(records, dd.DataFrame)
        is_view_process_based = self.is_view_process_based(view_key)

        view_types_diff = set(VIEW_TYPES).difference(view_types)

        if is_dask:
            local_view_types = records.index._meta.names
        else:
            local_view_types = records.index.names

        local_view_types_diff = set(local_view_types).difference([view_type])

        with log_block("build_agg_dict", layer=layer, view_key=view_key):
            view_agg = {}
            for col in records.columns:
                if "_bin_" in col:
                    view_agg[col] = ["sum"]
                elif any(map(col.endswith, view_types_diff)):
                    if is_dask:
                        view_agg[col] = [unique_set_flatten()]
                    else:
                        view_agg[col] = [unique_set_flatten_pd]
                elif col in it.chain.from_iterable(self.logical_views.values()):
                    if is_dask:
                        view_agg[col] = [unique_set_flatten()]
                    else:
                        view_agg[col] = [unique_set_flatten_pd]
                elif col.endswith("_sq"):
                    view_agg[col] = ["sum"]
                elif col.endswith("_call_min"):
                    view_agg[col] = ["min"]
                elif col.endswith("_call_max"):
                    view_agg[col] = ["max"]
                elif pd.api.types.is_numeric_dtype(records[col].dtype):
                    view_agg[col] = [
                        "sum",
                        "min",
                        "max",
                        "mean",
                        "std",
                    ]
                    if self.quantile_stats:
                        if is_dask:
                            view_agg[col].append(quantile_stats(0.01, 0.99))
                            view_agg[col].append(quantile_stats(0.05, 0.95))
                            view_agg[col].append(quantile_stats(0.1, 0.9))
                            view_agg[col].append(quantile_stats(0.25, 0.75))
                        else:
                            raise NotImplementedError("Quantile statistics not implemented for non-Dask DataFrames.")
                elif pd.api.types.is_string_dtype(records[col].dtype) or pd.api.types.is_object_dtype(records[col].dtype):
                    if is_dask:
                        view_agg[col] = [unique_set_flatten()]
                    else:
                        view_agg[col] = [unique_set_flatten_pd]
                else:
                    raise TypeError(
                        f"Unsupported data type '{records[col].dtype}' for column '{col}'. "
                        f"Developer must add explicit handling for this data type in _compute_view method."
                    )
            if is_dask:
                view_agg.update({col: [unique_set()] for col in local_view_types_diff})
            else:
                view_agg.update({col: [unique_set_pd] for col in local_view_types_diff})

        with log_block("fix_std_cols", layer=layer, view_key=view_key):
            # Fix std columns to avoid pandas extension dtypes producing object arrays inside Dask.
            std_cols = [col for col, aggs in view_agg.items() if isinstance(aggs, list) and "std" in aggs]
            if is_dask:
                records = records.map_partitions(fix_std_cols, std_cols=std_cols)
            else:
                records = fix_std_cols(records, std_cols=std_cols)

        with log_block("pre_grouping", layer=layer, view_key=view_key):
            pre_view = records.reset_index()
            if view_type != COL_PROC_NAME:
                pre_agg = {}
                for col in pre_view.columns:
                    if col in (view_type, COL_PROC_NAME):
                        continue
                    if col.endswith("_call_min"):
                        pre_agg[col] = "min"
                    elif col.endswith("_call_max"):
                        pre_agg[col] = "max"
                    else:
                        pre_agg[col] = "sum"
                pre_view = pre_view.groupby([view_type, COL_PROC_NAME]).agg(pre_agg).reset_index()

        with log_block("groupby_agg_pipeline", layer=layer, view_key=view_key):
            view = pre_view.groupby([view_type]).agg(view_agg)
            if is_dask:
                view = view.map_partitions(_replace_zeros_preserving_stall_time,
                    preserve=self.preset.name == "liveflow-v1" and layer in ("stall_storage", "stall_decode"))
            else:
                view = _replace_zeros_preserving_stall_time(view,
                    preserve=self.preset.name == "liveflow-v1" and layer in ("stall_storage", "stall_decode"))

        with log_block("flatten_column_names", layer=layer, view_key=view_key):
            view = flatten_column_names(view)
            view = view.rename(columns=build_view_rename_map(view.columns))

        with log_block("set_quantile_metrics", layer=layer, view_key=view_key):
            if is_dask:
                view = view.map_partitions(set_quantile_metrics)
            else:
                view = set_quantile_metrics(view)

        with log_block("set_unique_counts+fix_dtypes", layer=layer, view_key=view_key):
            if is_dask:
                view = (
                    view.map_partitions(derive_call_stats)
                    .map_partitions(set_unique_counts, layer=layer)
                    .map_partitions(fix_dtypes)
                )
            else:
                view = derive_call_stats(view)
                view = set_unique_counts(view, layer=layer)
                view = fix_dtypes(view)

        if is_dask:
            view = view.persist()

        return view

    def _process_flat_view(
        self,
        flat_view: pd.DataFrame,
        view_key: ViewKey,
        metric_boundaries: ViewMetricBoundaries,
        system_metrics: Optional[pd.DataFrame] = None,
    ):
        view_type = view_key[-1]
        is_view_process_based = self.is_view_process_based(view_key)
        if self.preset.name == "liveflow-v1":
            from .fastpath import preserve_measured_stall_zeros
            flat_view = preserve_measured_stall_zeros(flat_view)
        with log_block("set_view_metrics", view_key=view_key):
            flat_view = set_view_metrics(
                flat_view,
                is_view_process_based=is_view_process_based,
                metric_boundaries=metric_boundaries[view_type],
            )
        with log_block("set_cross_layer_metrics", view_key=view_key):
            flat_view = set_cross_layer_metrics(
                flat_view,
                async_layers=self.preset.async_layers,
                derived_metrics=self.preset.derived_metrics,
                is_view_process_based=is_view_process_based,
                layers=self.layers,
                layer_deps=self.preset.layer_deps,
                time_boundary_layer=self.preset.time_boundary_layer,
            )
        with log_block("set_additional_metrics", view_key=view_key):
            flat_view = self._set_additional_metrics(
                flat_view,
                view_key=view_key,
            )
        if system_metrics is not None and not system_metrics.empty:
            sys_cols = [c for c in system_metrics.columns if c.startswith("sys_")]
            if sys_cols:
                if COL_TIME_RANGE in view_key:
                    with log_block("join_system_metrics", view_key=view_key):
                        sys_lookup = system_metrics.set_index(COL_TIME_RANGE)[sys_cols]
                        flat_view = flat_view.join(sys_lookup, how="left")
                else:
                    with log_block("agg_system_metrics", view_key=view_key):
                        agg = {}
                        for c in sys_cols:
                            if "max" in c or "dirty" in c:
                                agg[c] = system_metrics[c].max()
                            else:
                                agg[c] = system_metrics[c].mean()
                        for c, v in agg.items():
                            flat_view[c] = v
                        logger.debug(
                            "system_metrics.joined",
                            view_key=view_key,
                            num_sys_rows=len(system_metrics),
                            sys_columns=sys_cols,
                            values={c: round(float(v), 2) if pd.notna(v) else None for c, v in agg.items()},
                        )
        elif system_metrics is None:
            logger.debug("system_metrics.absent", view_key=view_key)
        return flat_view.sort_index(axis=1)

    @staticmethod
    def _save_flat_view(view: pd.DataFrame, view_path: str):
        view.to_parquet(f"{view_path}.parquet")

    def _set_additional_metrics(self, view: pd.DataFrame, view_key: ViewKey, epsilon=1e-9) -> pd.DataFrame:
        view_type = view_key[-1]
        is_view_process_based = self.is_view_process_based(view_key)
        time_proc_metric = "time_sum" if is_view_process_based else "time_proc_max"
        view_additional_metrics = (self.preset.additional_metrics or {}).get(view_type, {})
        for metric, eval_condition in view_additional_metrics.items():
            eval_condition = eval_condition.format(
                epsilon=epsilon,
                time_interval=self.time_granularity,
                time_metric=time_proc_metric,
            )
            view = view.eval(f"{metric} = {eval_condition}")
            numerator_denominators = extract_numerator_and_denominators(eval_condition)
            if numerator_denominators:
                _, denominators = numerator_denominators
                if denominators:
                    denominator_conditions = [f"({denom}.isna() | {denom} == 0)" for denom in denominators]
                    mask_condition = " & ".join(denominator_conditions)
                    view[metric] = view[metric].mask(view.eval(mask_condition), pd.NA)
        return view

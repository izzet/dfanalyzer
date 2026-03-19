import abc
import dataclasses as dc
from collections import Counter, defaultdict, deque
import dask
import dask.dataframe as dd
import hashlib
import itertools as it
import json
import math
import os
import pandas as pd
import signal
import structlog
import time
from betterset import BetterSet as S
from dask.distributed import fire_and_forget, get_client, wait
from omegaconf import OmegaConf
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Module-level shutdown flag set by SIGTERM handler.
_shutdown_requested = False


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def install_shutdown_handler():
    """Install SIGTERM handler so the analyzer exits gracefully."""
    signal.signal(signal.SIGTERM, _sigterm_handler)

from .analysis_utils import (
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
    COL_FILE_NAME,
    COL_HOST_NAME,
    COL_PROC_NAME,
    COL_TIME_END,
    COL_TIME_RANGE,
    COL_TIME_START,
    VIEW_TYPES,
    Layer,
)
from .metrics import (
    find_layer_time_metrics,
    set_cross_layer_metrics,
    set_main_metrics,
    set_quantile_metrics,
    set_view_metrics,
)
from .fact_engine import FactEngine
from .types import (
    AnalyzerResultType,
    RawStats,
    ViewKey,
    ViewMetricBoundaries,
    ViewType,
    Views,
)
from .utils.dask_agg import quantile_stats, unique_set, unique_set_flatten
from .utils.expr_utils import extract_numerator_and_denominators
from .utils.file_utils import ensure_dir
from .utils.json_encoders import NpEncoder
from .utils.log_utils import console_block, log_block
from .utils.pandas_agg import unique_set_flatten_pd, unique_set_pd
from .utils.pandas_utils import flatten_column_names
from .streaming.window_buffer import WindowBoundaryTracker, WindowBuffer


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
HLM_EXTRA_COLS = ["cat", "io_cat", "acc_pat", "func_name"]
PARTITION_SIZE = "128MB"
VIEW_PERMUTATIONS = False

DataFrameType = Union[dd.DataFrame, pd.DataFrame]

logger = structlog.get_logger()


class Analyzer(abc.ABC):
    def __init__(
        self,
        preset: AnalyzerPresetConfig,
        checkpoint: bool = True,
        checkpoint_dir: str = "",
        debug: bool = False,
        facts_config: Optional[Union[FactsConfig, Dict[str, Any]]] = None,
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
            time_approximate: Whether to use approximate time for I/O operations.
            time_granularity: The time granularity for analysis, in seconds.
            time_resolution: The time resolution for analysis, in microseconds.
            time_sliced: Whether to slice time ranges for analysis.
            verbose: Whether to enable verbose logging.
        """
        if checkpoint:
            assert checkpoint_dir != "", "Checkpoint directory must be defined"

        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_tasks = []
        self.dask_client = get_client()
        self.debug = debug
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
        self.preset = preset
        self.time_approximate = time_approximate
        self.time_granularity = time_granularity
        self.time_resolution = time_resolution
        self.time_sliced = time_sliced
        self.verbose = verbose
        ensure_dir(self.checkpoint_dir)

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
        if self.fact_engine is None:
            return {}
        if not self.facts_config.emit_analysis_facts:
            logger.debug("facts.emit_analysis_facts.disabled")
            return {}
        return self.fact_engine.evaluate(
            flat_views=flat_views,
            raw_stats=raw_stats,
        )

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
    ) -> AnalyzerResultType:
        """Analyzes I/O trace data to identify performance bottlenecks.

        This method orchestrates the entire analysis process, including reading
        trace data, computing various metrics and views, evaluating these views
        to detect bottlenecks, and applying rules to characterize them.

        Args:
            trace_path: Path to the I/O trace file or directory.
            accuracy: The analysis accuracy mode ('optimistic' or 'pessimistic').
            exclude_characteristics: A list of I/O characteristics to exclude.
            logical_view_types: Whether to compute views based on logical relationships.
            metrics: A list of metrics to analyze (e.g., 'iops', 'bw', 'time').
            view_types: A list of view types to compute (e.g., 'file_name', 'proc_name').

        Returns:
            An AnalyzerResultType object containing the analysis results.
        """
        # Check if high-level metrics are checkpointed
        proc_view_types = self.ensure_proc_view_type(view_types=view_types)
        hlm_checkpoint_name = self.get_hlm_checkpoint_name(view_types=proc_view_types)
        traces = None
        raw_stats = None
        with console_block("Read trace & stats"):
            if not self.checkpoint or not self.has_checkpoint(name=hlm_checkpoint_name):
                # Read trace & stats
                with log_block("read_trace"):
                    traces = self.read_trace(
                        trace_path=path,
                        extra_columns=extra_columns,
                        extra_columns_fn=extra_columns_fn,
                    )
                with log_block("read_stats"):
                    raw_stats = self.read_stats(traces=traces)
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
            else:
                # Restore stats
                with log_block("restore_raw_stats"):
                    raw_stats = self.restore_extra_data(
                        name=self.get_stats_checkpoint_name(),
                        fallback=lambda: None,
                    )

        # Compute high-level metrics
        is_dask = isinstance(traces, dd.DataFrame)
        with console_block("Compute high-level metrics"):
            with log_block("compute_high_level_metrics"):
                hlm = self.compute_high_level_metrics(
                    checkpoint_name=hlm_checkpoint_name,
                    traces=traces,
                    view_types=view_types,
                )

            if is_dask:
                with log_block("persist"):
                    (hlm, raw_stats) = dask.persist(hlm, raw_stats)
                with log_block("wait"):
                    wait([hlm, raw_stats])

        return self._analyze_trace(
            traces=traces,
            proc_view_types=proc_view_types,
            logical_view_types=logical_view_types,
            raw_stats=raw_stats,
            metric_boundaries=metric_boundaries,
        )

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
        output_handler: Optional[Callable[[AnalyzerResultType], None]] = None,
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
        output_handler: Optional[Callable[[AnalyzerResultType], None]] = None,
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

            normalized_event = self.normalize_stream_event(
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

            traces = self.handle_stream_events(
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
        output_handler: Optional[Callable[[AnalyzerResultType], None]] = None,
        num_ranks: int = 1,
    ) -> None:
        from .streaming.mofka_io import open_consumer

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
        window_tracker = WindowBoundaryTracker(
            num_ranks=num_ranks,
            require_explicit_start=True,
        )
        selected_trace_drain_grace_ms = max(trace_drain_grace_ms, 0)

        control_wait_timeout_ms = 1000
        trace_drain_timeout_ms = 100
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
            normalized_event = self.normalize_stream_event(
                event=trace_event,
                extra_columns=normalized_extra_columns,
                extra_columns_fn=extra_columns_fn,
            )
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
            results = []
            metadata = mofka_event.metadata

            if isinstance(metadata, dict) and metadata.get("type") == "batch":
                # Batched format: events are in DataView as newline-delimited JSON
                data = mofka_event.data
                if isinstance(data, list):
                    data = b"".join(data)
                if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                    text = data.decode("utf-8", errors="replace")
                    for line in text.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            trace_event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(trace_event, dict):
                            results.append(_normalize_single_trace(trace_event))
            elif isinstance(metadata, dict):
                # Legacy single-event format: event JSON in metadata
                results.append(_normalize_single_trace(metadata))

            if hasattr(mofka_event, "acknowledge"):
                mofka_event.acknowledge()
            return results

        def _append_trace_event_to_window(
            normalized_event: Optional[dict],
            sink_events: List[dict],
            *,
            window_index: Optional[int] = None,
            start_counts: Optional[Dict[int, int]] = None,
        ) -> None:
            if not isinstance(normalized_event, dict):
                return
            event_pid = normalized_event.get(process_key)
            if event_pid is None:
                logger.warning("mofka.trace_event.missing_pid", name=normalized_event.get("name"))
                return

            event_pid = int(event_pid)
            if start_counts and event_pid in start_counts:
                if trace_events_seen_by_pid.get(event_pid, 0) <= start_counts[event_pid]:
                    return
            event_with_window = dict(normalized_event)
            current_window = window_index
            if current_window is None:
                current_window = window_tracker.current_window(event_pid)
            if current_window is None:
                return
            event_with_window["window"] = current_window
            # Keep the historical epoch field for downstream schema compatibility.
            event_with_window["epoch"] = current_window
            event_with_window["step"] = current_window
            sink_events.append(event_with_window)

        trace_events_seen_by_pid = defaultdict(int)
        pending_trace_events = deque()

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

        def _pop_pending_event_for_window(target_counts: Dict[int, int]):
            if not pending_trace_events:
                return None
            skipped_events = deque()
            selected_event = None
            while pending_trace_events:
                candidate = pending_trace_events.popleft()
                _, _, event_pid = candidate
                if _event_exceeds_target(event_pid, target_counts):
                    skipped_events.append(candidate)
                    continue
                selected_event = candidate
                break
            while skipped_events:
                pending_trace_events.appendleft(skipped_events.pop())
            return selected_event

        try:
            control_future = control_consumer.pull()
            trace_future = trace_consumer.pull()
            while not _shutdown_requested:
                # Block on control topic and only drain trace topic once a control
                # boundary arrives; this keeps consumer-side window buffering disabled.
                control_mofka_event = control_future.wait(timeout_ms=control_wait_timeout_ms)
                if control_mofka_event is None:
                    continue

                control_future = control_consumer.pull()
                control_event = control_mofka_event.metadata
                if not isinstance(control_event, dict) or control_event.get("type") != "boundary_event":
                    if hasattr(control_mofka_event, "acknowledge"):
                        control_mofka_event.acknowledge()
                    continue

                try:
                    pid = int(control_event["pid"])
                    events_written = int(control_event["events_written"])
                    trigger_name = str(control_event.get("trigger_event_name", ""))
                except (KeyError, TypeError, ValueError):
                    logger.warning("mofka.control_boundary.invalid", metadata=control_event)
                    if hasattr(control_mofka_event, "acknowledge"):
                        control_mofka_event.acknowledge()
                    continue

                if hasattr(control_mofka_event, "acknowledge"):
                    control_mofka_event.acknowledge()

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

                boundary_ts_us = _parse_control_int(control_event, "trace_ts_us")
                if boundary_ts_us is not None and boundary_ts_us > 0:
                    boundary_ts_ns = boundary_ts_us * 1000
                else:
                    boundary_ts_ns = int(control_event.get("ts_unix_ns", 0))
                completed_windows = window_tracker.observe_end_boundary(
                    pid=pid,
                    boundary_ts_ns=boundary_ts_ns,
                    events_written=events_written,
                )

                for completed_window in completed_windows:
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
                        start_counts = {
                            int(boundary_pid): int(boundary_count)
                            for boundary_pid, boundary_count in completed_window.start_events_written_by_pid.items()
                            if boundary_count is not None and int(boundary_count) > 0
                        }
                        drain_deadline = (
                            time.monotonic()
                            + (selected_trace_drain_grace_ms / 1000.0)
                        )
                        while not _targets_satisfied(target_counts):
                            pending_event = _pop_pending_event_for_window(target_counts)
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
                                if _event_exceeds_target(event_pid, target_counts):
                                    pending_trace_events.append((normalized_event, event_ts, event_pid))
                                    continue
                                _record_trace_event(
                                    normalized_event,
                                    event_pid,
                                    pulled_events,
                                    window_index=completed_window.window_index,
                                    start_counts=start_counts,
                                )
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

                    # Log event category breakdown for debugging
                    cat_counts = Counter(e.get("cat", "?") for e in pulled_events)
                    name_sample = Counter(e.get("name", "?") for e in pulled_events)
                    logger.info(
                        "mofka.window.drain_summary",
                        window=completed_window.window_index,
                        event_count=len(pulled_events),
                        target_counts=target_counts,
                        seen_counts={pid: trace_events_seen_by_pid.get(pid, 0) for pid in sorted(target_counts)},
                        cat_counts=dict(cat_counts),
                        top_names=dict(name_sample.most_common(10)),
                    )

                    traces = self.handle_stream_events(
                        events=pulled_events,
                        view_types=proc_view_types,
                        extra_columns=extra_columns,
                    )
                    try:
                        result = self._analyze_trace(
                            traces=traces,
                            proc_view_types=proc_view_types,
                            logical_view_types=logical_view_types,
                            raw_stats={},
                            metric_boundaries=metric_boundaries,
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
                    logger.info(
                        "mofka.window.analysis_complete",
                        window=completed_window.window_index,
                        event_count=len(pulled_events),
                        num_ranks=num_ranks,
                        flat_views=len(result.flat_views),
                        analysis_facts=len(result.analysis_facts),
                    )
                    output_handler(result)

        finally:
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
        output_handler: Optional[Callable[[AnalyzerResultType], None]] = None,
        num_ranks: int = 1,
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
        )

    def read_stats(self, traces: dd.DataFrame) -> RawStats:
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
        total_event_count = self.get_total_event_count(traces)
        unique_file_count = self.get_unique_file_count(traces)
        unique_host_count = self.get_unique_host_count(traces)
        unique_process_count = self.get_unique_process_count(traces)
        raw_stats = RawStats(
            **self.restore_extra_data(
                name=self.get_stats_checkpoint_name(),
                fallback=lambda: dict(
                    job_time=job_time,
                    time_granularity=self.time_granularity,
                    time_resolution=self.time_resolution,
                    total_event_count=total_event_count,
                    unique_file_count=unique_file_count,
                    unique_host_count=unique_host_count,
                    unique_process_count=unique_process_count,
                ),
            )
        )
        return raw_stats

    @abc.abstractmethod
    def read_trace(
        self,
        trace_path: str,
        extra_columns: Optional[Dict[str, str]],
        extra_columns_fn: Optional[Callable[[dict], dict]],
    ) -> dd.DataFrame:
        """Reads I/O trace data from the specified path.

        This is an abstract method that must be implemented by subclasses
        to handle specific trace formats.

        Args:
            trace_path: Path to the I/O trace file or directory.

        Returns:
            A Dask DataFrame containing the parsed I/O trace data.

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

    def normalize_stream_event(
        self,
        event: dict,
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
    ) -> dict:
        return event

    def handle_stream_events(
        self,
        events: List[dict],
        view_types: List[ViewType],
        extra_columns: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        traces = pd.DataFrame(events)
        return self.postread_trace(traces=traces, view_types=view_types)

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
            time_metric = "time_sum" if self.is_view_process_based(view_key) else "time_max"
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
    def set_layer_metrics(hlm: pd.DataFrame, derived_metrics: Dict[str, str]) -> pd.DataFrame:
        # Create an explicit copy to avoid SettingWithCopyWarning
        hlm = hlm.copy()
        hlm_columns = list(hlm.columns)
        for metric, condition in derived_metrics.items():
            is_data_metric = metric in ["data", "read", "write"]
            for col in hlm_columns:
                is_data_col = col == "size" or "size_bin" in col
                if not is_data_metric and is_data_col:
                    continue
                metric_col = f"{metric}_{col}"
                hlm[metric_col] = pd.NA
                if pd.api.types.is_string_dtype(hlm.dtypes[col]) and not is_data_col:
                    hlm[metric_col] = hlm[metric_col].map(lambda x: S())
                hlm[metric_col] = hlm[metric_col].mask(hlm.eval(condition), hlm[col])
                if not pd.api.types.is_string_dtype(hlm.dtypes[col]):
                    hlm[metric_col] = pd.to_numeric(hlm[metric_col], errors="coerce")
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
        store_flat_view_tasks = []
        for view_key in flat_views:
            flat_view_checkpoint_name = self.get_checkpoint_name(CHECKPOINT_FLAT_VIEW, *list(view_key))
            flat_view_checkpoint_path = self.get_checkpoint_path(name=flat_view_checkpoint_name)
            if self.has_checkpoint(name=flat_view_checkpoint_name):
                continue
            store_flat_view_tasks.append(
                self.dask_client.submit(
                    self._save_flat_view,
                    view=flat_views[view_key],
                    view_path=flat_view_checkpoint_path,
                )
            )
        return store_flat_view_tasks

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
        layer_main_views: Optional[Dict[Layer, DataFrameType]] = None,
        is_dask: bool = True,
    ) -> AnalyzerResultType:
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
                        if hlm is None:
                            raise ValueError("hlm must be provided when layer_main_views is not supplied")
                        layer_hlm = hlm.copy()
                        if layer_condition:
                            layer_hlm = hlm.query(layer_condition)
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
                    with open("metric_boundaries.json", "w") as f:
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
                    )
                    if self.debug:
                        flat_views[view_key].to_csv(f"flat_view_{'_'.join(view_key)}.csv", index=False)

        # Checkpoint flat views if enabled
        if self.checkpoint:
            with log_block("write_flat_view_checkpoints"):
                self.checkpoint_tasks.extend(self.store_flat_views(flat_views=flat_views))

        # Wait for all checkpoint tasks
        if self.checkpoint:
            with log_block("wait_for_checkpoints"):
                wait(self.checkpoint_tasks)

        with log_block("evaluate_fact_rules"):
            analysis_facts = self._evaluate_analysis_facts(
                flat_views=flat_views,
                raw_stats=raw_stats,
            )

        output_flat_views, output_analysis_facts = self._materialize_output_artifacts(
            flat_views=flat_views,
            analysis_facts=analysis_facts,
        )

        return AnalyzerResultType(
            _hlms=hlms,
            _main_views=main_views,
            _metric_boundaries=metric_boundaries,
            analysis_facts=output_analysis_facts,
            checkpoint_dir=self.checkpoint_dir,
            flat_views=output_flat_views,
            layers=self.layers,
            raw_stats=raw_stats,
            view_types=proc_view_types,
            views=views,
        )

    def _analyze_trace(
        self,
        traces: DataFrameType,
        proc_view_types: List[ViewType],
        logical_view_types: bool,
        raw_stats: RawStats,
        metric_boundaries: ViewMetricBoundaries,
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

            if is_dask:
                with log_block("persist"):
                    (hlm, raw_stats) = dask.persist(hlm, raw_stats)
                with log_block("wait"):
                    wait([hlm, raw_stats])

                # Analyze HLM
        result = self._analyze_hlm(
            hlm=hlm,
            is_dask=is_dask,
            logical_view_types=logical_view_types,
            metric_boundaries=metric_boundaries,
            proc_view_types=proc_view_types,
            raw_stats=raw_stats,
        )

        # Attach correct traces & view types
        result._traces = traces
        result.view_types = proc_view_types

        return result

    def _compute_high_level_metrics(
        self,
        traces: DataFrameType,
        view_types: list,
        partition_size: str,
    ) -> DataFrameType:
        # Add layer columns
        hlm_groupby = list(set(view_types).union(HLM_EXTRA_COLS))
        # Build agg_dict
        bin_cols = [col for col in traces.columns if "_bin_" in col]
        view_types_diff = list(set(VIEW_TYPES).difference(view_types))

        hlm_agg = dict(HLM_AGG)
        hlm_agg.update({col: "sum" for col in bin_cols})

        if isinstance(traces, dd.DataFrame):
            hlm_agg.update({col: unique_set() for col in view_types_diff})
            hlm = (
                traces.groupby(hlm_groupby)
                .agg(hlm_agg, split_out=math.ceil(math.sqrt(traces.npartitions)))
                .persist()
                .repartition(partition_size=partition_size)
                .replace(0, pd.NA)
                .map_partitions(fix_hlm_dtypes)
                .persist()
            )
        else:
            hlm_agg.update({col: unique_set_pd for col in view_types_diff})
            hlm = traces.groupby(hlm_groupby).agg(hlm_agg)
            hlm = hlm.replace(0, pd.NA)
            hlm = fix_hlm_dtypes(hlm)

        hlm[bin_cols] = hlm[bin_cols].astype("Int32")

        return hlm

    def _compute_main_view(
        self,
        layer: Layer,
        hlm: DataFrameType,
        view_types: List[ViewType],
        partition_size: str,
    ) -> DataFrameType:
        is_dask = isinstance(hlm, dd.DataFrame)
        with log_block("drop_and_set_metrics", layer=layer):
            # Set layer metrics
            if "posix" not in layer.lower():
                size_cols = [col for col in hlm.columns if col.startswith("size")]
                hlm = hlm.drop(columns=size_cols)  # type: ignore
                if "file_name" in hlm.columns:
                    hlm = hlm.drop(columns=["file_name"])  # type: ignore
            layer_derived_metrics = self.preset.derived_metrics[layer]
            if is_dask:
                hlm = hlm.map_partitions(
                    self.set_layer_metrics,
                    derived_metrics=layer_derived_metrics,
                )
            else:
                hlm = self.set_layer_metrics(
                    hlm=hlm,
                    derived_metrics=layer_derived_metrics,
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
                elif col not in HLM_EXTRA_COLS:
                    main_view_agg[col] = "sum"

        with log_block("compute_main_view", layer=layer):
            if is_dask:
                main_view = (
                    hlm.groupby(list(view_types))
                    .agg(main_view_agg, split_out=hlm.npartitions)
                    .map_partitions(set_main_metrics)
                    .replace(0, pd.NA)
                    .map_partitions(fix_dtypes)
                    .persist()
                )
            else:
                main_view = hlm.groupby(list(view_types)).agg(main_view_agg)
                main_view = set_main_metrics(main_view).replace(0, pd.NA)
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
                pre_view = pre_view.groupby([view_type, COL_PROC_NAME]).sum().reset_index()

        with log_block("groupby_agg_pipeline", layer=layer, view_key=view_key):
            view = pre_view.groupby([view_type]).agg(view_agg).replace(0, pd.NA)

        with log_block("flatten_column_names", layer=layer, view_key=view_key):
            view = flatten_column_names(view)

        with log_block("set_quantile_metrics", layer=layer, view_key=view_key):
            if is_dask:
                view = view.map_partitions(set_quantile_metrics)
            else:
                view = set_quantile_metrics(view)

        with log_block("set_unique_counts+fix_dtypes", layer=layer, view_key=view_key):
            if is_dask:
                view = view.map_partitions(set_unique_counts, layer=layer).map_partitions(fix_dtypes)
            else:
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
    ):
        view_type = view_key[-1]
        is_view_process_based = self.is_view_process_based(view_key)
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
            flat_view = self._set_additional_metrics(flat_view, is_view_process_based=is_view_process_based)
        return flat_view.sort_index(axis=1)

    @staticmethod
    def _save_flat_view(view: pd.DataFrame, view_path: str):
        view.to_parquet(f"{view_path}.parquet")

    def _set_additional_metrics(self, view: pd.DataFrame, is_view_process_based: bool, epsilon=1e-9) -> pd.DataFrame:
        time_metric = "time_sum" if is_view_process_based else "time_max"
        for metric, eval_condition in self.preset.additional_metrics.items():
            eval_condition = eval_condition.format(
                epsilon=epsilon,
                time_interval=self.time_granularity,
                time_metric=time_metric,
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

import abc
import dask.dataframe as dd
import hashlib
import itertools as it
import json
import math
import numpy as np
import os
import pandas as pd
import structlog
from dask import compute, persist
from dask.distributed import fire_and_forget, get_client, wait
from omegaconf import OmegaConf
from typing import Callable, Dict, List, Optional, Tuple

from .analysis_utils import (
    build_view_rename_map,
    derive_call_stats,
    fix_dtypes,
    fix_std_cols,
    set_file_dir,
    set_file_pattern,
    set_size_bins,
    set_unique_counts,
    split_duration_records_vectorized,
)
from .config import (
    CHECKPOINT_VIEWS,
    HASH_CHECKPOINT_NAMES,
    AdditionalFieldConfig,
    AnalyzerPresetConfig,
)
from .constants import (
    COL_COUNT,
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
    set_cross_layer_metrics,
    set_main_metrics,
    set_view_metrics,
)
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
from .utils.dask_utils import flatten_column_names
from .utils.expr_utils import extract_numerator_and_denominators
from .utils.file_utils import ensure_dir
from .utils.json_encoders import NpEncoder
from .utils.log_utils import console_block, log_block
from .utils.pandas_utils import to_nullable_numeric


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
SUPPORTED_ADDITIONAL_FIELD_AGGS = {"sum", "min", "max", "unique_set"}
PARTITION_SIZE = "128MB"
VIEW_PERMUTATIONS = False

logger = structlog.get_logger()


class Analyzer(abc.ABC):
    def __init__(
        self,
        preset: AnalyzerPresetConfig,
        additional_fields: Optional[Dict[str, AdditionalFieldConfig]] = None,
        checkpoint: bool = True,
        checkpoint_dir: str = "",
        debug: bool = False,
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
        lv = preset.logical_views
        self.logical_views = dict(OmegaConf.to_object(lv)) if OmegaConf.is_config(lv) else dict(lv)  # type: ignore
        self.preset = preset
        self.additional_fields = self._merge_additional_fields(
            preset_additional_fields=getattr(preset, "additional_fields", None),
            override_additional_fields=additional_fields,
        )
        self._additional_field_names = tuple(sorted(self.additional_fields.keys(), key=len, reverse=True))
        self._additional_unique_set_fields = {
            name for name, field_cfg in self.additional_fields.items() if field_cfg.agg == "unique_set"
        }
        self.time_approximate = time_approximate
        self.time_granularity = time_granularity
        self.time_resolution = time_resolution
        self.time_sliced = time_sliced
        self.verbose = verbose
        ensure_dir(self.checkpoint_dir)

    @staticmethod
    def _normalize_additional_fields(
        additional_fields: Optional[Dict[str, AdditionalFieldConfig]],
    ) -> Dict[str, AdditionalFieldConfig]:
        if not additional_fields:
            return {}

        normalized_fields = (
            OmegaConf.to_object(additional_fields) if OmegaConf.is_config(additional_fields) else additional_fields
        )
        result: Dict[str, AdditionalFieldConfig] = {}
        for field_name, field_cfg in normalized_fields.items():  # type: ignore[union-attr]
            if isinstance(field_cfg, AdditionalFieldConfig):
                config = AdditionalFieldConfig(
                    source=field_cfg.source,
                    dtype=field_cfg.dtype,
                    agg=field_cfg.agg.lower(),
                )
            elif isinstance(field_cfg, dict):
                config = AdditionalFieldConfig(
                    source=field_cfg["source"],
                    dtype=field_cfg["dtype"],
                    agg=field_cfg.get("agg", "sum").lower(),
                )
            else:
                raise TypeError(
                    f"additional_fields['{field_name}'] must be an AdditionalFieldConfig or dict, "
                    f"got {type(field_cfg)!r}"
                )
            Analyzer._validate_additional_field(field_name=field_name, field_cfg=config)
            result[field_name] = config
        return result

    @staticmethod
    def _validate_additional_field(field_name: str, field_cfg: AdditionalFieldConfig):
        if not field_cfg.source:
            raise ValueError(f"additional_fields['{field_name}'].source must be defined")
        if field_cfg.agg not in SUPPORTED_ADDITIONAL_FIELD_AGGS:
            raise ValueError(
                f"additional_fields['{field_name}'].agg must be one of "
                f"{sorted(SUPPORTED_ADDITIONAL_FIELD_AGGS)}, got {field_cfg.agg!r}"
            )
        try:
            field_dtype = pd.Series([], dtype=field_cfg.dtype).dtype
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"additional_fields['{field_name}'].dtype={field_cfg.dtype!r} is not a valid pandas dtype"
            ) from error

        if field_cfg.agg != "unique_set" and not pd.api.types.is_numeric_dtype(field_dtype):
            raise ValueError(
                f"additional_fields['{field_name}'].agg={field_cfg.agg!r} requires a numeric dtype, "
                f"got {field_cfg.dtype!r}"
            )

    @classmethod
    def _merge_additional_fields(
        cls,
        preset_additional_fields: Optional[Dict[str, AdditionalFieldConfig]],
        override_additional_fields: Optional[Dict[str, AdditionalFieldConfig]],
    ) -> Dict[str, AdditionalFieldConfig]:
        preset_fields = cls._normalize_additional_fields(preset_additional_fields)
        override_fields = cls._normalize_additional_fields(override_additional_fields)
        return {**preset_fields, **override_fields}

    @staticmethod
    def _resolve_additional_field_source(json_dict: dict, source: str):
        value = json_dict
        for part in source.split("."):
            if not hasattr(value, '__getitem__') or part not in value:
                return None
            value = value[part]
        return value

    @staticmethod
    def _coerce_additional_field_value(value, dtype: str):
        if value is None or value is pd.NA:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

        dtype_obj = pd.Series([], dtype=dtype).dtype
        if pd.api.types.is_string_dtype(dtype_obj):
            return str(value)
        if pd.api.types.is_bool_dtype(dtype_obj):
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "t", "yes", "y"}:
                    return True
                if lowered in {"0", "false", "f", "no", "n"}:
                    return False
            return bool(value)
        if pd.api.types.is_integer_dtype(dtype_obj):
            return int(value)
        if pd.api.types.is_float_dtype(dtype_obj):
            return float(value)
        return value

    def _build_additional_field_columns(self) -> Dict[str, str]:
        return {field_name: field_cfg.dtype for field_name, field_cfg in self.additional_fields.items()}

    def _build_additional_field_extractor(self) -> Callable[[dict], dict]:
        field_specs = tuple(
            (field_name, field_cfg.source, field_cfg.dtype) for field_name, field_cfg in self.additional_fields.items()
        )

        def extract_additional_fields(json_dict: dict) -> dict:
            values = {}
            for field_name, source, dtype in field_specs:
                value = Analyzer._resolve_additional_field_source(json_dict=json_dict, source=source)
                if value is None:
                    continue
                coerced_value = Analyzer._coerce_additional_field_value(value=value, dtype=dtype)
                if coerced_value is not None:
                    values[field_name] = coerced_value
            return values

        return extract_additional_fields

    def _match_additional_field(self, col: str) -> Optional[str]:
        for field_name in self._additional_field_names:
            if col == field_name or col.endswith(f"_{field_name}"):
                return field_name
        return None

    @staticmethod
    def _matches_suffix(col: str, suffixes: List[str]) -> bool:
        return any(col == suffix or col.endswith(f"_{suffix}") for suffix in suffixes)

    def _get_hlm_additional_aggregations(self) -> Dict[str, object]:
        hlm_fields = set(self.preset.hlm_fields)
        hlm_agg = {}
        for field_name, field_cfg in self.additional_fields.items():
            if field_name in hlm_fields:
                continue
            if field_cfg.agg == "unique_set":
                hlm_agg[field_name] = unique_set()
            else:
                hlm_agg[field_name] = field_cfg.agg
        return hlm_agg

    def _get_rollup_aggregation(self, col: str, view_type_suffixes: List[str]):
        if self._matches_suffix(col=col, suffixes=view_type_suffixes):
            return unique_set_flatten()
        # Per-call statistics columns from two-track stats
        if col.endswith("_call_min"):
            return "min"
        if col.endswith("_call_max"):
            return "max"
        if col.endswith("_sq"):
            return "sum"
        additional_field = self._match_additional_field(col)
        if additional_field is None:
            return "sum"
        if self.additional_fields[additional_field].agg == "unique_set":
            return unique_set_flatten()
        return self.additional_fields[additional_field].agg

    def analyze_trace(
        self,
        trace_path: str,
        view_types: List[ViewType],
        exclude_characteristics: List[str] = [],
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        logical_view_types: bool = False,
        metric_boundaries: ViewMetricBoundaries = {},
        unoverlapped_posix_only: Optional[bool] = False,
    ) -> AnalysisResult:
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
            An AnalysisResult object containing the analysis results.
        """
        proc_view_types = self.ensure_proc_view_type(view_types=view_types)
        hlm_checkpoint_name = self.get_hlm_checkpoint_name(view_types=proc_view_types)
        if extra_columns is None and extra_columns_fn is None and self.additional_fields:
            extra_columns = self._build_additional_field_columns()
            extra_columns_fn = self._build_additional_field_extractor()
        read_result = None
        profiles = None
        traces = None
        raw_stats = None
        with console_block("Read trace & stats"):
            with log_block("read_trace"):
                read_result = self.read_trace(
                    trace_path=trace_path,
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
                traces = self.postread_trace(traces=traces, view_types=proc_view_types)
            with log_block("set_file_format"):
                traces = self.set_file_format(traces)
                # Profile events (ph="C" from aggregator) share the
                # view_types groupby, so they must also carry the
                # derived file_format column or _compute_profile_hlm
                # will KeyError at groupby time.
                if profiles is not None:
                    profiles = self.set_file_format(profiles)
                    read_result.profiles = profiles
            with log_block("time_correlation"):
                traces = self.apply_time_correlation(
                    traces, extra_boundary_sources=[profiles],
                )
                # Also correlate profile events. Semantic span events
                # may live in EITHER stream depending on aggregation
                # mode: in SELECTIVE mode (cat IN POSIX/STDIO) spans
                # stay as X events in traces; in FULL mode they become
                # C events in profiles. apply_time_correlation tries
                # each candidate source until it finds boundaries.
                if profiles is not None:
                    profiles = self.apply_time_correlation(
                        profiles,
                        boundary_source=traces,
                        extra_boundary_sources=[profiles],
                    )
                    read_result.profiles = profiles
            with log_block("set_size_bins"):
                traces = traces.map_partitions(set_size_bins)
            if self.time_sliced:
                with log_block("split_duration_records_vectorized"):
                    traces = traces.map_partitions(
                        split_duration_records_vectorized,
                        time_granularity=self.time_granularity,
                        time_resolution=self.time_resolution,
                    )
            read_result.traces = traces

        # Compute high-level metrics
        with console_block("Compute high-level metrics"):
            if profiles is None:
                with log_block("compute_high_level_metrics"):
                    hlm_checkpoint_name = self.get_hlm_checkpoint_name(view_types=proc_view_types)
                    hlm = self.compute_high_level_metrics(
                        checkpoint_name=hlm_checkpoint_name,
                        traces=traces,
                        view_types=proc_view_types,
                    )
                trace_hlm = None
                profile_hlm = None
                with log_block("persist"):
                    (hlm, raw_stats) = persist(hlm, raw_stats)
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
                    (trace_hlm, profile_hlm, raw_stats) = persist(trace_hlm, profile_hlm, raw_stats)
                with log_block("wait"):
                    wait([trace_hlm, profile_hlm, raw_stats])

        # Validate time granularity
        # self.validate_time_granularity(hlm=hlm, view_types=hlm_view_types)

        # Compute system metrics (small table, safe to materialize)
        system_metrics = None
        if read_result.system_metrics is not None:
            with log_block("compute_system_metrics"):
                system_metrics = read_result.system_metrics.compute()

        # Analyze HLM
        result = self._analyze_hlm(
            hlm=hlm,
            logical_view_types=logical_view_types,
            metric_boundaries=metric_boundaries,
            profile_hlm=profile_hlm,
            proc_view_types=proc_view_types,
            raw_stats=raw_stats,
            system_metrics=system_metrics,
            trace_hlm=trace_hlm,
        )

        # Attach correct traces & view types
        result._read_result = read_result
        result.view_types = view_types

        # Return result
        return result

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
        unique_file_count = self.get_unique_file_count(traces, profiles)
        unique_host_count = self.get_unique_host_count(traces, profiles)
        unique_process_count = self.get_unique_process_count(traces, profiles)
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

    def compute_high_level_metrics(
        self,
        traces: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str = PARTITION_SIZE,
        checkpoint_name: Optional[str] = None,
    ) -> dd.DataFrame:
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

    def compute_hybrid_hlm(
        self,
        traces: dd.DataFrame,
        profiles: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str = PARTITION_SIZE,
        checkpoint_name: Optional[str] = None,
    ) -> dd.DataFrame:
        checkpoint_name = checkpoint_name or self.get_hlm_checkpoint_name(view_types)
        return self.restore_view(
            name=checkpoint_name,
            fallback=lambda: self._compute_hybrid_hlm(
                partition_size=partition_size,
                profiles=profiles,
                traces=traces,
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
        return self._reconcile_hlm(
            layer=layer,
            partition_size=partition_size,
            profile_hlm=profile_hlm,
            trace_hlm=trace_hlm,
            view_types=view_types,
        )

    def compute_main_view(
        self,
        layer: Layer,
        hlm: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str = PARTITION_SIZE,
    ) -> dd.DataFrame:
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
        main_view: dd.DataFrame,
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

    def get_time_boundary_layer(self):
        return list(self.preset.layer_defs)[0]

    def get_total_event_count(self, traces: dd.DataFrame) -> int:
        """Computes the total number of I/O events in the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            The total count of I/O events as an integer.
        """
        return traces.index.count().persist()

    def get_profile_event_count(self, profiles: Optional[dd.DataFrame]):
        # Profile rows from aggregated ph="C" events represent N coalesced
        # syscalls each; COL_COUNT (mapped from dft_cnt by
        # _standardize_profile_partition) carries that N. The total
        # syscall count is the sum of COL_COUNT across rows, never the
        # row count itself — falling back to row-count would silently
        # underreport whenever aggregation is on. If COL_COUNT is
        # missing we treat profiles as empty (an invariant violation
        # upstream would surface as a KeyError elsewhere).
        if profiles is None:
            return 0
        if COL_COUNT not in profiles.columns:
            return 0
        return profiles[COL_COUNT].fillna(0).sum().persist()

    def get_unique_host_count(
        self,
        traces: dd.DataFrame,
        profiles: Optional[dd.DataFrame] = None,
    ):
        """Computes the total number of unique hosts accessed.

        When ``profiles`` is provided and shares the host column, the
        unique count is taken over the union of trace and profile rows so
        aggregated events contribute. Without this, hosts seen only via
        aggregated profile events would be missed.
        """
        if profiles is not None and COL_HOST_NAME in profiles.columns:
            return dd.concat(
                [traces[COL_HOST_NAME], profiles[COL_HOST_NAME]],
                interleave_partitions=True,
            ).nunique()
        return traces[COL_HOST_NAME].nunique()

    def get_unique_file_count(
        self,
        traces: dd.DataFrame,
        profiles: Optional[dd.DataFrame] = None,
    ):
        """Computes the total number of unique files accessed.

        When ``profiles`` is provided and shares the file column, the
        unique count is taken over the union of trace and profile rows so
        aggregated events contribute. Without this, files seen only via
        aggregated profile events would be missed.
        """
        if profiles is not None and COL_FILE_NAME in profiles.columns:
            return dd.concat(
                [traces[COL_FILE_NAME], profiles[COL_FILE_NAME]],
                interleave_partitions=True,
            ).nunique()
        return traces[COL_FILE_NAME].nunique()

    def get_unique_process_count(
        self,
        traces: dd.DataFrame,
        profiles: Optional[dd.DataFrame] = None,
    ):
        """Computes the total number of unique processes accessed.

        When ``profiles`` is provided and shares the proc column, the
        unique count is taken over the union of trace and profile rows so
        aggregated events contribute.
        """
        if profiles is not None and COL_PROC_NAME in profiles.columns:
            return dd.concat(
                [traces[COL_PROC_NAME], profiles[COL_PROC_NAME]],
                interleave_partitions=True,
            ).nunique()
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
                        data=self.dask_client.submit(compute, data),
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

    def set_file_format(self, traces: dd.DataFrame) -> dd.DataFrame:
        """Derive file_format column from file_name extension."""
        traces['file_format'] = (
            traces[COL_FILE_NAME].str.extract(r'\.([^./]+)$', expand=False).str.lower().fillna('unknown')
        )
        return traces

    def apply_time_correlation(
        self,
        traces: dd.DataFrame,
        boundary_source: Optional[dd.DataFrame] = None,
        extra_boundary_sources: Optional[List[dd.DataFrame]] = None,
    ) -> dd.DataFrame:
        """Propagate fields from source-layer events to all events via time-window overlap.

        Boundary extraction tries each source in order: first
        ``boundary_source`` if supplied (else ``traces``), then each of
        ``extra_boundary_sources``. The first non-empty source wins. This is
        needed because with aggregation enabled, the semantic spans (step /
        llm / tool) may end up in either the trace stream (SELECTIVE mode,
        where they aren't aggregated and stay as X events) or the profile
        stream (FULL mode, where they get aggregated into C events); the
        correlator needs to find them wherever they are.
        """
        tc = self.preset.time_correlation
        if not tc or not tc.enabled or not tc.field:
            return traces
        fields = list(tc.field)
        fields = [f for f in fields if f in traces.columns]
        if not fields:
            return traces
        source_candidates = []
        source_candidates.append(boundary_source if boundary_source is not None else traces)
        if extra_boundary_sources:
            source_candidates.extend(s for s in extra_boundary_sources if s is not None)
        boundaries = None
        boundary_cols = [COL_TIME_START, COL_TIME_END] + fields
        for cand in source_candidates:
            if cand is None:
                continue
            cand_fields = [f for f in fields if f in cand.columns]
            if not cand_fields:
                continue
            if tc.layer and tc.layer in (self.preset.layer_defs or {}):
                layer_query = self.preset.layer_defs[tc.layer]
                try:
                    source = cand.query(layer_query)
                except Exception:
                    continue
            else:
                source = cand[cand[cand_fields[0]].notna()]
            try:
                candidate_boundaries = source[boundary_cols].compute()
            except Exception:
                continue
            if not candidate_boundaries.empty:
                boundaries = candidate_boundaries
                break
        if boundaries is None or boundaries.empty:
            return traces
        boundaries = boundaries.sort_values(COL_TIME_START).reset_index(drop=True)
        traces = traces.map_partitions(
            self._correlate_partition,
            boundaries=boundaries,
            fields=fields,
        )
        return traces

    @staticmethod
    def _correlate_partition(partition, boundaries, fields):
        """Assign field values to events based on time-window overlap with boundary events.

        Computes the time-window index mapping once, then fills all requested
        fields in a single pass over the partition.
        """
        partition = partition.copy()
        boundary_starts = boundaries[COL_TIME_START].to_numpy(dtype='int64', na_value=0)
        boundary_ends = boundaries[COL_TIME_END].to_numpy(dtype='int64', na_value=0)
        if len(boundary_starts) == 0:
            return partition

        for field in fields:
            needs_fill = partition[field].isna()
            if not needs_fill.any():
                continue
            event_starts = partition.loc[needs_fill, COL_TIME_START].to_numpy(
                dtype='int64',
                na_value=0,
            )
            if len(event_starts) == 0:
                continue
            indices = np.searchsorted(boundary_starts, event_starts, side='right') - 1
            valid = (indices >= 0) & (indices < len(boundary_starts))
            clipped = np.clip(indices, 0, max(len(boundary_starts) - 1, 0))
            valid &= (event_starts >= boundary_starts[clipped]) & (event_starts <= boundary_ends[clipped])
            boundary_values = boundaries[field]
            if pd.api.types.is_string_dtype(boundary_values):
                values = boundary_values.to_numpy(dtype='object')
                result = np.full(len(event_starts), fill_value=None, dtype='object')
            else:
                values = boundary_values.to_numpy(dtype='float64', na_value=np.nan)
                result = np.full(len(event_starts), fill_value=np.nan, dtype='float64')
            result[valid] = values[clipped[valid]]
            partition.loc[needs_fill, field] = result
        return partition

    @staticmethod
    def set_layer_metrics(
        hlm: pd.DataFrame,
        derived_metrics: Dict[str, str],
        size_derived_metrics: Optional[List[str]] = None,
        set_like_suffixes: Optional[List[str]] = None,
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
            max_io_time = hlm.groupby(view_types, dropna=False)["io_time"].sum().max().compute()
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
        hlm: Optional[dd.DataFrame],
        proc_view_types: List[ViewType],
        metric_boundaries: ViewMetricBoundaries,
        trace_hlm: Optional[dd.DataFrame],
        profile_hlm: Optional[dd.DataFrame],
        raw_stats: RawStats,
        logical_view_types: bool,
        layer_main_views: Optional[Dict[Layer, dd.DataFrame]] = None,
        system_metrics: Optional[pd.DataFrame] = None,
    ) -> AnalysisResult:
        """
        Analyze the high-level metrics (HLM) and compute views for each layer.

        This method computes the main views and additional views for each layer, either from the provided
        high-level metrics DataFrame (`hlm`) or from precomputed main views (`layer_main_views`). At least
        one of `hlm` or `layer_main_views` must be provided. If `layer_main_views` is given and contains
        a main view for a layer, it will be used; otherwise, the main view will be computed from `hlm`.

        Args:
            hlm (dd.DataFrame): The high-level metrics Dask DataFrame. Required unless layer-specific HLM data or
                all main views are provided in `layer_main_views`.
            proc_view_types (List[ViewType]): List of view types to process for each layer.
            metric_boundaries (ViewMetricBoundaries): Boundaries for metrics used in view computation.
            trace_hlm (Optional[dd.DataFrame]): Trace-derived HLM used for per-layer reconciliation.
            profile_hlm (Optional[dd.DataFrame]): Profile-derived HLM used for per-layer reconciliation.
            raw_stats (RawStats): Raw statistics to be computed alongside the views.
            logical_view_types (bool): Whether to compute logical views in addition to main views.
            layer_main_views (Optional[Dict[Layer, dd.DataFrame]]): Optional dictionary mapping each layer to its
                precomputed main view. If not provided, main views will be computed from `hlm`.

        Returns:
            AnalysisResult: The result of the analysis, including computed views and statistics.

        Raises:
            ValueError: If neither `hlm` nor `layer_main_views` is provided for a required layer.
        """
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

            with log_block("compute_views_and_raw_stats"):
                (views, raw_stats) = compute(views, raw_stats)

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
                        mem_bytes = int(df.memory_usage(deep=True).sum()) if hasattr(df, 'memory_usage') else -1
                        logger.debug(
                            "Flat view created",
                            view_key=view_key,
                            shape=getattr(df, 'shape', None),
                            mem_bytes=mem_bytes,
                        )
                    except Exception as e:
                        logger.exception("Failed to log flat view details", exc_info=e)

            # Compute metric boundaries for flat views
            with log_block("process_flat_views+metric_boundaries"):
                for view_key in flat_views:
                    if view_key in checkpointed_flat_views:
                        continue
                    view_type = view_key[-1]
                    top_layer = list(self.preset.layer_defs)[0]
                    time_proc_suffix = "time_sum" if self.is_view_process_based(view_key) else "time_proc_max"
                    with log_block("calculate_metric_boundary", view_key=view_key):
                        time_boundary = flat_views[view_key][f"{top_layer}_{time_proc_suffix}"].sum()
                        metric_boundaries.setdefault(view_type, {})
                        for layer in self.preset.layer_defs:
                            metric_boundaries[view_type][f"{layer}_{time_proc_suffix}"] = time_boundary
                    with log_block("process_flat_view", view_key=view_key):
                        # Process flat views to compute metrics and scores
                        flat_views[view_key] = self._process_flat_view(
                            flat_view=flat_views[view_key],
                            view_key=view_key,
                            metric_boundaries=metric_boundaries,
                            system_metrics=system_metrics,
                        )

        # Checkpoint flat views if enabled
        if self.checkpoint:
            with log_block("write_flat_view_checkpoints"):
                self.checkpoint_tasks.extend(self.store_flat_views(flat_views=flat_views))

        # Wait for all checkpoint tasks
        if self.checkpoint:
            with log_block("wait_for_checkpoints"):
                wait(self.checkpoint_tasks)

        return AnalysisResult(
            _hlms=hlms,
            _main_views=main_views,
            _metric_boundaries=metric_boundaries,
            additional_metrics={
                view_type: list(metrics.keys())
                for view_type, metrics in (self.preset.additional_metrics or {}).items()
            },  # type: ignore
            checkpoint_dir=self.checkpoint_dir,
            flat_views=flat_views,
            layers=self.layers,
            raw_stats=raw_stats,
            view_types=proc_view_types,
            views=views,
        )

    def _compute_high_level_metrics(
        self,
        traces: dd.DataFrame,
        view_types: list,
        partition_size: str,
    ) -> dd.DataFrame:
        # Add layer columns (use preset hlm_fields instead of hardcoded HLM_EXTRA_COLS)
        hlm_groupby = list(dict.fromkeys(view_types + list(self.preset.hlm_fields)))
        # Profile DataFrames (from ph="C" aggregated events) can lack columns
        # that are derived during trace post-processing or that come from
        # `args.*` fields only present on trace events (e.g. io_phase,
        # proc_name, completion_tokens). groupby with dropna=False treats
        # null as a valid group and sum/min/max over all-null columns
        # yields NA, so filling absent columns with NA is a faithful no-op
        # for aggregation.
        missing_groupby_cols = [c for c in hlm_groupby if c not in traces.columns]
        if missing_groupby_cols:
            traces = traces.assign(**{c: pd.NA for c in missing_groupby_cols})
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
        hlm_agg.update({col: unique_set() for col in view_types_diff})
        # Per-call support columns
        hlm_agg["time_sq"] = "sum"
        hlm_agg["size_sq"] = "sum"
        hlm_agg["time_call_min"] = "min"
        hlm_agg["time_call_max"] = "max"
        hlm_agg["size_call_min"] = "min"
        hlm_agg["size_call_max"] = "max"
        # Wire in additional aggregations, filtering out fields already in groupby
        additional_agg = self._get_hlm_additional_aggregations()
        hlm_fields_set = set(hlm_groupby)
        hlm_agg.update({k: v for k, v in additional_agg.items() if k not in hlm_fields_set})
        # Same fallback as above: also cover agg-dict columns that are
        # absent from the profile schema.
        missing_agg_cols = [c for c in hlm_agg if c not in traces.columns]
        if missing_agg_cols:
            traces = traces.assign(**{c: pd.NA for c in missing_agg_cols})
        hlm = (
            traces.groupby(hlm_groupby, dropna=False)
            .agg(hlm_agg, split_out=math.ceil(math.sqrt(traces.npartitions)))
            .persist()
            .repartition(partition_size=partition_size)
            .replace(0, pd.NA)
        )
        hlm[bin_cols] = hlm[bin_cols].astype("Int32")
        return hlm.persist()

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

    def _compute_hybrid_hlm(
        self,
        traces: dd.DataFrame,
        profiles: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str,
    ) -> dd.DataFrame:
        trace_hlm = self._compute_high_level_metrics(
            partition_size=partition_size,
            traces=traces,
            view_types=view_types,
        )
        profile_hlm = self.compute_profile_hlm(
            partition_size=partition_size,
            profiles=profiles,
            view_types=view_types,
        )
        return self._reconcile_hlm(
            partition_size=partition_size,
            profile_hlm=profile_hlm,
            trace_hlm=trace_hlm,
            view_types=view_types,
        )

    def _reconcile_hlm(
        self,
        trace_hlm: dd.DataFrame,
        profile_hlm: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str,
        layer: Optional[Layer] = None,
    ) -> dd.DataFrame:
        hlm_groupby = list(dict.fromkeys(view_types + HLM_EXTRA_COLS))
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
            combined_hlm.groupby(hlm_groupby, dropna=False)
            .agg(hlm_agg, split_out=max(1, math.ceil(math.sqrt(combined_hlm.npartitions))))
            .persist()
            .repartition(partition_size=partition_size)
            .replace(0, pd.NA)
        )
        if bin_cols:
            hlm[bin_cols] = hlm[bin_cols].astype("Int32")
        return hlm.persist()

    def _compute_main_view(
        self,
        layer: Layer,
        hlm: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str,
    ) -> dd.DataFrame:
        with log_block("drop_and_set_metrics", layer=layer):
            size_layers = {configured_layer.lower() for configured_layer in (self.preset.size_layers or [])}
            keep_size_for_layer = layer.lower() in size_layers
            if not keep_size_for_layer:
                size_cols = [col for col in hlm.columns if col.startswith("size")]
                hlm = hlm.drop(columns=size_cols)  # type: ignore
                if "file_name" in hlm.columns:
                    hlm = hlm.drop(columns=["file_name"])  # type: ignore
            view_types_diff = list(set(VIEW_TYPES).difference(view_types))
            hlm = hlm.map_partitions(
                self.set_layer_metrics,
                derived_metrics=self.preset.derived_metrics[layer],
                size_derived_metrics=(self.preset.size_derived_metrics or {}).get(layer.lower(), []),
                set_like_suffixes=sorted(set(view_types_diff).union(self._additional_unique_set_fields)),
            )
        with log_block("build_agg_dict", layer=layer):
            view_types_diff = list(set(VIEW_TYPES).difference(view_types))
            main_view_agg = {}
            for col in hlm.columns:
                if col in HLM_EXTRA_COLS:
                    continue
                main_view_agg[col] = self._get_rollup_aggregation(col=col, view_type_suffixes=view_types_diff)
        with log_block("compute_main_view", layer=layer):
            main_view = (
                hlm.groupby(list(view_types), dropna=False)
                .agg(main_view_agg, split_out=hlm.npartitions)
                .map_partitions(set_main_metrics)
                .replace(0, pd.NA)
                .map_partitions(fix_dtypes, time_sliced=self.time_sliced)
                .persist()
            )
        return main_view

    def _compute_view(
        self,
        layer: Layer,
        records: dd.DataFrame,
        view_key: ViewKey,
        view_type: str,
        view_types: List[ViewType],
    ) -> dd.DataFrame:
        is_view_process_based = self.is_view_process_based(view_key)

        view_types_diff = list(set(VIEW_TYPES).difference(view_types))
        local_view_types = records.index._meta.names
        local_view_types_diff = set(local_view_types).difference([view_type])

        with log_block("build_agg_dict", layer=layer, view_key=view_key):
            view_agg = {}
            for col in records.columns:
                if "_bin_" in col:
                    view_agg[col] = ["sum"]
                elif self._matches_suffix(col=col, suffixes=view_types_diff):
                    view_agg[col] = [unique_set_flatten()]
                elif col in it.chain.from_iterable(self.logical_views.values()):
                    view_agg[col] = [unique_set_flatten()]
                elif (additional_field := self._match_additional_field(col)) and (
                    self.additional_fields[additional_field].agg == "unique_set"
                ):
                    view_agg[col] = [unique_set_flatten()]
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
                        view_agg[col].append(quantile_stats(0.01, 0.99))
                        view_agg[col].append(quantile_stats(0.05, 0.95))
                        view_agg[col].append(quantile_stats(0.1, 0.9))
                        view_agg[col].append(quantile_stats(0.25, 0.75))
                else:
                    raise TypeError(
                        f"Unsupported data type '{records[col].dtype}' for column '{col}'. "
                        f"Developer must add explicit handling for this data type in _compute_view method."
                    )
            view_agg.update(
                {
                    col: [
                        unique_set_flatten() if view_type != COL_PROC_NAME and col != COL_PROC_NAME else unique_set()
                    ]
                    for col in local_view_types_diff
                }
            )

        with log_block("fix_std_cols", layer=layer, view_key=view_key):
            # Fix std columns to avoid pandas extension dtypes producing object arrays inside Dask.
            std_cols = [col for col, aggs in view_agg.items() if isinstance(aggs, list) and "std" in aggs]
            records = records.map_partitions(fix_std_cols, std_cols=std_cols)

        with log_block("pre_grouping", layer=layer, view_key=view_key):
            pre_view = records.reset_index()
            if view_type != COL_PROC_NAME:
                pre_group_agg = {}
                for col in pre_view.columns:
                    if col in {view_type, COL_PROC_NAME}:
                        continue
                    if col in local_view_types_diff:
                        pre_group_agg[col] = unique_set()
                    else:
                        pre_group_agg[col] = self._get_rollup_aggregation(col=col, view_type_suffixes=view_types_diff)
                pre_view = pre_view.groupby([view_type, COL_PROC_NAME], dropna=False).agg(pre_group_agg).reset_index()

        with log_block("groupby_agg_pipeline", layer=layer, view_key=view_key):
            view = pre_view.groupby([view_type], dropna=False).agg(view_agg).replace(0, pd.NA)
        with log_block("finalize", layer=layer, view_key=view_key):
            view = flatten_column_names(view)
            view = view.rename(columns=build_view_rename_map(view.columns))
            view = (
                view.map_partitions(derive_call_stats)
                .map_partitions(set_unique_counts, layer=layer)
                .map_partitions(fix_dtypes, time_sliced=self.time_sliced)
                .persist()
            )

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
                time_boundary_layer=self.get_time_boundary_layer(),
            )
        with log_block("set_additional_metrics", view_key=view_key):
            flat_view = self._set_additional_metrics(
                flat_view,
                view_key=view_key,
            )
        if system_metrics is not None and not system_metrics.empty and COL_TIME_RANGE in view_key:
            with log_block("join_system_metrics", view_key=view_key):
                sys_cols = [c for c in system_metrics.columns if c.startswith("sys_")]
                if sys_cols:
                    sys_lookup = system_metrics.set_index(COL_TIME_RANGE)[sys_cols]
                    flat_view = flat_view.join(sys_lookup, how="left")
        return flat_view.sort_index(axis=1)

    @staticmethod
    def _save_flat_view(view: pd.DataFrame, view_path: str):
        view.to_parquet(f"{view_path}.parquet")

    def _set_additional_metrics(self, view: pd.DataFrame, view_key: ViewKey, epsilon=1e-9) -> pd.DataFrame:
        view_type = view_key[-1]
        is_view_process_based = self.is_view_process_based(view_key)
        time_proc_metric = "time_sum" if is_view_process_based else "time_proc_max"
        view_additional_metrics = (self.preset.additional_metrics or {}).get(view_type, {})
        if view_additional_metrics:
            # Cast non-float numeric columns to float64 for pd.eval compatibility
            for col in view.columns:
                dtype = view[col].dtype
                if pd.api.types.is_extension_array_dtype(dtype) or dtype == 'object':
                    try:
                        view[col] = pd.to_numeric(view[col], errors='coerce')
                    except (ValueError, TypeError):
                        pass
        for metric, eval_condition in view_additional_metrics.items():
            eval_condition = eval_condition.format(
                epsilon=epsilon,
                time_interval=self.time_granularity,
                time_metric=time_proc_metric,
            )
            # Derived metric formulas can reference columns that are absent
            # when the HLM comes from profile-only events (e.g. llm_*
            # columns that require per-event args.* fields). A missing
            # column manifests as UndefinedVariableError from pandas eval;
            # in that case we set the metric to NA and move on instead of
            # aborting the whole analysis.
            try:
                view = view.eval(f"{metric} = {eval_condition}")
            except pd.errors.UndefinedVariableError:
                view[metric] = pd.NA
                continue
            numerator_denominators = extract_numerator_and_denominators(eval_condition)
            if numerator_denominators:
                _, denominators = numerator_denominators
                if denominators:
                    denominator_conditions = [f"({denom}.isna() | {denom} == 0)" for denom in denominators]
                    mask_condition = " & ".join(denominator_conditions)
                    try:
                        view[metric] = view[metric].mask(view.eval(mask_condition), pd.NA)
                    except pd.errors.UndefinedVariableError:
                        pass
        return view

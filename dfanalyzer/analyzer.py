import abc
import dask
import dask.dataframe as dd
import hashlib
import itertools as it
import json
import math
import numpy as np
import os
import pandas as pd
from distributed import fire_and_forget, get_client, wait
from typing import Callable, Dict, List, Optional, Tuple, Union

from .analysis_utils import (
    fix_dtypes,
    set_file_dir,
    set_file_pattern,
    set_size_bins,
    set_unique_counts,
    split_duration_records_vectorized,
)
from .config import CHECKPOINT_VIEWS, HASH_CHECKPOINT_NAMES, AnalyzerPresetConfig
from .constants import (
    COL_PROC_NAME,
    COL_TIME_END,
    COL_TIME_START,
    VIEW_TYPES,
    EventType,
    Layer,
)
from .metrics import (
    set_cross_layer_metrics,
    set_main_metrics,
    set_metric_scores,
    set_view_metrics,
)
from .types import (
    AnalyzerResultType,
    RawStats,
    ViewKey,
    ViewMetricBoundaries,
    ViewType,
    Views,
)
from .utils.dask_agg import quantile_stats, unique_set, unique_set_flatten
from .utils.dask_utils import event_logger
from .utils.expr_utils import extract_numerator_and_denominators
from .utils.file_utils import ensure_dir
from .utils.json_encoders import NpEncoder
from .utils.pandas_agg import unique_set_flatten_pd, unique_set_pd
from .utils.pandas_utils import flatten_column_names
from .utils.streaming import Stream, is_streaming_available


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


class Analyzer(abc.ABC):
    def __init__(
        self,
        preset: AnalyzerPresetConfig,
        checkpoint: bool = True,
        checkpoint_dir: str = "",
        debug: bool = False,
        time_approximate: bool = True,
        time_granularity: float = 1e6,
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
            time_granularity: The time granularity for analysis, in microseconds.
            time_resolution: The time resolution for analysis, in microseconds.
            time_sliced: Whether to slice time ranges for analysis.
            verbose: Whether to enable verbose logging.
        """
        if checkpoint:
            assert checkpoint_dir != "", "Checkpoint directory must be defined"

        self.additional_metrics = preset.additional_metrics or {}
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.debug = debug
        self.derived_metrics = preset.derived_metrics or {}
        self.layer_defs = preset.layer_defs
        self.layer_deps = preset.layer_deps or {}
        self.layers = list(preset.layer_defs.keys())
        self.logical_views = preset.logical_views or {}
        self.preset = preset
        self.threaded_layers = preset.threaded_layers or []
        self.time_approximate = time_approximate
        self.time_granularity = time_granularity
        self.time_resolution = time_resolution
        self.time_sliced = time_sliced
        self.unscored_metrics = preset.unscored_metrics or []
        self.verbose = verbose
        ensure_dir(self.checkpoint_dir)

    def analyze_file(
        self,
        path: str,
        view_types: List[ViewType],
        exclude_characteristics: List[str] = [],
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        logical_view_types: bool = False,
        metric_boundaries: ViewMetricBoundaries = {},
        percentile: Optional[float] = None,
        threshold: Optional[int] = None,
        time_view_type: Optional[ViewType] = None,
        unoverlapped_posix_only: Optional[bool] = False,
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
            percentile: The percentile to use for identifying critical views.
                        Mutually exclusive with 'threshold'.
            threshold: The threshold value for slope-based bottleneck detection.
                       Mutually exclusive with 'percentile'.
            view_types: A list of view types to compute (e.g., 'file_name', 'proc_name').

        Returns:
            An AnalyzerResultType object containing the analysis results.

        Raises:
            ValueError: If neither 'percentile' nor 'threshold' is defined.
        """
        # Check if both percentile and threshold are none
        if percentile is None and threshold is None:
            raise ValueError("Either percentile or threshold must be defined")
        is_slope_based = threshold is not None

        # Check if high-level metrics are checkpointed
        view_types = list(sorted(view_types))
        hlm_checkpoint_name = self.get_hlm_checkpoint_name(view_types=view_types)
        traces = None
        raw_stats = None
        if not self.checkpoint or not self.has_checkpoint(name=hlm_checkpoint_name):
            # Read trace & stats
            traces = self.read_trace(
                trace_path=path,
                extra_columns=extra_columns,
                extra_columns_fn=extra_columns_fn,
            )
            raw_stats = self.read_stats(traces=traces)
            traces = self.postread_trace(traces=traces, view_types=view_types)
            traces = traces.map_partitions(set_size_bins)
            if self.time_sliced:
                traces = traces.map_partitions(
                    split_duration_records_vectorized,
                    time_granularity=self.time_granularity / self.time_resolution,
                    time_resolution=self.time_resolution,
                )
        else:
            # Restore stats
            raw_stats = self.restore_extra_data(
                name=self.get_stats_checkpoint_name(),
                fallback=lambda: None,
            )

        # return traces

        return self._analyze_trace(
            traces=traces,
            view_types=view_types,
            logical_view_types=logical_view_types,
            raw_stats=raw_stats,
            metric_boundaries=metric_boundaries,
        )

    def analyze_zmq(
        self,
        address: str,
        view_types: List[ViewType],
        exclude_characteristics: List[str] = [],
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        logical_view_types: bool = False,
        metric_boundaries: ViewMetricBoundaries = {},
        percentile: Optional[float] = None,
        threshold: Optional[int] = None,
        time_view_type: Optional[ViewType] = None,
        unoverlapped_posix_only: Optional[bool] = False,
    ) -> AnalyzerResultType:
        extra_columns = {'epoch': 'Int8'}
        extra_columns_fn = lambda json_dict: {'epoch': json_dict.get('epoch', None)}
        analysis_stream = self.read_zmq(
            trace_address=address,
            extra_columns=extra_columns,
            extra_columns_fn=extra_columns_fn,
        )
        # analysis_stream.sink(print)
        analysis_stream = analysis_stream.epoch_window_via_dict()  # .gather()
        analysis_stream = self.postread_zmq(
            analysis_stream,
            view_types=view_types,
            extra_columns=extra_columns,
            extra_columns_fn=extra_columns_fn,
        )
        analysis_stream = analysis_stream.map(
            self._analyze_trace,
            view_types=view_types,
            logical_view_types=logical_view_types,
            raw_stats={},
            metric_boundaries=metric_boundaries,
        )
        return analysis_stream

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
        job_time = self.compute_job_time(traces=traces)
        total_count = self.compute_total_count(traces=traces)
        raw_stats = RawStats(
            **self.restore_extra_data(
                name=self.get_stats_checkpoint_name(),
                fallback=lambda: dict(
                    job_time=job_time,
                    time_granularity=self.time_granularity,
                    time_resolution=self.time_resolution,
                    total_count=total_count,
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
    ) -> Stream:
        """Reads I/O trace data from a ZMQ source.

        Connects to the specified ZMQ address and returns a Stream
        containing the parsed I/O trace data.

        Args:
            trace_address: The ZMQ address to connect to for reading traces.

        Returns:
            A Stream containing the parsed I/O trace data.

        Raises:
            RuntimeError: If streaming is not available.
        """

        if not is_streaming_available:
            raise RuntimeError("Streaming is not available. Please install the required dependencies.")

        s = Stream.from_zmq(trace_address, bind=True).map(lambda msg: msg.decode("utf-8"))
        # s.sink_to_textfile("incoming_trace.txt", mode="w")
        return s  # .scatter()

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
        trace_stream: Stream,
        view_types: List[ViewType],
        extra_columns: Optional[Dict[str, str]],
        extra_columns_fn: Optional[Callable[[dict], dict]],
    ) -> Stream:
        """Performs any post-processing on the raw ZMQ trace data.

        This method can be overridden by subclasses to perform additional
        transformations or filtering on the trace data after it has been read.
        By default, it returns the traces unmodified.

        Args:
            trace_stream: A Stream containing the I/O trace data.

        Returns:
            A Stream with any post-processing applied.
        """
        return trace_stream

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

    @event_logger(key=EventType.COMPUTE_HLM, message="Compute high-level metrics")
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

    @event_logger(key=EventType.COMPUTE_MAIN_VIEW, message="Compute main view")
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
        percentile: Optional[float],
        threshold: Optional[int],
        is_slope_based: bool,
    ) -> Views:
        """Computes multifaceted views for each specified metric.

        Iterates through all permutations of view_types for each metric,
        generating different "perspectives" on the data. Each perspective
        is a ViewResult, containing the filtered data and critical items.

        Args:
            main_view: The main aggregated Dask DataFrame.
            metrics: A list of metrics to compute views for.
            metric_boundaries: A dictionary of precomputed metric boundaries.
            percentile: The percentile used to identify critical items in views.
            threshold: The threshold value for slope-based critical item identification.
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
                is_slope_based=is_slope_based,
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
        percentile: Optional[float],
        threshold: Optional[int],
        is_slope_based: bool,
    ):
        """Computes views based on predefined logical relationships in the data.

        This method extends the existing view_results by adding new views
        derived from logical columns (e.g., file directory from file name).

        Args:
            main_view: The main aggregated Dask DataFrame.
            metric_boundaries: A dictionary of precomputed metric boundaries.
            metrics: A list of metrics to compute logical views for.
            percentile: The percentile used to identify critical items in views.
            threshold: The threshold value for slope-based critical item identification.
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
                    is_slope_based=is_slope_based,
                    layer=layer,
                    records=parent_records,
                    view_key=view_key,
                    view_type=view_type,
                    view_types=view_types,
                )
        return logical_views

    @event_logger(key=EventType.COMPUTE_VIEW, message="Compute view")
    def compute_view(
        self,
        layer: Layer,
        view_key: ViewKey,
        view_type: str,
        view_types: List[ViewType],
        records: dd.DataFrame,
        is_slope_based: bool,
    ) -> dd.DataFrame:
        """Computes a single view based on the provided parameters.

        This involves restoring a view from a checkpoint or computing it,
        then filtering it to identify critical items based on percentile or threshold.

        Args:
            metrics: The list of all metrics being analyzed.
            metric: The specific metric for this view.
            metric_boundary: The precomputed boundary for the current metric.
            percentile: The percentile to identify critical items.
            records: The Dask DataFrame (parent records) to compute the view from.
            threshold: The threshold for slope-based critical item identification.
            view_key: The key identifying this specific view.
            view_type: The primary dimension/column for this view.

        Returns:
            A ViewResult object containing the computed view, critical items,
            and filtered records.
        """
        return self.restore_view(
            name=self.get_checkpoint_name(CHECKPOINT_VIEW, str(layer), *list(view_key)),
            fallback=lambda: self._compute_view(
                is_slope_based=is_slope_based,
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

    def get_stats_checkpoint_name(self):
        return self.get_checkpoint_name(CHECKPOINT_RAW_STATS)

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
                    get_client().submit(
                        self.store_extra_data,
                        data=get_client().submit(dask.compute, data),
                        data_path=data_path,
                    )
                )
                return data
            with open(data_path, "r") as f:
                return json.load(f)
        return fallback()

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
                view = fallback()
                if not write_to_disk:
                    return view
                self.store_view(name=name, view=view)
                if not read_from_disk:
                    return view
                get_client().cancel(view)
            return dd.read_parquet(view_path)
        return fallback()

    @staticmethod
    def set_layer_metrics(hlm: pd.DataFrame, derived_metrics: Dict[str, str]) -> pd.DataFrame:
        hlm_columns = list(hlm.columns)
        for metric, condition in derived_metrics.items():
            is_data_metric = metric in ["data", "read", "write"]
            for col in hlm_columns:
                is_data_col = col == "size" or "size_bin" in col
                if not is_data_metric and is_data_col:
                    continue
                metric_col = f"{metric}_{col}"
                hlm[metric_col] = pd.NA
                if hlm.dtypes[col].name == "object":
                    hlm[metric_col] = hlm[metric_col].map(lambda x: set())
                hlm[metric_col] = hlm[metric_col].mask(hlm.eval(condition), hlm[col])
                if hlm.dtypes[col].name != "object":
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

    def store_view(self, name: str, view: dd.DataFrame, compute=True, partition_size="64MB"):
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
        return view.repartition(partition_size=partition_size).to_parquet(
            self.get_checkpoint_path(name=name),
            compute=compute,
            write_metadata_file=True,
        )

    def validate_time_granularity(self, hlm: dd.DataFrame, view_types: List[ViewType]):
        if "io_time" in hlm.columns:
            max_io_time = hlm.groupby(view_types)["io_time"].sum().max().compute()
            if max_io_time > (self.time_granularity / 1e6):
                raise ValueError(
                    f"The max 'io_time' exceeds the 'time_granularity' '{int(self.time_granularity / 1e6)}e6'. "
                    f"Please adjust the 'time_granularity' to '{int(2 * max_io_time)}e6' and rerun the analyzer."
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

    def _analyze_trace(
        self,
        traces: DataFrameType,
        view_types: List[ViewType],
        logical_view_types: bool,
        raw_stats: RawStats,
        metric_boundaries: ViewMetricBoundaries,
    ):
        is_dask = isinstance(traces, dd.DataFrame)
        # print("Is Dask DataFrame:", is_dask)

        hlm_checkpoint_name = self.get_hlm_checkpoint_name(view_types=view_types)
        # Compute high-level metrics
        hlm = self.compute_high_level_metrics(
            checkpoint_name=hlm_checkpoint_name,
            traces=traces,
            view_types=view_types,
        )
        # print("High-level metrics computed:", hlm)

        if is_dask:
            (hlm, raw_stats) = dask.persist(hlm, raw_stats)
            wait([hlm, raw_stats])

        # Validate time granularity
        # self.validate_time_granularity(hlm=hlm, view_types=hlm_view_types)

        # Compute layers & views
        hlms = {}
        main_views = {}
        main_indexes = {}
        views = {}
        view_keys = set()
        for layer, layer_condition in self.layer_defs.items():
            layer_hlm = hlm.copy()
            if layer_condition:
                layer_hlm = hlm.query(layer_condition)
            layer_main_view = self.compute_main_view(
                layer=layer,
                hlm=layer_hlm,
                view_types=view_types,
            )
            layer_main_index = layer_main_view.index.to_frame().reset_index(drop=True)
            layer_views = self.compute_views(
                layer=layer,
                main_view=layer_main_view,
                view_types=view_types,
                percentile=0,
                threshold=0,
                is_slope_based=False,
            )

            if logical_view_types:
                layer_logical_views = self.compute_logical_views(
                    layer=layer,
                    main_view=layer_main_view,
                    views=layer_views,
                    view_types=view_types,
                    percentile=0,
                    threshold=0,
                    is_slope_based=False,
                )
                layer_views.update(layer_logical_views)
            hlms[layer] = layer_hlm
            main_views[layer] = layer_main_view
            main_indexes[layer] = layer_main_index
            views[layer] = layer_views
            view_keys.update(layer_views.keys())

        # print("View keys updated:", view_keys)

        if is_dask:
            (views, raw_stats) = dask.compute(views, raw_stats)

        # Restore checkpointed flat views if available
        checkpointed_flat_views = {}
        if self.checkpoint:
            for view_key in view_keys:
                flat_view_checkpoint_name = self.get_checkpoint_name(CHECKPOINT_FLAT_VIEW, *list(view_key))
                flat_view_checkpoint_path = self.get_checkpoint_path(name=flat_view_checkpoint_name)
                if self.has_checkpoint(name=flat_view_checkpoint_name):
                    checkpointed_flat_views[view_key] = pd.read_parquet(f"{flat_view_checkpoint_path}.parquet")

        # Process views to create flat views
        flat_views = {}
        for layer in views:
            for view_key in views[layer]:
                if view_key in checkpointed_flat_views:
                    flat_views[view_key] = checkpointed_flat_views[view_key]
                    continue
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

        # print("Flat views computed:", flat_views)

        # Compute metric boundaries for flat views
        for view_key in flat_views:
            if view_key in checkpointed_flat_views:
                continue
            view_type = view_key[-1]
            top_layer = list(self.layer_defs)[0]
            time_suffix = "time_sum" if self.is_view_process_based(view_key) else "time_max"
            time_boundary = flat_views[view_key][f"{top_layer}_{time_suffix}"].sum()
            metric_boundaries[view_type] = metric_boundaries.get(view_type, {})
            for layer in self.layer_defs:
                metric_boundaries[view_type][f"{layer}_{time_suffix}"] = time_boundary
            # Process flat views to compute metrics and scores
            flat_views[view_key] = self._process_flat_view(
                flat_view=flat_views[view_key],
                view_key=view_key,
                metric_boundaries=metric_boundaries,
            )

        # print("Flat views processed:", flat_views)

        # Checkpoint flat views if enabled
        if self.checkpoint:
            for view_key in flat_views:
                if view_key in checkpointed_flat_views:
                    continue
                flat_view_checkpoint_name = self.get_checkpoint_name(CHECKPOINT_FLAT_VIEW, *list(view_key))
                flat_view_checkpoint_path = self.get_checkpoint_path(name=flat_view_checkpoint_name)
                flat_views[view_key].to_parquet(f"{flat_view_checkpoint_path}.parquet")

        result = AnalyzerResultType(
            _hlms=hlms,
            _main_views=main_views,
            _metric_boundaries=metric_boundaries,
            _traces=traces,
            checkpoint_dir=self.checkpoint_dir,
            flat_views=flat_views,
            layers=self.layers,
            raw_stats=raw_stats,
            view_types=view_types,
            views=views,
        )

        # print("Result created:", result)

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
        hlm_agg.update({col: sum for col in bin_cols})

        if isinstance(traces, dd.DataFrame):
            hlm_agg.update({col: unique_set() for col in view_types_diff})
            # print("hlm_agg dask", hlm_agg)
            hlm = (
                traces.groupby(hlm_groupby)
                .agg(hlm_agg, split_out=math.ceil(math.sqrt(traces.npartitions)))
                .persist()
                .repartition(partition_size=partition_size)
                .replace(0, np.nan)
                .persist()
            )
        else:
            hlm_agg.update({col: unique_set_pd for col in view_types_diff})
            # print("hlm_agg pandas", hlm_agg)
            hlm = traces.groupby(hlm_groupby).agg(hlm_agg).replace(0, np.nan)

        hlm[bin_cols] = hlm[bin_cols].astype("uint32[pyarrow]")

        return hlm

    def _compute_main_view(
        self,
        layer: Layer,
        hlm: DataFrameType,
        view_types: List[ViewType],
        partition_size: str,
    ) -> DataFrameType:
        is_dask = isinstance(hlm, dd.DataFrame)
        # Set layer metrics
        if "posix" not in layer.lower():
            size_cols = [col for col in hlm.columns if col.startswith("size")]
            hlm = hlm.drop(columns=size_cols)  # type: ignore
            if "file_name" in hlm.columns:
                hlm = hlm.drop(columns=["file_name"])  # type: ignore
        if is_dask:
            hlm = hlm.map_partitions(self.set_layer_metrics, derived_metrics=self.derived_metrics[layer])
        else:
            hlm = self.set_layer_metrics(hlm=hlm, derived_metrics=self.derived_metrics[layer])
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

        # print("main_view_agg pandas", main_view_agg)

        if is_dask:
            main_view = (
                hlm.groupby(list(view_types))
                .agg(main_view_agg, split_out=hlm.npartitions)
                .map_partitions(set_main_metrics)
                .replace(0, np.nan)
                .map_partitions(fix_dtypes)
                .persist()
            )
        else:
            main_view = hlm.groupby(list(view_types)).agg(main_view_agg)
            main_view = set_main_metrics(main_view).replace(0, np.nan)
            main_view = fix_dtypes(main_view)

        return main_view

    def _compute_view(
        self,
        layer: Layer,
        records: DataFrameType,
        view_key: ViewKey,
        view_type: str,
        view_types: List[ViewType],
        is_slope_based: bool,
    ) -> DataFrameType:
        is_dask = isinstance(records, dd.DataFrame)
        is_view_process_based = self.is_view_process_based(view_key)

        view_types_diff = set(VIEW_TYPES).difference(view_types)

        if is_dask:
            local_view_types = records.index._meta.names
        else:
            local_view_types = records.index.names
        # print("local_view_types", local_view_types, "for", view_key)

        local_view_types_diff = set(local_view_types).difference([view_type])

        # print("local_view_types_diff", local_view_types_diff, "for", view_key)

        view_agg = {}
        for col in records.columns:
            if "_bin_" in col:
                view_agg[col] = ["sum"]
            elif any(map(col.endswith, view_types_diff)):
                if is_dask:
                    view_agg[col] = [unique_set_flatten()]
                else:
                    view_agg[col] = [unique_set_flatten_pd]
            else:
                view_agg[col] = [
                    "sum",
                    "min",
                    "max",
                    "mean",
                    "std",
                ]
                if is_dask:
                    view_agg[col].extend(
                        [
                            quantile_stats(0.01, 0.99),
                            quantile_stats(0.05, 0.95),
                            quantile_stats(0.1, 0.9),
                            quantile_stats(0.25, 0.75),
                        ]
                    )

        if is_dask:
            view_agg.update({col: [unique_set()] for col in local_view_types_diff})
        else:
            view_agg.update({col: [unique_set_pd] for col in local_view_types_diff})

        # print("view_agg", view_agg)
        # print("records dtypes", records.reset_index().dtypes)
        # print("time_ranges", list(records.reset_index()["time_range"].unique()))
        # print('records dtypes', records)

        view = records.reset_index().groupby([view_type]).agg(view_agg).replace(0, np.nan)

        if is_dask:
            view = view.map_partitions(set_view_metrics, is_view_process_based=is_view_process_based)
        else:
            view = set_view_metrics(view, is_view_process_based=is_view_process_based)

        view = flatten_column_names(view)

        if is_dask:
            view = view.map_partitions(set_unique_counts, layer=layer).map_partitions(fix_dtypes).persist()
        else:
            view = set_unique_counts(view, layer=layer)
            view = fix_dtypes(view)

        return view

    def _process_flat_view(
        self,
        flat_view: pd.DataFrame,
        view_key: ViewKey,
        metric_boundaries: ViewMetricBoundaries,
    ):
        view_type = view_key[-1]
        is_view_process_based = self.is_view_process_based(view_key)
        print('Setting cross layer metrics...')
        flat_view = set_cross_layer_metrics(
            flat_view,
            layer_defs=self.layer_defs,
            layer_deps=self.layer_deps,
            is_view_process_based=is_view_process_based,
        )
        print('Setting additional metrics...')
        flat_view = self._set_additional_metrics(flat_view, is_view_process_based=is_view_process_based)
        print('Setting metric scores...')
        flat_view = set_metric_scores(
            flat_view,
            metric_boundaries=metric_boundaries[view_type],
            unscored_metrics=self.unscored_metrics,
        )
        return flat_view.sort_index(axis=1)

    def _set_additional_metrics(self, view: pd.DataFrame, is_view_process_based: bool, epsilon=1e-9) -> pd.DataFrame:
        try:
            time_metric = "time_sum" if is_view_process_based else "time_max"
            for metric, eval_condition in self.additional_metrics.items():
                eval_condition = eval_condition.format(
                    epsilon=epsilon,
                    time_interval=self.time_granularity / self.time_resolution,
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
        except Exception as e:
            print(f"Error setting additional metrics: {e}")
        return view

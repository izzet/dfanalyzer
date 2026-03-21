import abc
import colorsys
import dask
import dataclasses as dc
import inflect
import json
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Optional

from .constants import COL_PROC_NAME, HUMANIZED_LAYERS, GiB, Layer, MiB
from .types import (
    AnalysisResult,
    RawStats,
    ViewKey,
    humanized_view_name,
    view_name,
)


@dc.dataclass
class OutputLayerMetrics:
    time: float
    count: int
    size: float
    ops: float
    bandwidth: float
    num_files: int
    num_processes: int
    u_time: Optional[float]
    u_count: Optional[int]
    u_size: Optional[float]


@dc.dataclass
class OutputSummary:
    job_time: float
    layer_metrics: Dict[Layer, OutputLayerMetrics]
    layers: List[Layer]
    time_granularity: float
    time_resolution: float
    trace_event_count: int
    profile_event_count: int
    total_event_count: int
    unique_file_count: int
    unique_host_count: int
    unique_process_count: int


class Output(abc.ABC):
    def __init__(
        self,
        compact: bool = False,
        name: str = "",
        root_only: bool = False,
        view_names: List[str] = [],
    ):
        self.compact = compact
        self.name = name
        # self.output_dir = HydraConfig.get().runtime.output_dir
        self.pluralize = inflect.engine()
        self.root_only = root_only
        self.view_names = view_names

    def handle_result(self, result: AnalysisResult):
        raise NotImplementedError

    def _compute_raw_stats(self, result: AnalysisResult) -> RawStats:
        raw_stats = dask.compute(result.raw_stats)[0]
        if isinstance(raw_stats, dict):
            raw_stats = RawStats(**raw_stats)
        return raw_stats

    def _create_summary(self, result: AnalysisResult, view_key: ViewKey, raw_stats: Optional[RawStats] = None) -> OutputSummary:
        flat_view = result.flat_views[view_key]
        if raw_stats is None:
            raw_stats = self._compute_raw_stats(result)
        summary = OutputSummary(
            job_time=float(raw_stats.job_time),
            layer_metrics={},
            layers=result.layers,
            time_granularity=float(raw_stats.time_granularity),
            time_resolution=float(raw_stats.time_resolution),
            trace_event_count=int(raw_stats.trace_event_count),
            profile_event_count=int(raw_stats.profile_event_count),
            total_event_count=int(raw_stats.total_event_count),
            unique_file_count=int(raw_stats.unique_file_count),
            unique_host_count=int(raw_stats.unique_host_count),
            unique_process_count=int(raw_stats.unique_process_count),
        )
        is_process_based = view_key[-1] == COL_PROC_NAME
        time_metric = 'time_sum' if is_process_based else 'time_proc_max'
        for layer in result.layers:
            times = flat_view.get(f"{layer}_{time_metric}", pd.Series([0.0]))
            time = times.max() if is_process_based else times.sum()
            count = flat_view.get(f"{layer}_count_sum", pd.Series([0])).sum()
            size = None
            if 'posix' in layer:
                size = flat_view.get(f"{layer}_size_sum", pd.Series([0.0])).sum()
            num_files = flat_view.get(f"{layer}_file_name_nunique", pd.Series([0.0])).max()
            num_processes = flat_view.get(f"{layer}_proc_name_nunique", pd.Series([0.0])).max()
            u_time_col = f"u_{layer}_{time_metric}"
            u_time = None
            u_count = None
            u_size = None
            if u_time_col in flat_view:
                u_time_mask = flat_view[u_time_col] > 0
                u_times = flat_view.get(u_time_col, pd.Series([0.0]))[u_time_mask]
                u_time = 0.0
                if u_times.any():
                    u_time = u_times.max() if is_process_based else u_times.sum()
                u_counts = flat_view.get(f"{layer}_count_sum", pd.Series([0.0]))[u_time_mask]
                u_count = u_counts.sum() if u_counts.any() else 0
                if 'posix' in layer:
                    u_sizes = flat_view.get(f"{layer}_size_sum", pd.Series([0.0]))[u_time_mask]
                    u_size = u_sizes.sum() if u_sizes.any() else 0.0
            summary.layer_metrics[layer] = OutputLayerMetrics(
                time=float('nan') if pd.isna(time) else float(time),
                count=int(count),
                size=float('nan') if pd.isna(size) else float(size),
                ops=float('nan') if pd.isna(time) or time == 0 else float(count / time),
                bandwidth=float('nan') if pd.isna(time) or pd.isna(size) or time == 0 else float(size / time),
                num_files=0 if pd.isna(num_files) else int(num_files),
                num_processes=0 if pd.isna(num_processes) else int(num_processes),
                u_time=None if pd.isna(u_time) else float(u_time),
                u_count=None if pd.isna(u_count) else int(u_count),
                u_size=None if pd.isna(u_size) else float(u_size),
            )
        return summary

    def _humanized_layer_name(self, name: str) -> str:
        if name in HUMANIZED_LAYERS:
            return HUMANIZED_LAYERS[name]
        return (
            name.replace('_', ' ')
            .title()
            .replace('Posix', 'POSIX')
            .replace('Stdio', 'STDIO')
            .replace('Gpfs', '(GPFS)')
            .replace('Lustre', '(Lustre)')
            .replace('Ssd', '(SSD)')
        )

    @staticmethod
    def _additional_metric_scale_and_unit(metric: str):
        metric_lower = metric.lower()
        if metric_lower.endswith('_gbps'):
            return GiB, 'GB/s'
        if metric_lower.endswith('_mbps'):
            return MiB, 'MB/s'
        if metric_lower.endswith('_gb'):
            return GiB, 'GB'
        if metric_lower.endswith('_mb'):
            return MiB, 'MB'
        return 1.0, '-'


class ConsoleOutput(Output):
    def __init__(
        self,
        compact: bool = False,
        name: str = "",
        root_only: bool = False,
        show_debug: bool = False,
        show_header: bool = True,
        view_names: List[str] = [],
    ):
        super().__init__(compact, name, root_only, view_names)
        self.show_debug = show_debug
        self.show_header = show_header

    def handle_result(self, result: AnalysisResult):
        raw_stats = self._compute_raw_stats(result)
        print_objects = []
        for view_key in result.flat_views:
            if view_key[-1] not in result.view_types:
                continue
            summary = self._create_summary(result=result, view_key=view_key, raw_stats=raw_stats)
            summary_table = self._create_summary_table(summary=summary, view_key=view_key)
            layer_breakdown_table = self._create_layer_breakdown_table(summary=summary, view_key=view_key)
            print_objects.append(summary_table)
            additional_metrics_table = self._create_additional_metrics_table(result=result, view_key=view_key)
            if additional_metrics_table is not None:
                print_objects.append(additional_metrics_table)
            print_objects.append(layer_breakdown_table)
        console = Console(record=True)
        console.print(*print_objects)

    def _create_layer_breakdown_table(self, summary: OutputSummary, view_key: ViewKey) -> Table:
        breakdown_table_title = "Layer Breakdown"
        show_overlap = False
        if len(summary.layers) > 1:
            breakdown_table_title += " (w/ overlap %)"
            show_overlap = True
        breakdown_table = Table(title=breakdown_table_title, title_style="bold cyan", expand=True)
        breakdown_table.add_column("Layer", style="bold")
        breakdown_table.add_column("Time (s)", justify="right")
        breakdown_table.add_column("Ops", justify="right")
        breakdown_table.add_column("Ops/sec", justify="right")
        breakdown_table.add_column("Size (MB)", justify="right")
        breakdown_table.add_column("Bandwidth (MB/s)", justify="right")
        for layer in summary.layers:
            layer_metrics = summary.layer_metrics[layer]
            if layer_metrics.count == 0:
                continue
            if show_overlap:
                time_str = self._format_val_with_ovlp_pct(layer_metrics.time, layer_metrics.u_time)
                count_str = self._format_val_with_ovlp_pct(layer_metrics.count, layer_metrics.u_count, fmt_int=True)
            else:
                time_str = self._format_val(layer_metrics.time)
                count_str = self._format_val(layer_metrics.count, fmt_int=True)
            size_str = '-'
            if not pd.isna(layer_metrics.size):
                size_value = layer_metrics.size / MiB
                u_size_value = layer_metrics.u_size / MiB if layer_metrics.u_size is not None else None
                if show_overlap:
                    size_str = self._format_val_with_ovlp_pct(size_value, u_size_value)
                else:
                    size_str = self._format_val(size_value)
            ops_str = f"{layer_metrics.ops:.3f}"
            bandwidth_str = '-'
            if not pd.isna(layer_metrics.bandwidth):
                bandwidth_str = f"{layer_metrics.bandwidth / MiB:.3f}"
            breakdown_table.add_row(
                self._humanized_layer_name(layer),
                time_str,
                count_str,
                ops_str,
                size_str,
                bandwidth_str,
            )
        return breakdown_table

    def _create_summary_table(self, summary: OutputSummary, view_key: ViewKey) -> Table:
        view_name = humanized_view_name(view_key, ' ')

        summary_table = Table(title=f"{view_name} Summary", title_style='bold green', expand=True)
        summary_table.add_column(header='Metric', style='bold')
        summary_table.add_column(header='Unit', style='italic')
        summary_table.add_column(header='Value', justify='right')

        summary_table.add_row('Job Time', 'seconds', f"{summary.job_time:.3f}")
        summary_table.add_row('Trace Count', 'count', f"{summary.trace_event_count:,}")
        summary_table.add_row('Profile Count', 'count', f"{summary.profile_event_count:,}")
        summary_table.add_row('Total Count', 'count', f"{summary.total_event_count:,}")
        summary_table.add_row('Total Files', 'count', f"{summary.unique_file_count:,}")
        summary_table.add_row('Total Nodes', 'count', f"{summary.unique_host_count:,}")
        summary_table.add_row('Total Processes', 'count', f"{summary.unique_process_count:,}")

        for layer in summary.layers:
            layer_name = self._humanized_layer_name(layer)
            layer_metrics = summary.layer_metrics[layer]
            if layer_metrics.count == 0:
                continue
            summary_table.add_row(f"{layer_name} Count", 'count', f"{layer_metrics.count:,}")
            if layer_metrics.size > 0:
                avg_xfer_size = layer_metrics.size / layer_metrics.count
                summary_table.add_row(f"{layer_name} Size", 'MB', f"{layer_metrics.size / MiB:.3f}")
                summary_table.add_row(f"{layer_name} Bandwidth", 'MB/s', f"{layer_metrics.bandwidth / MiB:.3f}")
                summary_table.add_row(f"{layer_name} Avg Transfer Size", 'MB', f"{avg_xfer_size / MiB:.3f}")

        return summary_table

    def _create_additional_metrics_table(self, result: AnalysisResult, view_key: ViewKey) -> Optional[Table]:
        if not result.additional_metrics:
            return None

        flat_view = result.flat_views[view_key]
        view_type = view_key[-1]
        view_additional_metrics = result.additional_metrics.get(view_type, [])
        if not view_additional_metrics:
            return None
        view_name = humanized_view_name(view_key, ' ')
        additional_table = Table(title=f"{view_name} Additional Metrics", title_style='bold magenta', expand=True)
        additional_table.add_column(header='Metric', style='bold')
        additional_table.add_column(header='Unit', style='italic')
        additional_table.add_column(header='Non-null', justify='right')
        additional_table.add_column(header='Min', justify='right')
        additional_table.add_column(header='Mean', justify='right')
        additional_table.add_column(header='Max', justify='right')

        found_metric = False
        for metric in view_additional_metrics:
            if metric not in flat_view.columns:
                continue
            metric_series = pd.to_numeric(flat_view[metric], errors='coerce').replace([np.inf, -np.inf], pd.NA)
            scale, unit = self._additional_metric_scale_and_unit(metric)
            metric_series = metric_series / scale
            non_null = int(metric_series.notna().sum())
            if non_null == 0:
                additional_table.add_row(metric, unit, "0", "-", "-", "-")
                found_metric = True
                continue
            additional_table.add_row(
                metric,
                unit,
                f"{non_null:,}",
                f"{float(metric_series.min()):.3f}",
                f"{float(metric_series.mean()):.3f}",
                f"{float(metric_series.max()):.3f}",
            )
            found_metric = True

        if not found_metric:
            return None
        return additional_table

    def _format_val(self, value: float, fmt_int=False) -> str:
        if value is None or value == 0:
            return '-'
        if fmt_int:
            return f"{int(value):,}"
        return f"{value:.3f}"

    def _format_val_with_ovlp_pct(self, value, u_value, fmt_int=False):
        value = value or 0
        if value == 0 or u_value is None:
            value = self._format_val(value, fmt_int)
            return f"{value} (" + ('-' * 4) + ")"
        u_value = u_value or 0
        ovlp_pct = max(0.0, 1.0 - (u_value / value))
        padded_percent = f"{int(round(ovlp_pct * 100)):>3d}%"
        color = self._percentage_color(ovlp_pct)
        value = self._format_val(value, fmt_int)
        return f"{value} ([{color}]{padded_percent}[/{color}])"

    def _percentage_color(self, percentage: float) -> str:
        """
        Convert overlap percentage (0.0-1.0) to color name in hex (for rich).
        0% = red, 100% = green
        """
        percentage = max(0.0, min(1.0, percentage))  # Clamp to [0, 1]
        # HSV: Hue 0.0 (red) to 0.33 (green)
        h = percentage * 0.33  # red to green
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


class JSONOutput(Output):
    def __init__(
        self,
        compact: bool = False,
        file_path: str = "",
        name: str = "",
        root_only: bool = False,
        view_names: List[str] = [],
    ):
        super().__init__(compact, name, root_only, view_names)
        self.file_path = file_path

    def handle_result(self, result: AnalysisResult):
        raw_stats = self._compute_raw_stats(result)
        output = {
            "schema_version": "1",
            "raw_stats": self._create_raw_stats(raw_stats=raw_stats),
            "views": {},
        }
        for view_key in result.flat_views:
            if view_key[-1] not in result.view_types:
                continue
            summary = self._create_summary(result=result, view_key=view_key, raw_stats=raw_stats)
            output["views"][view_name(view_key, separator="/")] = {
                "summary": self._create_summary_payload(summary=summary),
                "additional_metrics": self._create_additional_metrics_payload(result=result, view_key=view_key),
            }

        output_path = self._resolve_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, allow_nan=False)
            f.write("\n")

    def _resolve_output_path(self) -> Path:
        if self.file_path:
            return Path(self.file_path)
        try:
            output_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            output_dir = "."
        return Path(output_dir) / "dfanalyzer_output.json"

    @staticmethod
    def _to_int_or_none(value):
        if value is None or pd.isna(value):
            return None
        return int(value)

    @staticmethod
    def _to_float_or_none(value):
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _create_raw_stats(self, raw_stats: RawStats):
        return {
            "job_time_s": self._to_float_or_none(raw_stats.job_time),
            "time_granularity_s": self._to_float_or_none(raw_stats.time_granularity),
            "time_resolution_ns": self._to_int_or_none(raw_stats.time_resolution),
            "total_event_count": self._to_int_or_none(raw_stats.total_event_count),
            "unique_file_count": self._to_int_or_none(raw_stats.unique_file_count),
            "unique_host_count": self._to_int_or_none(raw_stats.unique_host_count),
            "unique_process_count": self._to_int_or_none(raw_stats.unique_process_count),
        }

    def _create_summary_payload(self, summary: OutputSummary):
        summary_payload = {
            "job_time_s": self._to_float_or_none(summary.job_time),
            "total_event_count": self._to_int_or_none(summary.total_event_count),
            "unique_file_count": self._to_int_or_none(summary.unique_file_count),
            "unique_host_count": self._to_int_or_none(summary.unique_host_count),
            "unique_process_count": self._to_int_or_none(summary.unique_process_count),
            "time_granularity_s": self._to_float_or_none(summary.time_granularity),
            "time_resolution_ns": self._to_int_or_none(summary.time_resolution),
            "layers": {},
        }
        for layer in summary.layers:
            metrics = summary.layer_metrics[layer]
            summary_payload["layers"][layer] = {
                "time_s": self._to_float_or_none(metrics.time),
                "count": self._to_int_or_none(metrics.count),
                "size_bytes": self._to_float_or_none(metrics.size),
                "ops_per_s": self._to_float_or_none(metrics.ops),
                "bandwidth_bps": self._to_float_or_none(metrics.bandwidth),
                "num_files": self._to_int_or_none(metrics.num_files),
                "num_processes": self._to_int_or_none(metrics.num_processes),
                "u_time_s": self._to_float_or_none(metrics.u_time),
                "u_count": self._to_int_or_none(metrics.u_count),
                "u_size_bytes": self._to_float_or_none(metrics.u_size),
            }
        return summary_payload

    def _create_additional_metrics_payload(self, result: AnalysisResult, view_key: ViewKey):
        payload = {}
        flat_view = result.flat_views[view_key]
        view_type = view_key[-1]
        view_additional_metrics = result.additional_metrics.get(view_type, [])
        for metric in view_additional_metrics:
            if metric not in flat_view.columns:
                continue
            metric_series = pd.to_numeric(flat_view[metric], errors='coerce').replace([np.inf, -np.inf], np.nan)
            scale, unit = self._additional_metric_scale_and_unit(metric)
            metric_series = metric_series / scale
            non_null = int(metric_series.notna().sum())
            metric_payload = {
                "unit": unit,
                "non_null": non_null,
                "min": None,
                "mean": None,
                "max": None,
            }
            if non_null > 0:
                metric_payload.update(
                    {
                        "min": float(metric_series.min()),
                        "mean": float(metric_series.mean()),
                        "max": float(metric_series.max()),
                    }
                )
            payload[metric] = metric_payload
        return payload


class CSVOutput(Output):
    def handle_result(self, result: AnalysisResult):
        raise NotImplementedError("CSVOutput is not implemented yet.")


class SQLiteOutput(Output):
    def __init__(
        self,
        compact: bool = False,
        group_behavior: bool = False,
        name: str = "",
        root_only: bool = False,
        run_db_path: str = "",
        view_names: List[str] = [],
    ):
        super().__init__(compact, name, root_only, view_names)
        self.run_db_path = run_db_path

    def handle_result(self, result: AnalysisResult):
        raise NotImplementedError("SQLiteOutput is not implemented yet.")

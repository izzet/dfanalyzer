import dask.dataframe as dd
import inflect
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import venn
from dask.base import compute
from dataclasses import asdict, dataclass
from distributed import get_client
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pandas.api.types import is_numeric_dtype
from rich.console import Console
from rich.panel import Panel
from rich.padding import Padding
from rich.table import Table
from rich.terminal_theme import DEFAULT_TERMINAL_THEME
from rich.tree import Tree
from scipy.stats import skew
from typing import Dict, List, Set, Tuple

from .analysis import SCORE_BINS, SCORE_INITIALS, SCORE_NAMES
from .constants import (
    COL_APP_NAME,
    COL_FILE_DIR,
    COL_FILE_NAME,
    COL_FILE_PATTERN,
    COL_NODE_NAME,
    COL_PROC_NAME,
    COL_RANK,
    COL_TIME_RANGE,
    EVENT_ATT_REASONS,
    EVENT_COMP_HLM,
    EVENT_COMP_MAIN_VIEW,
    EVENT_COMP_PERS,
    EVENT_DET_BOT,
    EVENT_READ_TRACES,
    EVENT_SAVE_BOT,
)
from .rules import HUMANIZED_KNOWN_RULES, MAX_REASONS, BottleneckRule, KnownCharacteristics
from .types import (
    AnalysisRuntimeConfig,
    BottleneckOutput,
    Characteristics,
    MainView,
    Metric,
    RawStats,
    Score,
    ScoringPerViewPerMetric,
    ViewKey,
    ViewResultsPerViewPerMetric,
    humanized_view_name,
    view_name,
)
from .utils.dask_utils import flatten_column_names
from .utils.file_utils import ensure_dir


@dataclass
class OutputCharacteristicsType:
    complexity: float
    io_time: float
    job_time: float
    num_apps: int
    num_files: int
    num_nodes: int
    num_ops: int
    num_procs: int
    num_time_periods: int
    per_io_time: float


@dataclass
class OutputCountsType:
    raw_count: int
    hlm_count: int
    main_view_count: int
    avg_perspective_count: Dict[str, int]
    avg_perspective_count_std: Dict[str, float]
    avg_perspective_critical_count: Dict[str, int]
    avg_perspective_critical_count_std: Dict[str, float]
    perspective_skewness: Dict[str, float]
    root_perspective_skewness: Dict[str, float]
    per_records_discarded: Dict[str, float]
    per_records_retained: Dict[str, float]
    num_bottlenecks: Dict[str, int]
    num_metrics: int
    num_perspectives: int
    num_rules: int
    evaluated_records: Dict[str, int]
    perspective_count_tree: Dict[str, Dict[str, int]]
    perspective_critical_count_tree: Dict[str, Dict[str, int]]
    perspective_record_count_tree: Dict[str, Dict[str, int]]
    reasoned_records: Dict[str, int]
    slope_filtered_records: Dict[str, int]


@dataclass
class OutputSeveritiesType:
    critical_count: Dict[str, int]
    critical_tree: Dict[str, Dict[str, int]]
    very_high_count: Dict[str, int]
    very_high_tree: Dict[str, Dict[str, int]]
    high_count: Dict[str, int]
    high_tree: Dict[str, Dict[str, int]]
    medium_count: Dict[str, int]
    medium_tree: Dict[str, Dict[str, int]]
    low_count: Dict[str, int]
    very_low_count: Dict[str, int]
    trivial_count: Dict[str, int]
    none_count: Dict[str, int]
    root_critical_count: Dict[str, int]
    root_very_high_count: Dict[str, int]
    root_high_count: Dict[str, int]
    root_medium_count: Dict[str, int]
    root_low_count: Dict[str, int]
    root_very_low_count: Dict[str, int]
    root_trivial_count: Dict[str, int]
    root_none_count: Dict[str, int]


@dataclass
class OutputThroughputsType:
    bottlenecks: Dict[str, float]
    evaluated_records: Dict[str, float]
    perspectives: Dict[str, float]
    reasoned_records: Dict[str, float]
    rules: Dict[str, float]
    slope_filtered_records: Dict[str, float]


@dataclass
class OutputTimingsType:
    read_traces: Dict[str, float]
    compute_hlm: Dict[str, float]
    compute_main_view: Dict[str, float]
    compute_perspectives: Dict[str, float]
    detect_bottlenecks: Dict[str, float]
    attach_reasons: Dict[str, float]
    save_bottlenecks: Dict[str, float]


@dataclass
class OutputType:
    _bottlenecks: List[List[BottleneckOutput]]
    _characteristics: Characteristics
    _raw_stats: RawStats
    characteristics: OutputCharacteristicsType
    counts: OutputCountsType
    severities: OutputSeveritiesType
    throughputs: OutputThroughputsType
    timings: OutputTimingsType


def _colored_description(description: str, score: str = None):
    if score is None:
        return description
    if score == SCORE_NAMES[0]:
        return description
    elif score == SCORE_NAMES[1]:
        return f"[light_cyan3]{description}"
    elif score == SCORE_NAMES[2]:
        return f"[chartreuse2]{description}"
    elif score == SCORE_NAMES[3]:
        return f"[yellow4]{description}"
    elif score == SCORE_NAMES[4]:
        return f"[yellow3]{description}"
    elif score == SCORE_NAMES[5]:
        return f"[orange3]{description}"
    elif score == SCORE_NAMES[6]:
        return f"[dark_orange3]{description}"
    elif score == SCORE_NAMES[7]:
        return f"[red3]{description}"


def _create_characteristics_table(
    characteristics: Characteristics,
    compact: bool,
    job_time: float,
):
    char_table = Table(box=None, show_header=False)
    char_table.add_column(style="cyan")
    char_table.add_column()

    if compact:
        char_table.add_row('Runtime', f"{job_time:.2f} s")
    else:
        char_table.add_row('Runtime', f"{job_time:.2f} seconds")

    # Add each key-value pair to the table as a row
    for char in characteristics.values():
        if compact or char.detail_list is None or len(char.detail_list) == 0:
            if compact:
                char_table.add_row(char.description, char.compact_desc)
            else:
                char_table.add_row(char.description, char.value_fmt)
        else:
            detail_tree = Tree(char.value_fmt)
            for detail in char.detail_list:
                detail_tree.add(detail)
            char_table.add_row(char.description, detail_tree)

    return char_table


def _create_bottleneck_table(
    bottleneck_rules: Dict[str, BottleneckRule],
    bottlenecks: pd.DataFrame,
    compact: bool,
    max_bottlenecks: int,
    metric: str,
    pluralize: inflect.engine,
):
    view_names = list(bottlenecks['view_name'].unique())

    bot_table = Table(box=None, show_header=False)

    total_bot_count = 0
    total_reason_count = 0

    for view_name in view_names:
        view_bottlenecks = bottlenecks[bottlenecks['view_name'] == view_name]
        bot_count = len(view_bottlenecks)
        total_bot_count = total_bot_count + bot_count
        if bot_count == 0:
            continue
        reason_cols = view_bottlenecks.columns[view_bottlenecks.columns.str.contains('reason')]
        reason_rules = [(reason.split('.')[0], reason) for reason in reason_cols]
        reason_count = sum(
            len(view_bottlenecks[(view_bottlenecks[rule] & view_bottlenecks[reason])]) for rule, reason in reason_rules
        )
        total_reason_count = total_reason_count + reason_count
        view_key = tuple(view_name.split('.'))
        view_tree = Tree(
            (
                f"{humanized_view_name(view_key, '>').replace('Period', '').strip()} View "
                f"({bot_count} {pluralize.plural_noun('bottleneck', bot_count)} "
                f"with {reason_count} {pluralize.plural_noun('reason', reason_count)})"
            )
        )
        for _, bottleneck in view_bottlenecks[:max_bottlenecks].iterrows():
            bot_id = getattr(bottleneck, 'id')
            bot_desc = None
            bot_score = getattr(bottleneck, f"{metric}_score")
            reasons = []
            for rule, rule_impl in bottleneck_rules.items():
                view_type = view_key[-1]

                num_files = int(getattr(bottleneck, 'num_file_name', 0))
                num_processes = int(getattr(bottleneck, 'num_proc_name', 0))
                num_time_periods = int(getattr(bottleneck, 'num_time_range', 0))

                if view_type in [COL_FILE_NAME, COL_FILE_DIR, COL_FILE_PATTERN]:
                    num_files = int(getattr(bottleneck, f"num_{view_type}", 0))
                if view_type in [COL_APP_NAME, COL_NODE_NAME, COL_PROC_NAME, COL_RANK]:
                    num_processes = int(getattr(bottleneck, f"num_{view_type}", 0))

                # TODO move to upper level
                if bot_desc is None:
                    bot_desc = rule_impl.describe_bottleneck(
                        compact=compact,
                        metric=getattr(bottleneck, 'metric'),
                        num_files=num_files,
                        num_ops=int(bottleneck['count']),
                        num_processes=num_processes,
                        num_time_periods=num_time_periods,
                        subject=getattr(bottleneck, 'subject'),
                        time=float(getattr(bottleneck, 'time')),
                        time_overall=float(getattr(bottleneck, 'time_overall')),
                        view_type=view_key[-1],
                    )

                humanized_rule = HUMANIZED_KNOWN_RULES[rule]

                # Check if rule applies
                if getattr(bottleneck, rule, False):
                    num_reasons = len(rule_impl.rule.reasons)

                    # Check if any reason is found
                    if any(getattr(bottleneck, f"{rule}.reason.{i}", False) for i in range(num_reasons)):
                        for i in range(num_reasons):
                            # Check if reason applies
                            if getattr(bottleneck, f"{rule}.reason.{i}", False):
                                reason = rule_impl.describe_reason(
                                    bottleneck=dict(bottleneck),
                                    reason_index=i,
                                )
                                reasons.append(f"[{humanized_rule}] {reason}")
                    else:
                        # TODO give details
                        reasons.append(f"[{humanized_rule}] No reason found, investigation needed!")

            nice_bot_desc = f"[{SCORE_INITIALS[bot_score]}{bot_id}] {bot_desc}"
            bot_tree = Tree(_colored_description(nice_bot_desc, bot_score))
            for reason in reasons:
                bot_tree.add(_colored_description(reason, bot_score))
            view_tree.add(bot_tree)

        if max_bottlenecks > 0 and bot_count > max_bottlenecks:
            remaining_count = bot_count - max_bottlenecks
            view_tree.add(f"({remaining_count} more)")

        bot_table.add_row(view_tree)

    return bot_table, total_bot_count, total_reason_count


class Output(object):
    def __init__(
        self,
        bottleneck_dir: str,
        bottleneck_rules: Dict[str, BottleneckRule],
        characteristics: Characteristics,
        evaluated_views: ScoringPerViewPerMetric,
        main_view: MainView,
        raw_stats: RawStats,
        view_results: ViewResultsPerViewPerMetric,
    ) -> None:
        self.bottleneck_dir = bottleneck_dir
        self.bottleneck_rules = bottleneck_rules
        self.characteristics = characteristics
        self.evaluated_views = evaluated_views
        self.main_view = main_view
        self.pluralize = inflect.engine()
        self.raw_stats = raw_stats
        self.view_results = view_results

    def console(
        self,
        compact=False,
        group_behavior=True,
        max_bottlenecks=3,
        name='',
        root_only=False,
        show_debug=False,
        show_characteristics=True,
        show_header=True,
        view_names: List[str] = [],
    ):
        # TODO metric
        metric = 'iops'

        output = self._create_output_type(group_behavior=group_behavior)

        bottlenecks = output._bottlenecks
        characteristics = output._characteristics
        raw_stats = output._raw_stats

        if len(view_names) > 0:
            bottlenecks = bottlenecks.query('view_name in @view_names', local_dict={'view_names': view_names})
        elif root_only:
            bottlenecks = bottlenecks[bottlenecks['view_depth'] == 1]

        char_table = _create_characteristics_table(
            characteristics=characteristics,
            compact=compact,
            job_time=output.characteristics.job_time,
        )

        char_panel = Panel(
            renderable=char_table,
            title=' '.join([name, 'I/O Characteristics']).strip() if show_header else None,
            subtitle=('[bold]R[/bold]: Read - ' '[bold]W[/bold]: Write - ' '[bold]M[/bold]: Metadata '),
            subtitle_align='left',
            padding=1,
        )

        bot_table, total_bot_count, total_reason_count = _create_bottleneck_table(
            bottleneck_rules=self.bottleneck_rules,
            bottlenecks=bottlenecks,
            compact=compact,
            max_bottlenecks=max_bottlenecks,
            metric=metric,
            pluralize=self.pluralize,
        )

        bot_panel = Panel(
            bot_table,
            title=(
                f"{total_bot_count} I/O {self.pluralize.plural_noun('Bottleneck', total_bot_count)} with "
                f"{total_reason_count} {self.pluralize.plural_noun('Reason', total_reason_count)}"
            )
            if show_header
            else None,
            padding=1,
        )

        console = Console(record=True)

        if show_debug:
            main_view_count = output.counts.main_view_count
            raw_total_count = output.counts.raw_count

            debug_table = Table(box=None, show_header=False)
            debug_table.add_column(style="cyan")
            debug_table.add_column()

            retained_tree = Tree(f"raw: {raw_total_count} records (100%)")
            main_view_tree = retained_tree.add(
                f"aggregated view: {main_view_count} ({main_view_count/raw_total_count*100:.2f}% 100%)"
            )

            for metric in output.counts.perspective_count_tree:
                metric_tree = Tree(
                    (
                        f"{metric}: "
                        f"{output.counts.avg_perspective_count[metric]:.2f}±{output.counts.avg_perspective_count_std[metric]:.2f} "
                        f"({output.counts.avg_perspective_count[metric]/raw_total_count*100:.2f}% "
                        f"{output.counts.avg_perspective_count[metric]/main_view_count*100:.2f}%)"
                        " -S> "
                        f"{output.counts.avg_perspective_critical_count[metric]:.2f}±{output.counts.avg_perspective_critical_count_std[metric]:.2f} "
                        f"({output.counts.avg_perspective_critical_count[metric]/raw_total_count*100:.2f}% "
                        f"{output.counts.avg_perspective_critical_count[metric]/main_view_count*100:.2f}%)"
                    )
                )
                for count_key in output.counts.perspective_count_tree[metric]:
                    view_count = output.counts.perspective_count_tree[metric][count_key]
                    view_critical_count = output.counts.perspective_critical_count_tree[metric][count_key]
                    metric_tree.add(
                        (
                            f"{count_key}: "
                            f"{view_count} ({view_count/raw_total_count*100:.2f}% {view_count/main_view_count*100:.2f}%)"
                            " -S> "
                            f"{view_critical_count} ({view_critical_count/raw_total_count*100:.2f}% {view_critical_count/main_view_count*100:.2f}%)"
                        )
                    )
                main_view_tree.add(metric_tree)

            debug_table.add_row('Retained Records', retained_tree)

            count_table = Table(box=None, show_header=True)
            count_table.add_column()
            count_table.add_column('Count')
            count_table.add_column('Processed Record Count')
            count_table.add_row(
                'Perspectives',
                f"{output.counts.num_perspectives}",
                f"{output.counts.slope_filtered_records[metric]}",
            )
            count_table.add_row(
                'Bottlenecks',
                f"{output.counts.num_bottlenecks[metric]}",
                f"{output.counts.evaluated_records[metric]}",
            )
            count_table.add_row(
                'Rules',
                f"{output.counts.num_rules}",
                f"{output.counts.reasoned_records[metric]}",
            )

            debug_table.add_row('Counts', count_table)

            tput_table = Table(box=None, show_header=True)
            tput_table.add_column()
            tput_table.add_column('Throughput')
            tput_table.add_column('Record Throughput')
            tput_table.add_row(
                'Perspectives',
                f"{output.throughputs.perspectives[metric]:.2f} perspectives/sec",
                f"{output.throughputs.slope_filtered_records[metric]:.2f} records/sec",
            )
            tput_table.add_row(
                'Bottlenecks',
                f"{output.throughputs.bottlenecks[metric]:.2f} bottlenecks/sec",
                f"{output.throughputs.evaluated_records[metric]:.2f} records/sec",
            )
            tput_table.add_row(
                'Rules',
                f"{output.throughputs.rules[metric]:.2f} rules/sec",
                f"{output.throughputs.reasoned_records[metric]:.2f} records/sec",
            )

            debug_table.add_row('Throughputs', tput_table)

            tot_bottlenecks = 0
            for metric in output.counts.num_bottlenecks:
                num_bottlenecks = output.counts.num_bottlenecks[metric]
                tot_bottlenecks = tot_bottlenecks + num_bottlenecks
            severity_tree = Tree(f"total: {tot_bottlenecks} bottlenecks (100%)")
            for metric in output.counts.num_bottlenecks:
                num_bottlenecks = output.counts.num_bottlenecks[metric]
                severity_metric_tree = severity_tree.add(
                    (f"{metric}: " f"{num_bottlenecks} " f"({num_bottlenecks/tot_bottlenecks*100:.2f}% 100%)")
                )
                severity_metric_tree.add(
                    (
                        f"critical: "
                        f"{output.severities.critical_count[metric]} "
                        f"({output.severities.critical_count[metric]/tot_bottlenecks*100:.2f}% "
                        f"{output.severities.critical_count[metric]/num_bottlenecks*100:.2f}%)"
                    )
                )
                severity_metric_tree.add(
                    (
                        f"very high: "
                        f"{output.severities.very_high_count[metric]} "
                        f"({output.severities.very_high_count[metric]/tot_bottlenecks*100:.2f}% "
                        f"{output.severities.very_high_count[metric]/num_bottlenecks*100:.2f}%)"
                    )
                )
                severity_metric_tree.add(
                    (
                        f"high: "
                        f"{output.severities.high_count[metric]} "
                        f"({output.severities.high_count[metric]/tot_bottlenecks*100:.2f}% "
                        f"{output.severities.high_count[metric]/num_bottlenecks*100:.2f}%)"
                    )
                )
                severity_metric_tree.add(
                    (
                        f"medium: "
                        f"{output.severities.medium_count[metric]} "
                        f"({output.severities.medium_count[metric]/tot_bottlenecks*100:.2f}% "
                        f"{output.severities.medium_count[metric]/num_bottlenecks*100:.2f}%)"
                    )
                )
                severity_metric_tree.add(
                    (
                        f"low: "
                        f"{output.severities.low_count[metric]} "
                        f"({output.severities.low_count[metric]/tot_bottlenecks*100:.2f}% "
                        f"{output.severities.low_count[metric]/num_bottlenecks*100:.2f}%)"
                    )
                )
                severity_metric_tree.add(
                    (
                        f"very low: "
                        f"{output.severities.very_low_count[metric]} "
                        f"({output.severities.very_low_count[metric]/tot_bottlenecks*100:.2f}% "
                        f"{output.severities.very_low_count[metric]/num_bottlenecks*100:.2f}%)"
                    )
                )
                severity_metric_tree.add(
                    (
                        f"trivial: "
                        f"{output.severities.trivial_count[metric]} "
                        f"({output.severities.trivial_count[metric]/tot_bottlenecks*100:.2f}% "
                        f"{output.severities.trivial_count[metric]/num_bottlenecks*100:.2f}%)"
                    )
                )
                severity_metric_tree.add(
                    (
                        f"none: "
                        f"{output.severities.none_count[metric]} "
                        f"({output.severities.none_count[metric]/tot_bottlenecks*100:.2f}% "
                        f"{output.severities.none_count[metric]/num_bottlenecks*100:.2f}%)"
                    )
                )

            debug_table.add_row('Severities', severity_tree)

            debug_panel = Panel(debug_table, title='Debug', padding=1)

            if show_characteristics:
                console.print(char_panel, Padding(''), bot_panel, Padding(''), debug_panel)
            else:
                console.print(bot_panel, Padding(''), debug_panel)
        else:
            if show_characteristics:
                console.print(char_panel, Padding(''), bot_panel)
            else:
                console.print(bot_panel)

        run_dir = f"{self.runtime_config.working_dir}"

        console.save_html(f"{run_dir}/output.html", clear=False)
        console.save_text(f"{run_dir}/output.txt", clear=False)
        console.save_svg(f"{run_dir}/output.svg", theme=DEFAULT_TERMINAL_THEME, title="DFAnalyzer", clear=True)

    def csv(self, name: str, max_bottlenecks_per_view_type=3, show_debug=True):
        run_dir = f"{self.runtime_config.working_dir}"
        ensure_dir(run_dir)

        self._create_output_df(name=name).to_csv(f"{run_dir}/run.csv", encoding='utf8')

        # timings_df, timings_raw_df = self._create_timings_df()
        # timings_df.sort_values(['type', 'key']).to_csv(f"{run_dir}/timings.csv", encoding='utf8')
        # timings_df.sort_values(['time_start']).to_csv(f"{run_dir}/timings_ordered.csv", encoding='utf8')
        # timings_raw_df.to_csv(f"{run_dir}/timings_raw.csv", encoding='utf8')

        # TODO
        metric = 'iops'
        view_name = 'time_range'
        view_key = tuple([view_name])
        if view_key in self.view_results[metric]:
            view_result = self.view_results[metric][view_key]
            view_result.view[
                [
                    'count',
                    'count_per',
                    'time',
                    'time_per',
                    f"{metric}_slope",
                ]
            ].compute().to_csv(f"{run_dir}/slope_{view_name}.csv", encoding='utf8')

    def sqlite(self, name: str, run_db_path: str = None):
        run_dir = f"{self.runtime_config.working_dir}"
        ensure_dir(run_dir)

        con = sqlite3.connect(f"{run_dir}/result.db")

        output_df = self._create_output_df(name=name).reset_index()
        output_df['key'] = output_df['type'] + '_' + output_df['value']
        output_df = output_df.drop(columns=['type', 'value'])
        output_sql_df = output_df.set_index('key').T

        if run_db_path is None:
            output_sql_df.to_sql('run', con=con, index_label='key')
        else:
            run_con = sqlite3.connect(run_db_path)
            output_sql_df.to_sql('run', con=run_con, index_label='key', if_exists='append')
            run_con.close()

        # timings_df, timings_raw_df = self._create_timings_df()
        # timings_df.sort_values(['type', 'key']).to_sql('timings', con=con)
        # timings_df.sort_values(['time_start']).to_sql('timings_ordered', con=con)
        # timings_raw_df.to_sql('timings_raw', con=con)

        # TODO
        metric = 'iops'
        view_name = 'time_range'
        view_key = tuple([view_name])
        if view_key in self.view_results[metric]:
            view_result = self.view_results[metric][view_key]
            view_result.view[
                [
                    'count',
                    'count_per',
                    'time',
                    'time_per',
                    f"{metric}_slope",
                ]
            ].compute().to_sql(f"slope_{view_name}", con=con)

        con.close()

    def _create_output_df(self, name: str):
        output = self._create_output_type()
        output_dict = asdict(obj=output)
        for output_key in output_dict.copy():
            if output_key.startswith('_'):
                output_dict.pop(output_key)
        output_df = pd.DataFrame.from_dict(output_dict, orient='index').stack().to_frame()

        dropping_indices = []
        for ix, row in output_df.copy().iterrows():
            if isinstance(row[0], dict):
                dropping_indices.append(ix)
                type_type = ix[0]
                value_type = ix[1]
                nested_df = pd.json_normalize(row[0])
                for _, nested_row in nested_df.iterrows():
                    secondary_ix = None
                    value = None
                    if '_tree' in value_type:
                        for nested_col in nested_df.columns:
                            suffix = value_type.replace('_tree', '')
                            secondary_ix = f"{nested_col.replace('.', '_').replace('>', '_')}_{suffix}"
                            value = nested_row[nested_col]
                            output_df.loc[(type_type, secondary_ix),] = float(value)
                    else:
                        secondary_ix = f"{nested_row.index[0]}__{value_type}"
                        value = nested_row[0]
                        output_df.loc[(type_type, secondary_ix),] = float(value)

        output_df = output_df.drop(index=dropping_indices).rename(columns={output_df.columns[0]: name})

        output_df.index.set_names(['type', 'value'], inplace=True)

        return output_df.sort_index()

    def _read_bottlenecks(self, metric: str, group_behavior=True):
        bottlenecks = dd.read_parquet(self.bottleneck_dir)

        if group_behavior:
            cols = bottlenecks.columns

            agg_dict = {}
            agg_dict['behavior'] = 'count'  # count behaviors
            for col in cols:
                if 'num_' in col:
                    agg_dict[col] = 'sum'
                elif is_numeric_dtype(bottlenecks[col]):
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'first'

            bottlenecks = bottlenecks.groupby(['view_name', f"{metric}_score", 'behavior']).agg(agg_dict).compute()

            # TODO find a better way
            reason_cols = cols[cols.str.contains('reason')]
            rule_cols = [reason.split('.')[0] for reason in reason_cols]

            for col in [*reason_cols, *rule_cols]:
                bottlenecks[col] = bottlenecks[col].astype(bool)

            for col in cols:
                if 'count' in col or 'num_' in col:
                    bottlenecks[col] = bottlenecks[col].astype(int)

        else:
            bottlenecks = bottlenecks.compute()

        bottlenecks['id'] = np.arange(len(bottlenecks)) + 1  # set ids

        return bottlenecks

    def _create_output_type(self, group_behavior: bool) -> OutputType:
        # todo
        metric = 'iops'

        bottlenecks = self._read_bottlenecks(metric=metric, group_behavior=group_behavior)

        (characteristics,) = compute(self.characteristics)
        (raw_stats,) = compute(self.raw_stats)

        complexity = 0
        io_time = 0
        job_time = float(raw_stats.job_time)
        num_apps = 0
        num_files = 0
        num_nodes = 0
        num_ops = 0
        num_procs = 0
        num_time_periods = 0
        for characteristic in characteristics:
            characteristic_value = characteristics[characteristic].value
            if characteristic == KnownCharacteristics.APP_COUNT.value:
                num_apps = int(characteristic_value)
            elif characteristic == KnownCharacteristics.COMPLEXITY.value:
                complexity = float(characteristic_value)
            elif characteristic == KnownCharacteristics.FILE_COUNT.value:
                num_files = int(characteristic_value)
            elif characteristic == KnownCharacteristics.IO_COUNT.value:
                num_ops = int(characteristic_value)
            elif characteristic == KnownCharacteristics.IO_TIME.value:
                io_time = float(characteristic_value)
            elif characteristic == KnownCharacteristics.NODE_COUNT.value:
                num_nodes = int(characteristic_value)
            elif characteristic == KnownCharacteristics.PROC_COUNT.value:
                num_procs = int(characteristic_value)
            elif characteristic == KnownCharacteristics.TIME_PERIOD.value:
                num_time_periods = int(characteristic_value)

        per_io_time = 0
        if job_time > 0:
            per_io_time = io_time / job_time

        output_characteristics = OutputCharacteristicsType(
            complexity=complexity,
            io_time=io_time,
            job_time=job_time,
            num_apps=num_apps,
            num_files=num_files,
            num_nodes=num_nodes,
            num_ops=num_ops,
            num_procs=num_procs,
            num_time_periods=num_time_periods,
            per_io_time=per_io_time,
        )

        main_view_count = len(self.main_view)
        raw_count = int(raw_stats.total_count)

        perspective_count_tree = {}
        perspective_critical_count_tree = {}
        perspective_record_count_tree = {}
        num_metrics = 0
        perspectives = set()
        for metric in self.view_results:
            perspective_count_tree[metric] = {}
            perspective_critical_count_tree[metric] = {}
            perspective_record_count_tree[metric] = {}
            num_metrics = num_metrics + 1
            for view_key, view_result in self.view_results[metric].items():
                count_key = view_name(view_key, '>')
                bot_important_count = view_result.view.reduction(len, sum)
                view_critical_count = view_result.critical_view.reduction(len, sum)
                view_record_count = view_result.records.reduction(len, sum)
                perspective_count_tree[metric][count_key] = bot_important_count
                perspective_critical_count_tree[metric][count_key] = view_critical_count
                perspective_record_count_tree[metric][count_key] = view_record_count
                perspectives.add(view_key)
        num_metrics = num_metrics
        num_perspectives = len(perspectives)
        (
            perspective_count_tree,
            perspective_critical_count_tree,
            perspective_record_count_tree,
        ) = compute(
            perspective_count_tree,
            perspective_critical_count_tree,
            perspective_record_count_tree,
        )
        root_view_type_counts = {}
        for metric in self.view_results:
            root_view_type_counts[metric] = []
            for view_key, view_result in self.view_results[metric].items():
                if len(view_key) == 1:
                    view_critical_count = perspective_critical_count_tree[metric][count_key]
                    root_view_type_counts[metric].append(view_critical_count)

        avg_perspective_count = {}
        avg_perspective_count_std = {}
        avg_perspective_critical_count = {}
        avg_perspective_critical_count_std = {}
        per_records_discarded = {}
        per_records_retained = {}
        perspective_skewness = {}
        root_perspective_skewness = {}
        for metric in perspective_count_tree:
            perspective_counts = [
                perspective_count_tree[metric][count_key] for count_key in perspective_count_tree[metric]
            ]
            perspective_avg = np.average(perspective_counts)
            perspective_std = np.std(perspective_counts)

            perspective_critical_counts = [
                perspective_critical_count_tree[metric][count_key]
                for count_key in perspective_critical_count_tree[metric]
            ]
            perspective_critical_avg = np.average(perspective_critical_counts)
            perspective_critical_std = np.std(perspective_critical_counts)

            avg_perspective_count[metric] = perspective_avg
            avg_perspective_count_std[metric] = perspective_std
            avg_perspective_critical_count[metric] = perspective_critical_avg
            avg_perspective_critical_count_std[metric] = perspective_critical_std

            perspective_critical_per = perspective_critical_avg / raw_count
            per_records_discarded[metric] = 1 - perspective_critical_per
            per_records_retained[metric] = perspective_critical_per

            perspective_skewness[metric] = abs(skew(perspective_counts))
            root_perspective_skewness[metric] = abs(skew(root_view_type_counts[metric]))

        num_rules = len(self.bottleneck_rules)
        view_names = list(bottlenecks['view_name'].unique())

        bot_total_count = {}
        bot_critical_count = {}
        bot_very_high_count = {}
        bot_high_count = {}
        bot_medium_count = {}
        bot_low_count = {}
        bot_very_low_count = {}
        bot_trivial_count = {}
        bot_none_count = {}

        bot_total_count[metric] = len(bottlenecks)
        bot_critical_count[metric] = len(bottlenecks[bottlenecks[f"{metric}_score"] == Score.CRITICAL.value])
        bot_very_high_count[metric] = len(bottlenecks[bottlenecks[f"{metric}_score"] == Score.VERY_HIGH.value])
        bot_high_count[metric] = len(bottlenecks[bottlenecks[f"{metric}_score"] == Score.HIGH.value])
        bot_medium_count[metric] = len(bottlenecks[bottlenecks[f"{metric}_score"] == Score.MEDIUM.value])
        bot_low_count[metric] = len(bottlenecks[bottlenecks[f"{metric}_score"] == Score.LOW.value])
        bot_very_low_count[metric] = len(bottlenecks[bottlenecks[f"{metric}_score"] == Score.VERY_LOW.value])
        bot_trivial_count[metric] = len(bottlenecks[bottlenecks[f"{metric}_score"] == Score.TRIVIAL.value])
        bot_none_count[metric] = len(bottlenecks[bottlenecks[f"{metric}_score"] == Score.NONE.value])

        bot_root_critical_count = {}
        bot_root_very_high_count = {}
        bot_root_high_count = {}
        bot_root_medium_count = {}
        bot_root_low_count = {}
        bot_root_very_low_count = {}
        bot_root_trivial_count = {}
        bot_root_none_count = {}

        bot_root_critical_count[metric] = len(
            bottlenecks[(bottlenecks[f"{metric}_score"] == Score.CRITICAL.value) & (bottlenecks['view_depth'] == 1)]
        )
        bot_root_very_high_count[metric] = len(
            bottlenecks[(bottlenecks[f"{metric}_score"] == Score.VERY_HIGH.value) & (bottlenecks['view_depth'] == 1)]
        )
        bot_root_high_count[metric] = len(
            bottlenecks[(bottlenecks[f"{metric}_score"] == Score.HIGH.value) & (bottlenecks['view_depth'] == 1)]
        )
        bot_root_medium_count[metric] = len(
            bottlenecks[(bottlenecks[f"{metric}_score"] == Score.MEDIUM.value) & (bottlenecks['view_depth'] == 1)]
        )
        bot_root_low_count[metric] = len(
            bottlenecks[(bottlenecks[f"{metric}_score"] == Score.LOW.value) & (bottlenecks['view_depth'] == 1)]
        )
        bot_root_very_low_count[metric] = len(
            bottlenecks[(bottlenecks[f"{metric}_score"] == Score.VERY_LOW.value) & (bottlenecks['view_depth'] == 1)]
        )
        bot_root_trivial_count[metric] = len(
            bottlenecks[(bottlenecks[f"{metric}_score"] == Score.TRIVIAL.value) & (bottlenecks['view_depth'] == 1)]
        )
        bot_root_none_count[metric] = len(
            bottlenecks[(bottlenecks[f"{metric}_score"] == Score.NONE.value) & (bottlenecks['view_depth'] == 1)]
        )

        bot_critical_tree = {}
        bot_very_high_tree = {}
        bot_high_tree = {}
        bot_medium_tree = {}

        bot_critical_tree[metric] = {
            view_name: len(
                bottlenecks[
                    (bottlenecks[f"{metric}_score"] == Score.CRITICAL.value) & (bottlenecks['view_name'] == view_name)
                ]
            )
            for view_name in view_names
        }
        bot_very_high_tree[metric] = {
            view_name: len(
                bottlenecks[
                    (bottlenecks[f"{metric}_score"] == Score.VERY_HIGH.value) & (bottlenecks['view_name'] == view_name)
                ]
            )
            for view_name in view_names
        }
        bot_high_tree[metric] = {
            view_name: len(
                bottlenecks[
                    (bottlenecks[f"{metric}_score"] == Score.HIGH.value) & (bottlenecks['view_name'] == view_name)
                ]
            )
            for view_name in view_names
        }
        bot_medium_tree[metric] = {
            view_name: len(
                bottlenecks[
                    (bottlenecks[f"{metric}_score"] == Score.MEDIUM.value) & (bottlenecks['view_name'] == view_name)
                ]
            )
            for view_name in view_names
        }

        severities = OutputSeveritiesType(
            critical_count=bot_critical_count,
            very_high_count=bot_very_high_count,
            high_count=bot_high_count,
            medium_count=bot_medium_count,
            low_count=bot_low_count,
            very_low_count=bot_very_low_count,
            trivial_count=bot_trivial_count,
            none_count=bot_none_count,
            critical_tree=bot_critical_tree,
            very_high_tree=bot_very_high_tree,
            high_tree=bot_high_tree,
            medium_tree=bot_medium_tree,
            root_critical_count=bot_root_critical_count,
            root_very_high_count=bot_root_very_high_count,
            root_high_count=bot_root_high_count,
            root_medium_count=bot_root_medium_count,
            root_low_count=bot_root_low_count,
            root_very_low_count=bot_root_very_low_count,
            root_trivial_count=bot_root_trivial_count,
            root_none_count=bot_root_none_count,
        )

        elapsed_times = {}
        for _, event in get_client().get_events('elapsed_times'):
            elapsed_times[event['key']] = event['elapsed_time']
        attach_reasons = {}
        compute_hlm = {}
        compute_main_view = {}
        compute_perspectives = {}
        detect_bottlenecks = {}
        read_traces = {}
        save_bottlenecks = {}
        attach_reasons[metric] = float(elapsed_times.get(EVENT_ATT_REASONS, 0))
        compute_hlm[metric] = float(elapsed_times.get(EVENT_COMP_HLM, 0))
        compute_main_view[metric] = float(elapsed_times.get(EVENT_COMP_MAIN_VIEW, 0))
        compute_perspectives[metric] = float(elapsed_times.get(EVENT_COMP_PERS, 0))
        detect_bottlenecks[metric] = float(elapsed_times.get(EVENT_DET_BOT, 0))
        read_traces[metric] = float(elapsed_times.get(EVENT_READ_TRACES, 0))
        save_bottlenecks[metric] = float(elapsed_times.get(EVENT_SAVE_BOT, 0))
        timings = OutputTimingsType(
            attach_reasons=attach_reasons,
            compute_hlm=compute_hlm,
            compute_main_view=compute_main_view,
            compute_perspectives=compute_perspectives,
            detect_bottlenecks=detect_bottlenecks,
            read_traces=read_traces,
            save_bottlenecks=save_bottlenecks,
        )

        evaluated_record_dict = {}
        slope_filtered_record_dict = {}
        for metric in self.evaluated_views:
            evaluated_record_dict[metric] = {}
            slope_filtered_record_dict[metric] = {}
            for view_key in self.evaluated_views[metric]:
                scoring = self.evaluated_views[metric][view_key]
                evaluated_record_dict[metric][view_key] = scoring.critical_view.reduction(len, sum)
                slope_filtered_record_dict[metric][view_key] = scoring.records_index.reduction(len, sum)
        (
            evaluated_record_dict,
            slope_filtered_record_dict,
        ) = compute(
            evaluated_record_dict,
            slope_filtered_record_dict,
        )
        evaluated_records = {}
        reasoned_records = {}
        slope_filtered_records = {}
        for metric in self.evaluated_views:
            evaluated_records[metric] = 0
            reasoned_records[metric] = 0
            slope_filtered_records[metric] = 0
            for view_key in self.evaluated_views[metric]:
                view_evaluated_records = evaluated_record_dict[metric][view_key]
                view_slope_filtered_records = slope_filtered_record_dict[metric][view_key]
                evaluated_records[metric] = evaluated_records[metric] + view_evaluated_records
                reasoned_records[metric] = evaluated_records[metric] * num_rules
                slope_filtered_records[metric] = slope_filtered_records[metric] + view_slope_filtered_records

        counts = OutputCountsType(
            avg_perspective_count=avg_perspective_count,
            avg_perspective_count_std=avg_perspective_count_std,
            avg_perspective_critical_count=avg_perspective_critical_count,
            avg_perspective_critical_count_std=avg_perspective_critical_count_std,
            evaluated_records=evaluated_records,
            hlm_count=main_view_count,
            main_view_count=main_view_count,
            num_bottlenecks=bot_total_count,
            num_metrics=num_metrics,
            num_perspectives=num_perspectives,
            num_rules=num_rules,
            per_records_discarded=per_records_discarded,
            per_records_retained=per_records_retained,
            perspective_count_tree=perspective_count_tree,
            perspective_critical_count_tree=perspective_critical_count_tree,
            perspective_record_count_tree=perspective_record_count_tree,
            perspective_skewness=perspective_skewness,
            raw_count=raw_count,
            reasoned_records=reasoned_records,
            root_perspective_skewness=root_perspective_skewness,
            slope_filtered_records=slope_filtered_records,
        )

        bottlenecks_tput = {}
        perspectives_tput = {}
        reasoned_records_tput = {}
        rules_tput = {}
        slope_filtered_records_tput = {}
        evaluated_records_tput = {}

        for metric in evaluated_records:
            bottlenecks_tput[metric] = bot_total_count[metric] / detect_bottlenecks[metric]
            evaluated_records_tput[metric] = evaluated_records[metric] / detect_bottlenecks[metric]
            perspectives_tput[metric] = num_perspectives / compute_perspectives[metric]
            reasoned_records_tput[metric] = (evaluated_records[metric] * num_rules) / attach_reasons[metric]
            rules_tput[metric] = num_rules / attach_reasons[metric]
            slope_filtered_records_tput[metric] = slope_filtered_records[metric] / compute_perspectives[metric]

        throughputs = OutputThroughputsType(
            bottlenecks=bottlenecks_tput,
            evaluated_records=evaluated_records_tput,
            perspectives=perspectives_tput,
            reasoned_records=reasoned_records_tput,
            rules=rules_tput,
            slope_filtered_records=slope_filtered_records_tput,
        )

        return OutputType(
            _bottlenecks=bottlenecks,
            _characteristics=characteristics,
            _raw_stats=raw_stats,
            characteristics=output_characteristics,
            counts=counts,
            runtime_config=self.runtime_config,
            severities=severities,
            throughputs=throughputs,
            timings=timings,
        )

    def _create_timings_df(self):
        timing_events = get_client().get_events('timings')

        timings = []
        for _, timing in timing_events:
            timings.append(timing)

        raw_df = pd.DataFrame(timings)

        timings_df = raw_df.copy()
        timings_df['key_type'] = timings_df['key'] + '_' + timings_df['type']

        timings_df = (
            timings_df[['key_type', 'key', 'type', 'time', 'size']]
            .groupby(['key_type'])
            .first()
            .reset_index()
            .pivot(index='key', columns='type', values=['size', 'time'])
        )

        timings_df['time', 'elapsed'] = timings_df['time', 'end'] - timings_df['time', 'start']

        timings_df = flatten_column_names(timings_df)

        timings_df['type'] = timings_df.index.map(lambda x: x.split('_')[-1])

        return timings_df, raw_df


class AnalysisResultPlots(object):
    def __init__(
        self,
        main_view: MainView,
        view_results: ViewResultsPerViewPerMetric,
        evaluated_views: ScoringPerViewPerMetric,
    ):
        self.main_view = main_view
        self.view_results = view_results
        self.bottlenecks = evaluated_views  # TODO
        self._cmap = plt.get_cmap('RdYlGn')

    def bottleneck_bar(
        self,
        figsize: Tuple[int, int],
        metrics: List[Metric],
        thresholds: List[float],
        markers: List[str],
        colors: List[str],
        labels: List[str] = [],
        marker_size=72,
    ):
        proc_names = list(self.view_results['time'][(COL_PROC_NAME,)].view.index)
        proc_names.sort(key=lambda x: int(x.split('#')[2]))  # order by rank

        dur_data = (
            self.view_results['time'][(COL_PROC_NAME,)]
            .records.groupby([COL_PROC_NAME, COL_TIME_RANGE])['time']
            .compute()
        )

        fig, ax = plt.subplots(figsize=figsize)

        bar_data = []
        bar_h = 1

        for y, proc_name in enumerate(proc_names):
            try:
                bar_args = dict(
                    xranges=dur_data.loc[proc_name].to_dict().items(),
                    yrange=(y, bar_h),
                    facecolors='C0',
                    alpha=0.8,
                )

                ax.broken_barh(**bar_args)

                bar_data.append(bar_args)
            except KeyError:
                continue

        scatter_data = {}

        for m, metric in enumerate(metrics):
            scatter_data[metric] = []

            data = self.bottlenecks[metric][(COL_PROC_NAME,)]['mid_level_view'].compute()

            for y, proc_name in enumerate(proc_names):
                try:
                    for time_range, threshold in data.loc[proc_name][f"{metric}_th"].to_dict().items():
                        # print(proc_name, y, time_range, threshold)
                        if threshold >= thresholds[m]:
                            scatter_args = dict(
                                x=time_range,
                                y=y + (bar_h / 2),
                                s=marker_size,
                                c=colors[m],
                                marker=markers[m],
                                alpha=0.6,
                            )
                            ax.scatter(**scatter_args)

                            scatter_data[metric].append(scatter_args)

                except KeyError:
                    continue

        # len(bot_dur_proc_ml.index.get_level_values(0).unique()))
        ax.set_ylim(0, len(proc_names))
        ax.set_xlim(0, max(dur_data.index.get_level_values(1)))
        ax.set_ylabel('Ranks')
        ax.set_xlabel('Job Time')

        legend_handles = [Line2D([0], [0], color='C0', label='I/O Op')]
        for m, metric in enumerate(metrics):
            legend_handles.append(
                Line2D(
                    xdata=[0],
                    ydata=[0],
                    color='w',
                    label=metric if len(labels) == 0 else labels[m],
                    marker=markers[m],
                    markerfacecolor=colors[m],
                    markersize=marker_size / 8,
                )
            )

        plt.legend(handles=legend_handles, loc='upper right')

        return fig, ax, bar_data, scatter_data

    def bottleneck_timeline(self, metric: Metric):
        return self._bottleneck_timeline_plot(metric=metric, figsize=(10, 5), title=metric)

    def bottleneck_timeline3(
        self,
        metric1: Metric,
        metric2: Metric,
        metric3: Metric,
        figsize: Tuple[int, int],
        label1: str = None,
        label2: str = None,
        label3: str = None,
        threshold=0.0,
        sample_count=0,
    ):
        # plt.style.use('seaborn-poster')
        fig = plt.figure()

        ax1_line, _ = self._bottleneck_timeline_plot(
            metric=metric1,
            figsize=figsize,
            threshold=threshold,
            yaxis_formatter=self._ticker_for_metric(metric1),
            yaxis_label=label1,
            sample_count=sample_count,
            scatter_zorder=4,
        )
        ax1_line.set_xlabel('Timeline')

        ax2 = ax1_line.twinx()
        ax2.spines['right'].set_position(('axes', 1.0))
        self._bottleneck_timeline_plot(
            metric=metric2,
            figsize=figsize,
            ax=ax2,
            color='C4',
            marker='x',
            threshold=threshold,
            yaxis_formatter=self._ticker_for_metric(metric2),
            yaxis_label=label2,
            sample_count=sample_count,
            scatter_zorder=5,
        )

        ax3 = ax1_line.twinx()
        ax3.spines['right'].set_position(('axes', 1.25))
        self._bottleneck_timeline_plot(
            metric=metric3,
            figsize=figsize,
            ax=ax3,
            color='C5',
            marker='v',
            threshold=threshold,
            yaxis_formatter=self._ticker_for_metric(metric3),
            yaxis_label=label3,
            sample_count=sample_count,
            scatter_zorder=6,
        )

        plt.tight_layout()

        legend_handles = [
            Line2D([0], [0], color='C0', label=label1, lw=2, marker='o'),
            Line2D([0], [0], color='C4', label=label2, lw=2, marker='X'),
            Line2D([0], [0], color='C5', label=label3, lw=2, marker='v'),
        ]

        plt.legend(handles=legend_handles, loc='upper right')

        # Add the colorbar separately using the RdYlGn colormap
        # You can adjust the position and size of the colorbar as desired
        cmap = plt.get_cmap('RdYlGn')  # Choose the RdYlGn colormap
        norm = plt.Normalize(vmin=0, vmax=1)  # Normalize the data
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)  # Create the mappable
        mappable.set_array(SCORE_BINS['time'])  # Set the data for the colorbar

        # Position the colorbar within the figure
        # Left, bottom, width, and height are in fractions of the figure size (0 to 1)
        # You can adjust these values
        colorbar_ax = fig.add_axes([0.68, 0.1, 0.2, 0.03])
        colorbar = plt.colorbar(mappable, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_ticklabels(['critical', 'medium', 'trivial'])
        # Adjust the font size as needed
        colorbar.ax.tick_params(labelsize=12, pad=2)

        # Add a label to the colorbar
        # colorbar.set_label('Colorbar Label')
        colorbar_label = 'Bottleneck Severity'
        # Position the label at the top of the colorbar
        colorbar.ax.xaxis.set_label_position('top')
        # Adjust font size and labelpad as needed
        colorbar.ax.set_xlabel(colorbar_label, fontsize=12, labelpad=4)

        return fig

    def _bottleneck_timeline_plot(
        self,
        metric: Metric,
        figsize: Tuple[int, int],
        threshold: float = 0,
        ax: Axes = None,
        title: str = None,
        color='C0',
        marker='o',
        marker_size=96,
        yaxis_formatter: ticker.Formatter = None,
        yaxis_label: str = None,
        sample_count=0,
        scatter_zorder=0,
    ):
        bott = self.bottlenecks[metric][(COL_TIME_RANGE,)].bottlenecks
        metric_col = next(col for col in bott.columns if metric in col)
        data = bott.compute()
        ax_line = data[metric_col].plot(ax=ax, color=color, figsize=figsize, title=title, alpha=0.8)
        if yaxis_formatter is not None:
            ax_line.yaxis.set_major_formatter(yaxis_formatter)
        if yaxis_label is not None:
            ax_line.yaxis.set_label(yaxis_label)
        filtered_data = data.query(f"{metric}_th >= {threshold}").reset_index()
        if sample_count > 0:
            filtered_data = filtered_data.sort_values(f"{metric}_th", ascending=False).head(sample_count)
        colors = np.vectorize(self._color_map)(filtered_data[f"{metric}_th"])
        ax_scatter = filtered_data.plot.scatter(
            ax=ax_line,
            x=COL_TIME_RANGE,
            y=metric_col,
            c=colors,
            cmap=self._cmap,
            marker=marker,
            s=marker_size,
            zorder=scatter_zorder,
        )
        ax_scatter.set_ylabel(yaxis_label)
        return ax_line, ax_scatter

    def metric_relations2(
        self,
        view_key: ViewKey,
        metric1: Metric,
        metric2: Metric,
        label1: str = None,
        label2: str = None,
    ):
        return self._metric_relations(
            view_key=view_key,
            metrics=[metric1, metric2],
            labels=[label1, label2],
        )

    def metric_relations3(
        self,
        view_key: ViewKey,
        metric1: Metric,
        metric2: Metric,
        metric3: Metric,
        label1: str = None,
        label2: str = None,
        label3: str = None,
    ):
        return self._metric_relations(
            view_key=view_key,
            metrics=[metric1, metric2, metric3],
            labels=[label1, label2, label3],
        )

    def _metric_relations(self, view_key: ViewKey, metrics: List[Metric], labels: List[str]):
        sets = [set(self.view_results[metric][view_key].records['id'].unique().compute()) for metric in metrics]
        labels = self._venn_labels(sets, labels, metrics)
        fig, ax = venn.venn3(labels, names=metrics, figsize=(5, 5))
        # ax.get_legend().remove()
        # fig.tight_layout()
        return fig, ax

    def slope(
        self,
        metric: Metric,
        view_keys: List[ViewKey],
        legends: List[str] = [],
        threshold: int = 45,
        ax: Axes = None,
        xlabel: str = None,
        ylabel: str = None,
        figsize: Tuple[int, int] = None,
        color: str = None,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        legend_handles = []
        x_col = f"{metric}_per_rev_cs"
        y_col = 'count_cs_per_rev'
        for i, view_key in enumerate(view_keys):
            view_result = self.view_results[metric][view_key]
            view = view_result.view.compute()
            color = f"C{i}" if color is None else color
            self._plot_slope(
                ax=ax,
                color=color,
                metric=metric,
                threshold=threshold,
                view=view,
                x_col=x_col,
                y_col=y_col,
            )
            if len(legends) > 0:
                legend_handles.append(Line2D([0], [0], color=color, label=legends[i]))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if len(legend_handles) > 0:
            ax.legend(handles=legend_handles)
        return ax, view

    @staticmethod
    def _plot_slope(
        ax: Axes,
        color: str,
        metric: Metric,
        threshold: int,
        view: pd.DataFrame,
        x_col: str,
        y_col: str,
    ):
        slope_cond = view[f"{metric}_slope"] < threshold
        view.loc[slope_cond, f"{x_col}_line"] = view[x_col]
        line = view[f"{x_col}_line"].to_numpy()
        x = view[x_col].to_numpy()
        y = view[y_col].to_numpy()
        last_non_nan_index = np.where(~np.isnan(line))[0][-1]
        dotted = np.copy(line)
        if np.all(np.isnan(line[last_non_nan_index + 1 :])):
            # complete dotted line if all values after last non-nan are nan
            dotted[last_non_nan_index + 1 :] = x[last_non_nan_index + 1 :]
        mask = np.isfinite(dotted)
        ax.plot(dotted[mask], y[mask], c=color, ls=':')
        ax.plot(line, y, c=color)

    def view_relations2(
        self,
        metric: Metric,
        view_key1: ViewKey,
        view_key2: ViewKey,
        label1: str = None,
        label2: str = None,
    ):
        return self._view_relations(
            metric=metric,
            view_keys=[view_key1, view_key2],
            labels=[label1, label2],
        )

    def view_relations3(
        self,
        metric: Metric,
        view_key1: ViewKey,
        view_key2: ViewKey,
        view_key3: ViewKey,
        label1: str = None,
        label2: str = None,
        label3: str = None,
    ):
        return self._view_relations(
            metric=metric,
            view_keys=[view_key1, view_key2, view_key3],
            labels=[label1, label2, label3],
        )

    def _view_relations(self, metric: Metric, view_keys: List[ViewKey], labels: List[str]):
        names = [view_name(view_key).replace('_', '\_') for view_key in view_keys]
        sets = [set(self.view_results[metric][view_key].records['id'].unique().compute()) for view_key in view_keys]
        labels = self._venn_labels(sets, labels, names)
        fig, ax = venn.venn3(labels, names=names, figsize=(5, 5))
        # ax.get_legend().remove()
        # fig.tight_layout()
        return fig, ax

    @staticmethod
    def _color_map(threshold: float):
        if threshold >= 0.9:
            return 'red'
        elif threshold >= 0.75:
            return 'darkorange'
        elif threshold >= 0.5:
            return 'orange'
        elif threshold >= 0.25:
            return 'gold'
        elif threshold >= 0.1:
            return 'yellow'
        elif threshold >= 0.01:
            return 'yellowgreen'
        elif threshold >= 0.001:
            return 'limegreen'
        else:
            return 'green'

    def _ticker_for_metric(self, metric: Metric):
        if metric == 'bw':
            return self._ticker_bw_formatter
        elif metric == 'duration':
            return ticker.StrMethodFormatter('{x:.1f}')
        elif metric == 'iops':
            return self._ticker_human_formatter
        return None

    @ticker.FuncFormatter
    def _ticker_human_formatter(x, pos):
        num = float('{:.3g}'.format(x))
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

    @ticker.FuncFormatter
    def _ticker_bw_formatter(x, pos):
        return f"{math.ceil(x / 1024.0 ** 3)}"

    @staticmethod
    def _venn_labels(sets: List[Set], labels: List[str], names: List[str]):
        fixed_labels = {}
        for label, value in venn.get_labels(sets).items():
            label_value = int(value)
            if label_value > 1_000_000:
                fixed_labels[label] = f"{label_value // 1_000_000}M"
            if label_value > 1_000:
                fixed_labels[label] = f"{label_value // 1_000}K"
            else:
                fixed_labels[label] = value

        pos_arr = ['10', '01'] if len(sets) == 2 else ['100', '010', '001']
        for index, pos in enumerate(pos_arr):
            bold_label = names[index] if labels[index] == None else labels[index]
            fixed_labels[pos] = f"{fixed_labels[pos]}\n" + rf"$\bf{{{bold_label}}}$"
        return fixed_labels


class AnalysisResult(object):
    def __init__(
        self,
        bottleneck_dir: str,
        bottleneck_rules: Dict[str, BottleneckRule],
        characteristics: Characteristics,
        evaluated_views: ScoringPerViewPerMetric,
        main_view: MainView,
        metric_boundaries,
        raw_stats: RawStats,
        view_results: ViewResultsPerViewPerMetric,
    ):
        self.bottleneck_dir = bottleneck_dir
        self.bottleneck_rules = bottleneck_rules
        self.characteristics = characteristics
        self.evaluated_views = evaluated_views
        self.main_view = main_view
        self.metric_boundaries = metric_boundaries
        self.raw_stats = raw_stats
        self.view_results = view_results

        self.output = Output(
            bottleneck_dir=bottleneck_dir,
            bottleneck_rules=bottleneck_rules,
            characteristics=characteristics,
            evaluated_views=evaluated_views,
            main_view=main_view,
            raw_stats=raw_stats,
            view_results=view_results,
        )

        self.plots = AnalysisResultPlots(
            evaluated_views=evaluated_views,
            main_view=main_view,
            view_results=view_results,
        )

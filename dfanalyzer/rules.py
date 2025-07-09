import abc
import dask.dataframe as dd
import functools as ft
import inflect
import numpy as np
import pandas as pd
from dask.delayed import Delayed
from dask.utils import format_bytes
from dataclasses import dataclass
from enum import Enum
from jinja2 import Environment
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Dict, List, Union

# from .analysis import compute_time_boundaries
from .analysis_utils import set_file_dir, set_file_pattern, set_proc_name_parts
from .constants import (
    ACC_PAT_SUFFIXES,
    COL_APP_NAME,
    COL_COUNT,
    COL_FILE_DIR,
    COL_FILE_NAME,
    COL_FILE_PATTERN,
    COL_NODE_NAME,
    COL_PROC_NAME,
    COL_RANK,
    COL_TIME_RANGE,
    COMPACT_IO_TYPES,
    HUMANIZED_COLS,
    HUMANIZED_VIEW_TYPES,
    IO_TYPES,    
    AccessPattern,
    Layer,
)
from .types import (
    Metric,
    RawStats,
    Rule,
    RuleReason,
    RuleResult,
    RuleResultReason,
    ViewKey,
)
from .utils.collection_utils import get_intervals, join_with_and
from .utils.common_utils import format_number, numerize


MAX_REASONS = 5
METADATA_ACCESS_RATIO_THRESHOLD = 0.5
SCORING_ORDER = dict(
    app_name=('app_name', 'time_range', 'file_name'),
    file_dir=('file_dir', 'proc_name', 'time_range'),
    file_name=('file_name', 'proc_name', 'time_range'),
    file_pattern=('file_pattern', 'proc_name', 'time_range'),
    node_name=('node_name', 'time_range', 'file_name'),
    proc_name=('proc_name', 'time_range', 'file_name'),
    rank=('rank', 'time_range', 'file_name'),
    time_range=('time_range', 'proc_name', 'file_name'),
)


jinja_env = Environment()
jinja_env.filters['format_bytes'] = format_bytes
jinja_env.filters['format_number'] = format_number


class KnownCharacteristics(Enum):
    ACCESS_PATTERN = 'access_pattern'
    APP_COUNT = 'app_count'
    APP_TIME = 'app_time'
    COMPLEXITY = 'complexity'
    FILE_COUNT = 'file_count'
    IO_COUNT = 'io_count'
    IO_SIZE = 'io_size'
    IO_TIME = 'io_time'
    NODE_COUNT = 'node_count'
    PROC_COUNT = 'proc_count'
    READ_XFER_SIZE = 'read_xfer_size'
    TIME_PERIOD = 'time_period'
    WRITE_XFER_SIZE = 'write_xfer_size'


class KnownRules(Enum):
    EXCESSIVE_METADATA_ACCESS = 'excessive_metadata_access'
    OPERATION_IMBALANCE = 'operation_imbalance'
    RANDOM_OPERATIONS = 'random_operations'
    SIZE_IMBALANCE = 'size_imbalance'
    SMALL_READS = 'small_reads'
    SMALL_WRITES = 'small_writes'
    UNOVERLAPPED_IO = 'unoverlapped_io'


KNOWN_RULES = {
    KnownRules.EXCESSIVE_METADATA_ACCESS.value: Rule(
        name='Excessive metadata access',
        condition='(metadata_time / time) >= 0.5',
        reasons=[
            RuleReason(
                condition='(open_time > close_time) & (open_time > seek_time)',
                message=(
                    'Overall {{ "%.2f" | format((metadata_time / time) * 100) }}% ({{ "%.2f" | format(metadata_time) }} '
                    'seconds) of I/O time is spent on metadata access, specifically {{ "%.2f" | format((open_time / time) * 100) }}% '
                    '({{ "%.2f" | format(open_time) }} seconds) on the "open" operation.'
                ),
            ),
            RuleReason(
                condition='(close_time > open_time) & (close_time > seek_time)',
                message=(
                    'Overall {{ "%.2f" | format((metadata_time / time) * 100) }}% ({{ "%.2f" | format(metadata_time) }} '
                    'seconds) of I/O time is spent on metadata access, specifically {{ "%.2f" | format((open_time / time) * 100) }}% '
                    '({{ "%.2f" | format(open_time) }} seconds) on the "close" operation.'
                ),
            ),
            RuleReason(
                condition='(seek_time > open_time) & (seek_time > close_time)',
                message=(
                    'Overall {{ "%.2f" | format((metadata_time / time) * 100) }}% ({{ "%.2f" | format(metadata_time) }} '
                    'seconds) of I/O time is spent on metadata access, specifically {{ "%.2f" | format((open_time / time) * 100) }}% '
                    '({{ "%.2f" | format(open_time) }} seconds) on the "seek" operation.'
                ),
            ),
        ],
    ),
    KnownRules.OPERATION_IMBALANCE.value: Rule(
        name='Operation imbalance',
        condition='(abs(write_count - read_count) / count) > 0.1',
        reasons=[
            RuleReason(
                condition='read_count > write_count',
                message=(
                    '"read" operations are {{ "%.2f" | format((read_count / count) * 100) }}% '
                    '({{ read_count | format_number }} operations) of total I/O operations.'
                ),
            ),
            RuleReason(
                condition='write_count > read_count',
                message=(
                    '"write" operations are {{ "%.2f" | format((write_count / count) * 100) }}% '
                    '({{ write_count | format_number }} operations) of total I/O operations.'
                ),
            ),
        ],
    ),
    KnownRules.RANDOM_OPERATIONS.value: Rule(
        name='Random operations',
        condition='random_count / count > 0.5',
        reasons=[
            RuleReason(
                condition='random_count / count > 0.5',
                message=(
                    'Issued high number of random operations, specifically {{ "%.2f" | format((random_count / count) * 100) }}% '
                    '({{ random_count | format_number }} operations) of total I/O operations.'
                ),
            ),
        ],
    ),
    KnownRules.SIZE_IMBALANCE.value: Rule(
        name='Size imbalance',
        condition='size > 0 & (abs(write_size - read_size) / size) > 0.1',
        reasons=[
            RuleReason(
                condition='read_size > write_size',
                message=(
                    '"read" size is {{ "%.2f" | format((read_size / size) * 100) }}% '
                    '({{ read_size | format_bytes }}) of total I/O size.'
                ),
            ),
            RuleReason(
                condition='write_size > read_size',
                message=(
                    '"write" size is {{ "%.2f" | format((write_size / size) * 100) }}% '
                    '({{ write_size | format_bytes }}) of total I/O size.'
                ),
            ),
        ],
    ),
    KnownRules.SMALL_READS.value: Rule(
        name='Small reads',
        condition='(read_time / time) > 0.5 & (read_size / count) < 1048576',
        reasons=[
            RuleReason(
                condition='(read_time / time) > 0.5',
                message=(
                    '"read" time is {{ "%.2f" | format((read_time / time) * 100) }}% '
                    '({{ "%.2f" | format(read_time) }} seconds) of I/O time.'
                ),
            ),
            RuleReason(
                condition='(read_size / count) < 1048576',
                message=(
                    'Average "read"s are {{ (read_size / count) | format_bytes }}, '
                    'which is smaller than {{ 1048576 | format_bytes }}.'
                ),
            ),
        ],
    ),
    KnownRules.SMALL_WRITES.value: Rule(
        name='Small writes',
        condition='(write_time / time) > 0.5 & (write_size / count) < 1048576',
        reasons=[
            RuleReason(
                condition='(write_time / time) > 0.5',
                message=(
                    '"write" time is {{ "%.2f" | format((write_time / time) * 100) }}% '
                    '({{ "%.2f" | format(write_time) }} seconds) of I/O time.'
                ),
            ),
            RuleReason(
                condition='(write_size / count) < 1048576',
                message=(
                    'Average "write"s are {{ (write_size / count) | format_bytes }}, '
                    'which is smaller than {{ 1048576 | format_bytes }}.'
                ),
            ),
        ],
    ),
    KnownRules.UNOVERLAPPED_IO.value: Rule(
        name='Unoverlapped I/O',
        condition='(u_io_time / time) > 0.5',
        layers=[Layer.APP],
        reasons=[
            RuleReason(
                condition='(u_io_time / time) > 0.5',
                message=(
                    'Unoverlapped I/O time is {{ "%.2f" | format((u_io_time / time) * 100) }}% '
                    '({{ "%.2f" | format(u_io_time) }} seconds) of I/O time.'
                ),
            ),
        ],
    ),
}

HUMANIZED_KNOWN_RULES = {rule: KNOWN_RULES[rule].name for rule in KNOWN_RULES}


class RuleHandler(abc.ABC):
    def __init__(self, rule_key: str) -> None:
        super().__init__()
        self.pluralize = inflect.engine()
        self.rule_key = rule_key


@dataclass
class BottleneckRule:
    rule_key: str
    rule: Rule
    verbose: bool = False


# class BottleneckRule(RuleHandler):
#     def __init__(self, rule_key: str, rule: Rule, verbose: bool = False) -> None:
#         super().__init__(rule_key=rule_key)
#         self.rule = rule
#         self.verbose = verbose

#     def define_tasks(
#         self,
#         metric: Metric,
#         metric_boundary: dd.core.Scalar,
#         scoring_result: ScoringResult,
#         view_key: ViewKey,
#     ) -> Dict[str, Delayed]:
#         view_type = view_key[-1]

#         bottlenecks = scoring_result.scored_view.query(self.rule.condition)
#         bottlenecks['time_overall'] = bottlenecks['time'] / metric_boundary

#         details = (
#             scoring_result.records_index.to_frame()
#             .reset_index(drop=True)
#             .query(
#                 f"{view_type} in @indices", local_dict={'indices': bottlenecks.index}
#             )
#         )

#         tasks = {}
#         tasks['bottlenecks'] = bottlenecks
#         tasks['details'] = details

#         for i, reason in enumerate(self.rule.reasons):
#             tasks[f"reason{i}"] = bottlenecks.eval(reason.condition)

#         return tasks

#     def handle_task_results(
#         self,
#         metric: Metric,
#         view_key: ViewKey,
#         result: Dict[str, Union[str, int, pd.DataFrame, pd.Series, pd.Index]],
#     ) -> List[RuleResult]:
#         # t0 = time.perf_counter()

#         view_type = view_key[-1]

#         bottlenecks = result['bottlenecks']

#         if len(bottlenecks) == 0:
#             return []

#         details = result['details'].to_frame(index=False)
#         metric_boundary = result['metric_boundary']

#         files = {}
#         processes = {}
#         time_periods = {}

#         num_files = {}
#         num_ops = bottlenecks[COL_COUNT].to_dict()
#         num_processes = {}
#         num_time_periods = {}

#         # print('handle_task_results t0', time.perf_counter() - t0)

#         # Logical view type fix
#         if view_type == COL_FILE_DIR:
#             details = set_file_dir(df=details.set_index(COL_FILE_NAME))
#         elif view_type == COL_FILE_PATTERN:
#             details = set_file_pattern(df=details.set_index(COL_FILE_NAME))
#         elif view_type in [COL_APP_NAME, COL_NODE_NAME, COL_RANK]:
#             details = set_proc_name_parts(df=details.set_index(COL_PROC_NAME))

#         if self.verbose:
#             for col in details.columns:
#                 if col in [COL_FILE_NAME, COL_FILE_DIR, COL_FILE_PATTERN]:
#                     files = details.groupby(view_type)[col].unique().to_dict()
#                     num_files = {f: len(files[f]) for f in files}
#                 if col in [COL_APP_NAME, COL_NODE_NAME, COL_PROC_NAME, COL_RANK]:
#                     processes = details.groupby(view_type)[col].unique().to_dict()
#                     num_processes = {p: len(processes[p]) for p in processes}
#                 if col == COL_TIME_RANGE:
#                     time_periods = details.groupby(view_type)[col].unique().to_dict()
#                     num_time_periods = {t: len(time_periods[t]) for t in time_periods}
#         else:
#             for col in details.columns:
#                 if col in [COL_FILE_NAME, COL_FILE_DIR, COL_FILE_PATTERN]:
#                     num_files = details.groupby(view_type)[col].nunique().to_dict()
#                 if col in [COL_APP_NAME, COL_NODE_NAME, COL_PROC_NAME, COL_RANK]:
#                     num_processes = details.groupby(view_type)[col].nunique().to_dict()
#                 if col == COL_TIME_RANGE:
#                     num_time_periods = (
#                         details.groupby(view_type)[col].nunique().to_dict()
#                     )

#         # print('handle_task_results t1', time.perf_counter() - t0)

#         reasoning = {}
#         reasoning_templates = {}
#         for i, reason in enumerate(self.rule.reasons):
#             reasoning[i] = result[f"reason{i}"].to_dict()
#             reasoning_templates[i] = jinja_env.from_string(reason.message)

#         results = []

#         for row in bottlenecks.itertuples():
#             bot_files = list(files.get(row.Index, []))
#             bot_processes = list(processes.get(row.Index, []))
#             bot_time_periods = list(time_periods.get(row.Index, []))

#             bot_num_files = num_files.get(row.Index, 0)
#             bot_num_ops = num_ops.get(row.Index, 0)
#             bot_num_processes = num_processes.get(row.Index, 0)
#             bot_num_time_periods = num_time_periods.get(row.Index, 0)

#             description = self.describe_bottleneck(
#                 files=bot_files,
#                 subject=row.Index,
#                 metric=metric,
#                 metric_boundary=metric_boundary,
#                 num_files=bot_num_files,
#                 num_ops=bot_num_ops,
#                 num_processes=bot_num_processes,
#                 num_time_periods=bot_num_time_periods,
#                 processes=bot_processes,
#                 row=row,
#                 time_periods=bot_time_periods,
#                 view_type=view_type,
#             )

#             row_dict = row._asdict()

#             if self.verbose:
#                 row_dict['files'] = bot_files
#                 row_dict['processes'] = bot_processes
#                 row_dict['time_periods'] = list(map(int, bot_time_periods))
#                 # print(row_dict)

#             row_dict['num_files'] = bot_num_files
#             row_dict['num_ops'] = bot_num_ops
#             row_dict['num_processes'] = bot_num_processes
#             row_dict['num_time_periods'] = bot_num_time_periods

#             # Fix index
#             row_dict[view_type] = row_dict['Index']
#             del row_dict['Index']

#             reasons = []
#             for i, reason in enumerate(self.rule.reasons):
#                 if reasoning[i][row.Index]:
#                     reasons.append(
#                         RuleResultReason(
#                             description=reasoning_templates[i].render(row_dict).strip()
#                         )
#                     )

#             if len(reasons) == 0:
#                 reasons.append(
#                     RuleResultReason(
#                         description='No reason found, further investigation needed.'
#                     )
#                 )

#             result = RuleResult(
#                 compact_desc=None,
#                 description=description,
#                 detail_list=None,
#                 extra_data=row_dict,
#                 object_hash=hash(
#                     '_'.join(
#                         [
#                             '{:,.6f}'.format(row_dict['time']),
#                             f"{row_dict['num_files']}",
#                             f"{row_dict['num_processes']}",
#                             f"{row_dict['num_time_periods']}",
#                         ]
#                     )
#                 ),
#                 reasons=reasons,
#                 value=row_dict['time'],
#                 value_fmt='{:,.6f}'.format(row_dict['time']),
#             )

#             results.append(result)

#         # print('handle_task_results t3', time.perf_counter() - t0, len(results))

#         return results

#     def describe_bottleneck(
#         self,
#         metric: Metric,
#         num_files: int,
#         num_ops: int,
#         num_processes: int,
#         num_time_periods: int,
#         subject: Union[str, int],
#         time: float,
#         time_overall: float,
#         view_type: str,
#         files=[],
#         processes=[],
#         time_periods=[],
#         compact=False,
#     ) -> str:
#         if num_files > 0 and num_processes > 0 and num_time_periods > 0:
#             nice_view_type = HUMANIZED_VIEW_TYPES[COL_PROC_NAME].lower()
#             accessor_name = ' '
#             accessed = HUMANIZED_VIEW_TYPES[COL_FILE_NAME].lower()
#             accessed_name = ' '
#             if view_type in [COL_FILE_NAME, COL_FILE_DIR, COL_FILE_PATTERN]:
#                 accessed = HUMANIZED_VIEW_TYPES[view_type].lower()
#                 if num_files == 1:
#                     if compact:
#                         accessed_name = f" ({Path(subject).name}) "
#                     else:
#                         accessed_name = f" ({subject}) "
#             if view_type in [COL_APP_NAME, COL_NODE_NAME, COL_PROC_NAME, COL_RANK]:
#                 nice_view_type = HUMANIZED_VIEW_TYPES[view_type].lower()
#                 if num_processes == 1:
#                     accessor_name = f" ({subject}) "

#             accessor_noun = self.pluralize.plural_noun(nice_view_type, num_processes)
#             accessor_verb = self.pluralize.plural_verb('accesses', num_processes)
#             accessed_noun = self.pluralize.plural_noun(accessed, num_files)
#             time_period_name = f" ({subject}) " if view_type == COL_TIME_RANGE else ' '
#             time_period_noun = self.pluralize.plural_noun(
#                 'time period', num_time_periods
#             )

#             if self.verbose:
#                 time_intervals = get_intervals(values=time_periods)

#                 description = (
#                     f"{self.pluralize.join(processes)} {accessor_noun} {accessor_verb} "
#                     f"{accessed_noun} {self.pluralize.join(files)} "
#                     f"during the {join_with_and(values=time_intervals)}th {time_period_noun} "
#                     f"and {self.pluralize.plural_verb('has', num_processes)} an I/O time of {time:.2f} seconds which is "
#                     f"{time_overall*100:.2f}% of overall I/O time of the workload."
#                 )
#             else:
#                 # 32 processes access 1 file pattern within 6 time periods and have an I/O time of 2.92 seconds which
#                 # is 70.89% of overall I/O time of the workload.
#                 description = (
#                     f"{num_processes:,} {accessor_noun}{accessor_name}{accessor_verb} "
#                     f"{num_files:,} {accessed_noun}{accessed_name}"
#                     f"within {num_time_periods:,} {time_period_noun}{time_period_name}"
#                     f"across {num_ops:,} I/O {self.pluralize.plural_noun('operation', num_ops)} "
#                     f"and {self.pluralize.plural_verb('has', num_processes)} an I/O time of {time:.2f} seconds which is "
#                     f"{time_overall*100:.2f}% of overall I/O time of the workload."
#                 )

#         else:
#             nice_subject = subject
#             nice_view_type = HUMANIZED_VIEW_TYPES[view_type].lower()

#             count = 1
#             if view_type in [COL_FILE_NAME, COL_FILE_DIR, COL_FILE_PATTERN]:
#                 count = num_files
#                 if compact:
#                     nice_subject = Path(subject).name
#             elif view_type in [COL_APP_NAME, COL_NODE_NAME, COL_PROC_NAME, COL_RANK]:
#                 count = num_processes
#             else:
#                 count = num_time_periods

#             description = (
#                 f"{count:,} {self.pluralize.plural_noun(nice_view_type, count)} ({nice_subject}) "
#                 f"{self.pluralize.plural_verb('has', count)} an I/O time of {time:.2f} seconds "
#                 f"across {num_ops:,} I/O {self.pluralize.plural_noun('operation', num_ops)} "
#                 f"which is {time_overall*100:.2f}% of overall I/O time of the workload."
#             )

#         return description

#     def describe_reason(self, bottleneck: dict, reason_index: int):
#         reason_template = self.rule.reasons[reason_index].message
#         return jinja_env.from_string(reason_template).render(bottleneck).strip()

#     @staticmethod
#     def _group_similar_behavior(bottlenecks: pd.DataFrame, metric: str, view_type: str):
#         behavior_col = 'behavior'
#         cols = bottlenecks.columns

#         if len(bottlenecks) > 1:
#             behavior_cols = cols[cols.str.contains('_min|_max|_count|_size')]

#             link_mat = linkage(bottlenecks[behavior_cols], method='single')
#             behavior_labels = fcluster(link_mat, t=10, criterion='distance')

#             bottlenecks[behavior_col] = behavior_labels
#         else:
#             bottlenecks[behavior_col] = 1

#         agg_dict = {col: 'mean' for col in cols}
#         agg_dict[view_type] = list
#         agg_dict.pop(f"{metric}_score")

#         return bottlenecks.reset_index().groupby(behavior_col).agg(agg_dict)

#     @staticmethod
#     def _union_details(
#         details: pd.DataFrame, behavior: int, indices: list, view_type: str
#     ):
#         view_types = details.index.names

#         filtered_details = details.query(
#             f"{view_type} in @indices", local_dict={'indices': indices}
#         )

#         if len(view_types) == 1:
#             # This means there is only one view type
#             detail_groups = filtered_details

#             # So override aggregations accordingly
#             agg_dict = {}
#             agg_dict[view_type] = set
#         else:
#             agg_dict = {col: set for col in SCORING_ORDER[view_type]}
#             agg_dict.pop(view_type)

#             for agg_key in agg_dict.copy():
#                 if agg_key not in view_types:
#                     agg_dict.pop(agg_key)

#             detail_groups = (
#                 filtered_details.reset_index().groupby(view_type).agg(agg_dict)
#             )

#             # Then create unions for other view types
#             agg_dict = {col: lambda x: set.union(*x) for col in agg_dict}
#             agg_dict[view_type] = set

#         return (
#             detail_groups.reset_index()
#             .assign(behavior=behavior)
#             .groupby(['behavior'])
#             .agg(agg_dict)
#             .reset_index(drop=True)
#         )


# class CharacteristicRule(RuleHandler):
#     deps: List[str] = []
#     prefer_app_characteristics: bool = False

#     @abc.abstractmethod
#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         raise NotImplementedError

#     @abc.abstractmethod
#     def handle_task_results(
#         self,
#         result: Dict[str, Union[str, int, pd.DataFrame, pd.Series]],
#         characteristics: Dict[str, RuleResult] = None,
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         raise NotImplementedError


# class CharacteristicAccessPatternRule(CharacteristicRule):
#     def __init__(self) -> None:
#         super().__init__(rule_key=KnownCharacteristics.ACCESS_PATTERN.value)

#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         acc_pat_cols = []
#         for acc_pat in list(AccessPattern):
#             for col_suffix in ACC_PAT_SUFFIXES:
#                 col_name = f"{acc_pat.name.lower()}_{col_suffix}"
#                 acc_pat_cols.append(col_name)

#         return {'acc_pat_sum': view[acc_pat_cols].sum()}

#     def handle_task_results(
#         self,
#         result: Dict[str, Union[str, int, pd.DataFrame, pd.Series]],
#         characteristics: Dict[str, RuleResult] = None,
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         acc_pat_sum = result['acc_pat_sum']

#         sequential_count = int(acc_pat_sum['sequential_count'])
#         random_count = int(acc_pat_sum['random_count']) if 'random_count' in acc_pat_sum else 0
#         total_count = sequential_count + random_count

#         sequential_title = '[bold]Sequential[/bold]'
#         random_title = '[bold]Random[/bold]'

#         if total_count > 0:
#             sequential_per_fmt = f"{sequential_count / total_count * 100:.2f}"
#             random_per_fmt = f"{random_count / total_count * 100:.2f}"

#             compact_desc = f"{sequential_title}: {sequential_per_fmt}% - {random_title}: {random_per_fmt}% "

#             value_fmt = (
#                 f"{sequential_title}: {sequential_count:,} ops ({sequential_per_fmt}%) - "
#                 f"{random_title}: {random_count:,} ops ({random_per_fmt}%) "
#             )
#         else:
#             compact_desc = f"{sequential_title}: N/A - {random_title}: N/A "

#             value_fmt = f"{sequential_title}: N/A - {random_title}: N/A "

#         return RuleResult(
#             compact_desc=compact_desc,
#             description='Access Pattern',
#             detail_list=None,
#             extra_data=dict(acc_pat_sum),
#             object_hash=None,
#             reasons=None,
#             value=None,
#             value_fmt=value_fmt,
#         )


# class CharacteristicAppTimeRule(CharacteristicRule):
#     def __init__(self) -> None:
#         super().__init__(rule_key=KnownCharacteristics.APP_TIME.value)

#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         tasks = {}
#         tasks['times'] = compute_time_boundaries(view, COL_TIME_RANGE)
#         return tasks

#     def handle_task_results(
#         self,
#         result: Dict[str, Union[str, int, pd.DataFrame, pd.Series]],
#         characteristics: Dict[str, RuleResult] = None,
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         app_time = result['times']
#         job_time = raw_stats['job_time']
#         detail_list = []
#         for time_col, time_value in app_time.items():
#             time_per = f"{time_value / job_time * 100:.2f}%"
#             if time_col in HUMANIZED_COLS:
#                 time_name = HUMANIZED_COLS[time_col]
#                 detail = f"{time_name}: {time_value:.2f} seconds ({time_per})"
#                 detail_list.append(detail)
#         return RuleResult(
#             description='App Time',
#             detail_list=detail_list,
#             extra_data=app_time.to_dict(),
#             value_fmt=f"Runtime: {job_time:.2f} seconds (100%)",
#         )


# class CharacteristicComplexityRule(CharacteristicRule):
#     def __init__(self) -> None:
#         super().__init__(rule_key=KnownCharacteristics.COMPLEXITY.value)
#         self.deps = [
#             KnownCharacteristics.FILE_COUNT.value,
#             KnownCharacteristics.PROC_COUNT.value,
#             KnownCharacteristics.TIME_PERIOD.value,
#         ]

#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         tasks = {}

#         return tasks

#     def handle_task_results(
#         self,
#         result: dict,
#         characteristics: Dict[str, RuleResult],
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         num_files = characteristics[KnownCharacteristics.FILE_COUNT.value].value
#         num_processes = characteristics[KnownCharacteristics.PROC_COUNT.value].value
#         num_time_periods = characteristics[KnownCharacteristics.TIME_PERIOD.value].value

#         complexities = np.array([num_processes, num_time_periods, num_files])
#         complexity = np.log10(ft.reduce(np.multiply, complexities[complexities != 0]))

#         return RuleResult(
#             compact_desc=f"{complexity:.2f}",
#             description='Complexity',
#             detail_list=None,
#             extra_data=None,
#             object_hash=None,
#             reasons=None,
#             value=complexity,
#             value_fmt=f"{complexity:.2f}",
#         )


# class CharacteristicFileCountRule(CharacteristicRule):
#     def __init__(self) -> None:
#         super().__init__(rule_key=KnownCharacteristics.FILE_COUNT.value)

#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         x = view.reset_index()

#         tasks = {}

#         if COL_FILE_NAME in x.columns:
#             tasks['total_count'] = x[COL_FILE_NAME].nunique()

#             if COL_PROC_NAME in x.columns:
#                 fpp = x.groupby([COL_FILE_NAME])[COL_PROC_NAME].nunique().to_frame()

#                 fpp_count = fpp[fpp[COL_PROC_NAME] == 1][COL_PROC_NAME].count()

#             else:
#                 fpp_count = 0

#             tasks['fpp_count'] = fpp_count
#         else:
#             tasks['total_count'] = 0
#             tasks['fpp_count'] = 0

#         return tasks

#     def handle_task_results(
#         self,
#         result: Dict[str, Union[str, int, pd.DataFrame, pd.Series]],
#         characteristics: Dict[str, RuleResult] = None,
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         total_count = int(result['total_count'])
#         fpp_count = int(result['fpp_count'])

#         value_fmt = f"{total_count:,} {self.pluralize.plural_noun('file', total_count)}"

#         compact_desc = [value_fmt]
#         detail_list = []

#         shared_title = 'Shared'
#         fpp_title = 'FPP'

#         if total_count == 0 or fpp_count == 0:
#             detail_list.append(f"{shared_title}: N/A")
#             detail_list.append(f"{fpp_title}: N/A")
#         else:
#             fpp_per = f"{fpp_count / total_count * 100:.2f}%"

#             shared_count = total_count - fpp_count
#             shared_per = f"{shared_count / total_count * 100:.2f}%"

#             shared_fmt = f"{shared_count:,} {self.pluralize.plural_noun('file', shared_count)}"
#             fpp_fmt = f"{fpp_count:,} {self.pluralize.plural_noun('file', fpp_count)}"

#             compact_desc.append(f"[bold]{shared_title}[/bold]: {shared_per}")
#             compact_desc.append(f"[bold]{fpp_title}[/bold]: {fpp_per}")

#             detail_list.append(f"{shared_title}: {shared_fmt} ({shared_per})")
#             detail_list.append(f"{fpp_title}: {fpp_fmt} ({fpp_per})")

#         return RuleResult(
#             compact_desc=' - '.join(compact_desc),
#             description='Files',
#             detail_list=detail_list,
#             extra_data=None,
#             object_hash=None,
#             reasons=None,
#             value=total_count,
#             value_fmt=value_fmt,
#         )


# class CharacteristicIOOpsRule(CharacteristicRule):
#     def __init__(self, is_unoverlapped: bool) -> None:
#         super().__init__(rule_key=KnownCharacteristics.IO_COUNT.value)
#         self.is_unoverlapped = is_unoverlapped

#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         tasks = {}
#         tasks['total_count'] = view['count'].sum()
#         for io_type in IO_TYPES:
#             count_col = f"{io_type}_count"
#             tasks[count_col] = view[count_col].sum()
#         return tasks

#     def handle_task_results(
#         self,
#         result: Dict[str, Union[str, int, pd.DataFrame, pd.Series]],
#         characteristics: Dict[str, RuleResult] = None,
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         total_count = int(result['total_count'])

#         compact_desc = [f"{numerize(total_count, 1)} ops"]
#         detail_list = []

#         for i, io_type in enumerate(IO_TYPES):
#             count_col = f"{io_type}_count"
#             count = int(result[count_col])
#             percent = f"{count / total_count * 100:.2f}%"
#             compact_desc.append(f"[bold]{COMPACT_IO_TYPES[i]}[/bold]: {percent}")
#             detail_list.append(f"{io_type.capitalize()} - {count:,} ops ({percent})")

#         return RuleResult(
#             compact_desc=' - '.join(compact_desc),
#             description='Unoverlapped I/O Ops.' if self.is_unoverlapped else 'I/O Ops.',
#             detail_list=detail_list,
#             extra_data=None,
#             object_hash=None,
#             reasons=None,
#             value=total_count,
#             value_fmt=f"{total_count:,} ops",
#         )


# class CharacteristicIOSizeRule(CharacteristicRule):
#     def __init__(self, is_unoverlapped: bool) -> None:
#         super().__init__(rule_key=KnownCharacteristics.IO_SIZE.value)
#         self.is_unoverlapped = is_unoverlapped

#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         tasks = {}
#         tasks['total_size'] = view['data_size'].sum()
#         for io_type in IO_TYPES:
#             if io_type != 'metadata':
#                 size_col = f"{io_type}_size"
#                 tasks[size_col] = view[size_col].sum()
#         return tasks

#     def handle_task_results(
#         self,
#         result: Dict[str, Union[str, int, pd.DataFrame, pd.Series]],
#         characteristics: Dict[str, RuleResult] = None,
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         total_size = int(result['total_size'])

#         value_fmt = format_bytes(total_size)

#         compact_desc = [value_fmt]
#         detail_list = []

#         if total_size > 0:
#             for i, io_type in enumerate(IO_TYPES):
#                 if io_type != 'metadata':
#                     size_col = f"{io_type}_size"
#                     size = int(result[size_col])
#                     compact_desc.append((f"[bold]{COMPACT_IO_TYPES[i]}[/bold]: {size / total_size * 100:.2f}%"))
#                     detail_list.append(
#                         (f"{io_type.capitalize()} - {format_bytes(size)} ({size / total_size * 100:.2f}%)")
#                     )

#         return RuleResult(
#             compact_desc=' - '.join(compact_desc),
#             description='Unoverlapped I/O Size' if self.is_unoverlapped else 'I/O Size',
#             detail_list=detail_list,
#             extra_data=None,
#             object_hash=None,
#             reasons=None,
#             value=total_size,
#             value_fmt=value_fmt,
#         )


# class CharacteristicIOTimeRule(CharacteristicRule):
#     def __init__(self, is_unoverlapped: bool) -> None:
#         super().__init__(rule_key=KnownCharacteristics.IO_TIME.value)
#         self.deps = [KnownCharacteristics.APP_TIME.value]
#         self.is_unoverlapped = is_unoverlapped
#         self.prefer_app_characteristics = True

#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         tasks = {}
#         tasks['times'] = compute_time_boundaries(view, COL_TIME_RANGE)
#         return tasks

#     def handle_task_results(
#         self,
#         result: Dict[str, Union[str, int, pd.DataFrame, pd.Series]],
#         characteristics: Dict[str, RuleResult] = None,
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         app_time = {}
#         total_unovlp_io_time = 0
#         if not self.is_unoverlapped:
#             # todo(izzet): introduce new config for this
#             # app_time = characteristics[KnownCharacteristics.APP_TIME.value].extra_data
#             # total_unovlp_io_time = app_time['u_io_time']
#             pass
#         total_time_col = 'u_io_time' if self.is_unoverlapped else 'time'
#         total_time = result['times'][total_time_col]
#         compact_desc = [f"{total_time:.2f} s"]
#         detail_list = []
#         for i, io_type in enumerate(IO_TYPES):
#             time_col = f"u_{io_type}_time" if self.is_unoverlapped else f"{io_type}_time"
#             time = result['times'][time_col]
#             time_per = f"{time / total_time * 100:.2f}%"
#             compact_desc.append(f"[bold]{COMPACT_IO_TYPES[i]}[/bold]: {time_per}")
#             detail = f"{io_type.capitalize()}: {time:.2f} seconds ({time_per})"
#             if f"u_{io_type}_time" in app_time and not self.is_unoverlapped:
#                 unovlp_time = app_time[f"u_{io_type}_time"]
#                 unovlp_time_per = f"{unovlp_time / time * 100:.2f}%"
#                 detail += f" - [slate_blue3]Unoverlapped: {unovlp_time:.2f} seconds ({unovlp_time_per})"
#             detail_list.append(detail)
#         value_fmt = f"{total_time:.2f} seconds (100%)"
#         if total_unovlp_io_time > 0 and not self.is_unoverlapped:
#             total_unovlp_io_time_per = f"{total_unovlp_io_time / total_time * 100:.2f}%"
#             value_fmt += (
#                 f" - [slate_blue3]Unoverlapped: {total_unovlp_io_time:.2f} seconds ({total_unovlp_io_time_per})"
#             )
#         return RuleResult(
#             compact_desc=' - '.join(compact_desc),
#             description='Unoverlapped I/O Time' if self.is_unoverlapped else 'I/O Time',
#             detail_list=detail_list,
#             value=total_time,
#             value_fmt=value_fmt,
#         )


# class CharacteristicProcessCount(CharacteristicRule):
#     def __init__(self, rule_key: str) -> None:
#         super().__init__(rule_key=rule_key)
#         self.col = COL_PROC_NAME
#         self.description = 'Processes/Ranks'
#         if rule_key is KnownCharacteristics.APP_COUNT.value:
#             self.col = COL_APP_NAME
#             self.description = 'Apps'
#             self.deps = [
#                 KnownCharacteristics.IO_COUNT.value,
#                 KnownCharacteristics.IO_SIZE.value,
#                 KnownCharacteristics.IO_TIME.value,
#             ]
#         elif rule_key is KnownCharacteristics.NODE_COUNT.value:
#             self.col = COL_NODE_NAME
#             self.description = 'Nodes'
#             self.deps = [
#                 KnownCharacteristics.IO_COUNT.value,
#                 KnownCharacteristics.IO_SIZE.value,
#                 KnownCharacteristics.IO_TIME.value,
#             ]

#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         view_types = view.index._meta.names

#         tasks = {}

#         if COL_PROC_NAME not in view_types:
#             if self.col == COL_PROC_NAME:
#                 tasks[f"{self.col}s"] = 0
#                 return tasks
#             else:
#                 tasks[f"{self.col}s"] = pd.DataFrame()
#                 return tasks

#         if self.col == COL_PROC_NAME:
#             tasks[f"{self.col}s"] = view.map_partitions(lambda df: df.index.get_level_values(COL_PROC_NAME)).nunique()
#         else:
#             tasks[f"{self.col}s"] = (
#                 view.map_partitions(set_proc_name_parts)
#                 .reset_index()
#                 .groupby([self.col, COL_PROC_NAME])
#                 .agg(
#                     {
#                         'count': sum,
#                         'time': sum,
#                         'read_size': sum,
#                         'write_size': sum,
#                     }
#                 )
#                 .groupby([self.col])
#                 .agg(
#                     {
#                         'count': sum,
#                         'time': max,
#                         'read_size': sum,
#                         'write_size': sum,
#                     }
#                 )
#                 .sort_values('time', ascending=False)
#             )

#         return tasks

#     def handle_task_results(
#         self,
#         result: Dict[str, Union[str, int, pd.DataFrame, pd.Series]],
#         characteristics: Dict[str, RuleResult] = None,
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         if self.col == COL_PROC_NAME:
#             num_processes = int(result[f"{self.col}s"])

#             value_fmt = f"{num_processes:,} {self.pluralize.plural_noun('process', num_processes)}"

#             return RuleResult(
#                 compact_desc=value_fmt,
#                 description=self.description,
#                 detail_list=None,
#                 extra_data=None,
#                 object_hash=None,
#                 reasons=None,
#                 value=num_processes,
#                 value_fmt=value_fmt,
#             )

#         max_io_time = characteristics[KnownCharacteristics.IO_TIME.value].value
#         total_ops = characteristics[KnownCharacteristics.IO_COUNT.value].value
#         total_size = characteristics[KnownCharacteristics.IO_SIZE.value].value

#         nodes_apps = pd.DataFrame(result[f"{self.col}s"])

#         detail_list = []
#         if len(nodes_apps) > 1 and total_size > 0:
#             for node, row in nodes_apps.iterrows():
#                 read_size = row['read_size']
#                 write_size = row['write_size']
#                 read_size_fmt = format_bytes(read_size)
#                 write_size_fmt = format_bytes(write_size)
#                 read_size_per = read_size / total_size * 100
#                 write_size_per = write_size / total_size * 100
#                 detail_list.append(
#                     ' - '.join(
#                         [
#                             node,
#                             f"{row['time']:.2f} s ({row['time'] / max_io_time * 100:.2f}%)",
#                             f"{read_size_fmt}/{write_size_fmt} R/W ({read_size_per:.2f}/{write_size_per:.2f}%)",
#                             f"{int(row['count']):,} ops ({row['count'] / total_ops * 100:.2f}%)",
#                         ]
#                     )
#                 )

#         num_nodes_apps = len(nodes_apps)

#         if self.col == COL_NODE_NAME:
#             value_fmt = f"{num_nodes_apps} {self.pluralize.plural_noun('node', num_nodes_apps)}"
#         else:
#             value_fmt = f"{num_nodes_apps} {self.pluralize.plural_noun('app', num_nodes_apps)}"

#         return RuleResult(
#             compact_desc=value_fmt,
#             description=self.description,
#             detail_list=detail_list,
#             extra_data=None,
#             object_hash=None,
#             reasons=None,
#             value=num_nodes_apps,
#             value_fmt=value_fmt,
#         )


# class CharacteristicTimePeriodCountRule(CharacteristicRule):
#     def __init__(self) -> None:
#         super().__init__(rule_key=KnownCharacteristics.TIME_PERIOD.value)

#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         x = view.reset_index()

#         tasks = {}

#         if COL_TIME_RANGE in x.columns:
#             tasks['total_count'] = x[COL_TIME_RANGE].nunique()
#         else:
#             tasks['total_count'] = 0

#         return tasks

#     def handle_task_results(
#         self,
#         result: Dict[str, int],
#         characteristics: Dict[str, RuleResult] = None,
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         num_time_periods = int(result["total_count"])
#         compact_desc = f"{num_time_periods:,} {self.pluralize.plural_noun('time period', num_time_periods)}"
#         time_granularity = raw_stats["time_granularity"]
#         time_granularity_fmt = f"{time_granularity:,.2f}"
#         if time_granularity % 1e6 == 0:
#             time_granularity_fmt = f"{time_granularity / 1e6:,.2f} seconds"
#         return RuleResult(
#             compact_desc=compact_desc,
#             description='Time Periods',
#             detail_list=None,
#             extra_data=None,
#             object_hash=None,
#             reasons=None,
#             value=num_time_periods,
#             value_fmt=f"{compact_desc} (Time Granularity: {time_granularity_fmt})",
#         )


# class CharacteristicXferSizeRule(CharacteristicRule):
#     def __init__(self, rule_key: str) -> None:
#         super().__init__(rule_key)
#         self.io_op = 'write' if rule_key is KnownCharacteristics.WRITE_XFER_SIZE.value else 'read'

#     def define_tasks(self, view: dd.DataFrame) -> Dict[str, Delayed]:
#         tasks = {}

#         count_col, min_col, max_col, per_col, xfer_col = (
#             f"{self.io_op}_count",
#             f"{self.io_op}_min",
#             f"{self.io_op}_max",
#             'per',
#             'xfer',
#         )

#         min_view = view[view[min_col] > 0]
#         max_view = view[view[max_col] > 0]

#         tasks['min_xfer_size'] = min_view[min_col].min()
#         tasks['max_xfer_size'] = max_view[max_col].max()

#         tasks['xfer_sizes'] = max_view.groupby(max_col)[count_col].sum()

#         return tasks

#     def handle_task_results(
#         self,
#         result: Dict[str, Union[str, int, pd.DataFrame, pd.Series]],
#         characteristics: Dict[str, RuleResult] = None,
#         raw_stats: RawStats = None,
#     ) -> RuleResult:
#         count_col, min_col, max_col, per_col, xfer_col = (
#             f"{self.io_op}_count",
#             f"{self.io_op}_min",
#             f"{self.io_op}_max",
#             'per',
#             'xfer',
#         )

#         min_xfer_size = 0
#         max_xfer_size = 0
#         if not np.isnan(result['min_xfer_size']):
#             min_xfer_size = int(result['min_xfer_size'])
#         if not np.isnan(result['max_xfer_size']):
#             max_xfer_size = int(result['max_xfer_size'])

#         xfer_sizes = pd.DataFrame(result['xfer_sizes'])
#         xfer_sizes[xfer_col] = pd.cut(
#             xfer_sizes.index,
#             bins=XFER_SIZE_BINS,
#             labels=XFER_SIZE_BIN_LABELS,
#             right=True,
#         )
#         xfer_bins = xfer_sizes.groupby([xfer_col], observed=True).sum().replace(0, np.nan).dropna()
#         xfer_bins.loc[:, per_col] = xfer_bins[count_col] / xfer_bins[count_col].sum()

#         total_ops = int(xfer_bins[count_col].sum())

#         compact_desc = self._get_xfer_size(max_xfer_size)

#         if min_xfer_size > 0 and max_xfer_size > 0:
#             if min_xfer_size == max_xfer_size:
#                 compact_desc = f"{self._get_xfer_size(min_xfer_size, True)}-{self._get_xfer_size(max_xfer_size)}"
#             else:
#                 compact_desc = f"{self._get_xfer_size(min_xfer_size)}-{self._get_xfer_size(max_xfer_size)}"

#         detail_list = []
#         others_count = 0
#         others_per = 0.0
#         for xfer, row in xfer_bins.iterrows():
#             xfer_count = int(row[count_col])
#             xfer_per = row['per'] * 100
#             if xfer_per < 1:
#                 others_count += xfer_count
#                 others_per += xfer_per
#             else:
#                 detail_list.append(f"{xfer}: {xfer_count:,} ops ({xfer_per:.2f}%)")

#         if others_count > 0:
#             detail_list.append(f"[bright_black][Others: {others_count:,} ops ({others_per:.2f}%)]")

#         result = RuleResult(
#             _dataframe=xfer_bins,
#             compact_desc=compact_desc,
#             description='Write Requests' if self.io_op == 'write' else 'Read Requests',
#             detail_list=detail_list,
#             extra_data=None,
#             object_hash=None,
#             reasons=None,
#             value=(min_xfer_size, max_xfer_size),
#             value_fmt=f"{compact_desc} - {total_ops:,} ops",
#         )

#         return result

#     @staticmethod
#     def _get_xfer_size(size: float, previous=False):
#         size_bin = np.digitize(size, bins=XFER_SIZE_BINS, right=True)
#         if previous:
#             size_bin = size_bin - 1
#         size_label = np.choose(size_bin, choices=XFER_SIZE_BIN_NAMES, mode='clip')
#         return size_label

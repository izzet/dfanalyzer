"""Step attribution and reductions over native, already aggregated records."""

import numpy as np
import pandas as pd
from betterset import BetterSet

from .analyzer import HLM_EXTRA_COLS

UNASSIGNED_STEP = '<unassigned>'
STEP_SCOPE = ('trace_id', 'workflow_id', 'agent_id', 'host_hash')


def attribute_steps(events, profiles=None):
    """Attribute whole native aggregates to a uniquely containing step window.

    Step phases share a step ID, so their union's bounding interval defines
    the step. Native aggregates keep min(ts)/max(te); a group crossing a step
    boundary is deliberately left unassigned. Scope fields constrain a match
    when both sides know them. No pid constraint: tools can be child processes.
    """
    events = events.copy()
    fields = (*STEP_SCOPE, 'step')
    for c in fields:
        events[c] = events[c].astype('string').fillna('')
    spans = events[(events['cat'] == 'step') & (events['step'] != '')]
    windows = (
        spans.groupby(list(fields), sort=False, dropna=False)
        .agg(start=('time_start', 'min'), end=('time_end', 'max'))
        .reset_index()
    )
    # Step IDs are local to a workflow/agent. Qualify a reused ID so the
    # default step view never folds different agents' steps together.
    repeated = set(windows.loc[windows['step'].duplicated(keep=False), 'step'])

    def label(step, workflow='', agent='', trace=''):
        if not step:
            return UNASSIGNED_STEP
        if step in repeated:
            return '/'.join((workflow or trace or 'workflow', agent or 'agent', step))
        return step

    def apply(df):
        if df is None:
            return None
        df = df.copy()
        for c in fields:
            df[c] = df[c].astype('string').fillna('')
        starts = df['time_start'].to_numpy(dtype='int64')
        ends = df['time_end'].to_numpy(dtype='int64')
        order = np.argsort(starts, kind='stable')
        sorted_starts = starts[order]
        hits = np.zeros(len(df), dtype='int64')
        match = np.full(len(df), -1, dtype='int64')
        # Search only records starting inside each window, rather than forming
        # the Cartesian product of all aggregate rows and all agent steps.
        for i, window in windows.iterrows():
            lo, hi = np.searchsorted(sorted_starts, [window['start'], window['end']])
            rows = order[lo:hi]
            keep = ends[rows] <= window['end']
            for c in fields:
                values = df[c].iloc[rows].to_numpy(dtype=str)
                known = window[c]
                if known:
                    keep &= (values == '') | (values == known)
            rows = rows[keep]
            hits[rows] += 1
            match[rows] = i
        unique = np.flatnonzero(hits == 1)
        for c in fields:
            values = df[c].to_numpy(dtype=str).astype(object)
            missing = unique[values[unique] == '']
            values[missing] = windows[c].iloc[match[missing]].to_numpy()
            df[c] = pd.array(values, dtype='string')
        df['step'] = pd.array(
            [label(r.step, r.workflow_id, r.agent_id, r.trace_id) for r in df.itertuples()], dtype='string'
        )
        df.loc[hits > 1, 'step'] = UNASSIGNED_STEP
        if 'total_tokens' in df:
            total = df['prompt_tokens'].fillna(0) + df['completion_tokens'].fillna(0)
            df['total_tokens'] = df['total_tokens'].where(df['total_tokens'].fillna(0) != 0, total)
        return df

    return apply(events), apply(profiles)


def reduce_typed_hlm(frame, view_types, metric_fields):
    """Reduce the native read after attribution, preserving its power sums.

    The native reader already materializes these rows at the coordinator.
    Reuse them for every view: re-querying the index would discard the inferred
    step labels. No raw events are read here.
    """
    keys = list(dict.fromkeys(list(view_types) + HLM_EXTRA_COLS))
    metrics = [
        'time',
        'count',
        'size',
        'time_sq',
        'size_sq',
        'time_call_min',
        'time_call_max',
        'size_call_min',
        'size_call_max',
    ]
    metrics += list(metric_fields)
    # The folded per-call bounds follow the existing View HLM convention.
    df = frame.copy()
    for m in ('time', 'size'):
        df[m + '_call_min'] = df[m]
        df[m + '_call_max'] = df[m]
    aggs = {c: ('min' if c.endswith('_call_min') else 'max' if c.endswith('_call_max') else 'sum') for c in metrics}
    hlm = df.groupby(keys, dropna=False, observed=True).agg(aggs)
    for c in metrics:
        dtype = 'Int64' if c in ('count', 'size', 'size_call_min', 'size_call_max') else 'Float64'
        hlm[c] = hlm[c].replace(0, pd.NA).astype(dtype)
    return hlm


def coalesce_process_records(frame, view_type):
    """Combine step/bucket fragments before computing per-process statistics.

    A process may contribute several steps to one time bucket, or several
    buckets to one step. A maximum across those fragments would understate
    that process's time; sum its contributions before comparing processes.
    """
    work = frame.reset_index()
    aggs = {}
    for col in frame.columns:
        if col.endswith('_call_min'):
            aggs[col] = 'min'
        elif col.endswith('_call_max'):
            aggs[col] = 'max'
        elif pd.api.types.is_numeric_dtype(frame[col].dtype):
            aggs[col] = 'sum'
        else:
            aggs[col] = lambda values: frozenset(BetterSet.flatten(values.dropna()))
    return work.groupby([view_type, 'proc_name'], observed=True, dropna=False).agg(aggs)

import dask.dataframe as dd
import numpy as np
import portion as P
from betterset import BetterSet as S


def nunique():
    return dd.Aggregation(
        name="nunique",
        chunk=lambda s: s.apply(lambda x: list(set(x))),
        agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
        finalize=lambda s1: s1.apply(lambda final: len(set(final))),
    )


def quantile_stats(min, max):
    def quantile_stats_finalize(values_list):
        if len(values_list) == 0:
            return [np.nan, np.nan, np.nan]
        values_array = np.array(values_list)
        q_min, q_max = np.quantile(values_array, [min, max])
        filtered_mask = (values_array >= q_min) & (values_array <= q_max)
        filtered_values = values_array[filtered_mask]
        if len(filtered_values) == 0:
            return [np.nan, np.nan, np.nan]
        return [np.mean(filtered_values), np.std(filtered_values), len(filtered_values)]

    return dd.Aggregation(
        f"q{min * 100:.0f}_q{max * 100:.0f}_stats",
        lambda s: s.apply(lambda x: x.replace(0, np.nan).dropna().tolist()),
        lambda s0: s0.obj.groupby(level=0).sum(),
        lambda s1: s1.apply(quantile_stats_finalize),
    )


def unique_set():
    return dd.Aggregation(
        'unique',
        lambda s: s.apply(lambda x: S(x.dropna().unique().tolist())),
        lambda s0: s0.apply(S.flatten),
    )


def unique_set_flatten():
    return dd.Aggregation(
        'unique',
        lambda s: s.apply(S.flatten),
        lambda s0: s0.agg(S.flatten),
    )


def union_portions():
    def union_s(s):
        emp = P.empty()
        for x in s:
            emp = emp | x
        return emp

    def fin(s):
        val = 0.0
        for i in s:
            if not i.is_empty():
                val += i.upper - i.lower
        return val

    return dd.Aggregation(
        'portion',
        union_s,
        union_s,
        fin,
    )

"""pandas equivalents of the Dask set-union aggregations in `dask_agg`.

Dask expresses these as `dask.dataframe.Aggregation` objects, which pandas'
`groupby().agg()` cannot consume. These are plain callables with the same
semantics, so a view computed on either engine produces the same column.

Kept in step with `python/dftracer/analyzer/utils/dask_agg.py`.
"""

from betterset import BetterSet


def unique_set_pd(s):
    """Distinct non-null values of a series, as a frozenset."""
    return frozenset(BetterSet(s.dropna().unique().tolist()))


def unique_set_flatten_pd(s):
    """Union of the set-valued entries of a series, as a frozenset."""
    return frozenset(BetterSet.flatten(s.dropna()))


# `groupby().agg()` names the output column after the callable, and the Dask
# aggregations it mirrors are both named "unique".
unique_set_pd.__name__ = "unique"
unique_set_flatten_pd.__name__ = "unique"

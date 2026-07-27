"""Engine differences between Dask and pandas frames, isolated in one place.

An aggregation pipeline that must run on either engine otherwise fills up with
`if is_dask:` branches. This keeps the algorithm written once: callers ask
`DataFrameOps.of(frame)` and then work through the returned object, which is
the only thing that knows which engine is in play.

Most of the surface is mechanical -- `map_partitions(fn)` versus `fn(df)`,
`persist()` versus nothing. The part worth reading is how the pandas side
emulates Dask's `meta` semantics, because getting that wrong produces frames
that differ only in dtype, only for empty inputs, and only sometimes:

* Dask coerces **every** partition to a meta schema. Helpers that short-circuit
  on an empty frame and return it unchanged therefore still come out with the
  populated schema. `finalize_view_partials` does exactly this: given an empty
  frame it returns the raw `_count`/`_sumsq` columns rather than `_mean`/`_std`,
  and Dask silently repairs it.
* When no explicit meta is supplied, Dask *infers* one by running the function
  against a synthetic non-empty frame (`meta_nonempty`). So even without a meta,
  an empty partition ends up with the schema the function would have produced
  had there been rows.

pandas does neither, so `_PandasOps.apply` reproduces both. Without that, empty
layers come out as `double[pyarrow]` on one engine and `Float64` on the other.

Not a dataframe implementation: it adapts execution semantics for frames that
already exist.
"""

from typing import Any, Callable, Optional

import dask.dataframe as dd

from .dask_agg import unique_set, unique_set_flatten
from .pandas_agg import unique_set_flatten_pd, unique_set_pd


class DataFrameOps:
    """Dispatch table for the operations that differ between engines."""

    is_dask: bool

    @staticmethod
    def of(df) -> "DataFrameOps":
        return _DaskOps() if isinstance(df, dd.DataFrame) else _PandasOps()

    def apply(self, df, fn: Callable, *args, meta: Optional[Callable] = None, **kwargs):
        """Apply a per-partition function.

        `meta` is a thunk rather than a value so the Dask-only meta builders are
        never invoked on a pandas frame.
        """
        raise NotImplementedError

    def index_names(self, df):
        """Names of the frame's index levels."""
        raise NotImplementedError

    def set_union(self):
        """Aggregation collecting distinct values into a set."""
        raise NotImplementedError

    def set_union_flatten(self):
        """Aggregation unioning set-valued entries."""
        raise NotImplementedError

    def finalize(self, df):
        """Make the result concrete, if the engine has such a notion."""
        raise NotImplementedError

    def mutable(self, df):
        """A frame whose columns may be assigned without affecting the caller."""
        raise NotImplementedError

    def meta_source(self, df) -> Any:
        """Something the Dask meta builders can read `.columns` and `._meta` from."""
        raise NotImplementedError


class _DaskOps(DataFrameOps):
    is_dask = True

    def apply(self, df, fn, *args, meta=None, **kwargs):
        if meta is not None:
            return df.map_partitions(fn, *args, meta=meta(), **kwargs)
        return df.map_partitions(fn, *args, **kwargs)

    def index_names(self, df):
        return df.index._meta.names

    def set_union(self):
        return unique_set()

    def set_union_flatten(self):
        return unique_set_flatten()

    def finalize(self, df):
        return df.persist()

    def mutable(self, df):
        # assignment on a Dask frame builds a new graph; nothing is aliased
        return df

    def meta_source(self, df):
        return df


class _PandasOps(DataFrameOps):
    is_dask = False

    def apply(self, df, fn, *args, meta=None, **kwargs):
        out = fn(df, *args, **kwargs)
        if not getattr(out, "empty", False):
            return out
        schema = meta() if meta is not None else self._infer_schema(df, fn, args, kwargs)
        if schema is None:
            return out
        out = out.reindex(columns=schema.columns)
        return out.astype({c: dt for c, dt in schema.dtypes.items() if c in out.columns})

    @staticmethod
    def _infer_schema(df, fn, args, kwargs):
        """What Dask would have inferred: run `fn` on a synthetic non-empty frame."""
        try:
            from dask.dataframe.utils import meta_nonempty

            return fn(meta_nonempty(df.iloc[:0]), *args, **kwargs).iloc[:0]
        except Exception:
            # Better to leave the frame alone than to guess at a schema.
            return None

    def index_names(self, df):
        return df.index.names

    def set_union(self):
        return unique_set_pd

    def set_union_flatten(self):
        return unique_set_flatten_pd

    def finalize(self, df):
        return df

    def mutable(self, df):
        # column assignment must not reach back into the caller's frame
        return df.copy(deep=False)

    def meta_source(self, df):
        return _MetaSource(df)


class _MetaSource:
    """Adapts a pandas frame to the `.columns` / `._meta` shape the Dask meta
    builders expect, so one implementation of those builders serves both
    engines rather than being duplicated per engine."""

    __slots__ = ("columns", "_meta")

    def __init__(self, df):
        self.columns = df.columns
        self._meta = df.iloc[:0]

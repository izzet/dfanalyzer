import dask.dataframe as dd

def flatten_column_names(ddf: dd.DataFrame):
    ddf.columns = ['_'.join(tup).rstrip('_') for tup in ddf.columns.values]
    return ddf


def persisted_nbytes(df):
    """Materialised size of an already-computed Dask collection, or None.

    Never triggers computation. `persist()` is asynchronous, so partitions that
    have not finished are simply absent from the scheduler's map; that is
    reported as unknown rather than guessed at. Callers must not wait on this --
    waiting would impose a barrier on exactly the large inputs it exists to
    protect.
    """
    try:
        from distributed import futures_of

        futures = futures_of(df)
        if not futures:
            return None
        known = futures[0].client.nbytes(summary=False) or {}
        sizes = [known.get(f.key) for f in futures]
        if any(size is None for size in sizes):
            return None
        return int(sum(sizes))
    except Exception:
        return None

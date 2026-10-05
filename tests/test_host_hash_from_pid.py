"""Host hash propagation for DFTracer builds that write hhash only in per-process metadata records
(PyPI dftracer 1.0.15, SFNO venv): events carry no args, so every event's host_hash was NA and the
analyzer reported unique_host_count=0 (schema-valid zeros). Fill NA from the same pid's metadata record."""
import pandas as pd
import dask.dataframe as dd
import pytest

from dftracer.analyzer import dftracer as D

Cls = next(v for k, v in vars(D).items() if isinstance(v, type) and hasattr(v, "_fill_host_hash_from_pid"))


def _raw():
    return pd.DataFrame({
        "type": [D.TYPE_HOST_HASH, D.TYPE_METADATA, D.TYPE_EVENT, D.TYPE_EVENT, D.TYPE_EVENT, D.TYPE_EVENT],
        "pid": [10, 10, 10, 10, 11, 12],
        "tid": [10, 10, 10, 77, 11, 12],
        "host_hash": pd.array(["h1", "h1", None, None, "own", None], dtype="string"),
    })


@pytest.mark.parametrize("dask", [False, True])
def test_fill_only_na_from_same_pid(dask):
    raw = _raw()
    rr = dd.from_pandas(raw, npartitions=2) if dask else raw
    pid_hosts = Cls._pid_host_table(rr, dask)
    ev = rr[rr["type"] == D.TYPE_EVENT]
    out = Cls._fill_host_hash_from_pid(ev, pid_hosts)
    out = out.compute() if dask else out
    assert list(out["pid"]) == [10, 10, 11, 12]
    assert out["host_hash"].tolist()[:3] == ["h1", "h1", "own"]   # same pid (any tid) filled; existing kept
    assert pd.isna(out["host_hash"].tolist()[3])                   # unknown pid stays NA (never invented)

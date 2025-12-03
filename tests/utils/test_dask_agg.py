import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from dftracer.analyzer.utils.dask_agg import unique_set, unique_set_flatten

# Ensure this module runs in both smoke and full CI modes
pytestmark = [pytest.mark.smoke, pytest.mark.full]


def test_unique_set_scalar_column_via_dask():
    df = pd.DataFrame(
        [
            {"g": "a", "col": 1},
            {"g": "a", "col": 2},
            {"g": "a", "col": 2},
            {"g": "b", "col": 3},
            {"g": "b", "col": 4},
            {"g": "b", "col": 4},
        ]
    )
    ddf = dd.from_pandas(df, npartitions=2)
    res = ddf.groupby("g").agg({"col": unique_set()}).compute()["col"]
    assert set(res.loc["a"]) == {1, 2}
    assert set(res.loc["b"]) == {3, 4}


def test_unique_set_flatten_grouped_two_stage_via_dask():
    df = pd.DataFrame(
        [
            {"g": "a", "p": "x", "col": 1},
            {"g": "a", "p": "x", "col": 2},
            {"g": "a", "p": "y", "col": 2},
            {"g": "b", "p": "y", "col": 3},
            {"g": "b", "p": "y", "col": 4},
            {"g": "b", "p": "y", "col": 4},
        ]
    )
    ddf = dd.from_pandas(df, npartitions=2)
    res = (
        ddf.groupby(["g", "p"])
        .agg({"col": unique_set()})
        .groupby(["p"])
        .agg({"col": unique_set_flatten()})
        .compute()["col"]
    )
    assert set(res.loc["x"]) == {1, 2}
    assert set(res.loc["y"]) == {2, 3, 4}


def test_unique_set_flatten_grouped_three_stage_via_dask():
    df = pd.DataFrame(
        [
            {"g": "a", "p": "x", "q": "z", "col": 1},
            {"g": "a", "p": "x", "q": "z", "col": 2},
            {"g": "a", "p": "y", "q": "w", "col": 2},
            {"g": "b", "p": "y", "q": "w", "col": 3},
            {"g": "b", "p": "y", "q": "w", "col": 4},
            {"g": "b", "p": "y", "q": "z", "col": 4},
        ]
    )
    ddf = dd.from_pandas(df, npartitions=2)
    res = (
        ddf.groupby(["g", "p", "q"])
        .agg({"col": unique_set()})
        .groupby(["p", "q"])
        .sum()
        .groupby(["q"])
        .agg({"col": unique_set_flatten()})
        .compute()["col"]
    )
    assert set(res.loc["z"]) == {1, 2, 4}
    assert set(res.loc["w"]) == {2, 3, 4}


def test_unique_set_handles_missing_values_via_dask():
    df = pd.DataFrame(
        [
            {"g": "a", "col": 1},
            {"g": "a", "col": np.nan},
            {"g": "a", "col": 2},
            {"g": "b", "col": pd.NA},
            {"g": "b", "col": 3},
            {"g": "b", "col": np.nan},
        ]
    )
    df["col"] = df["col"].astype("Int64")
    ddf = dd.from_pandas(df, npartitions=2)
    res = ddf.groupby("g").agg({"col": unique_set()}).compute()["col"]
    assert set(res.loc["a"]) == {1, 2}
    assert set(res.loc["b"]) == {3}


def test_unique_set_empty_dataframe_returns_empty_series():
    df = pd.DataFrame({"g": pd.Series(dtype="object"), "col": pd.Series(dtype="object")})
    ddf = dd.from_pandas(df, npartitions=2)
    res = ddf.groupby("g").agg({"col": unique_set()}).compute()["col"]
    assert len(res) == 0

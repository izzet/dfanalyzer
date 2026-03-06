import pandas as pd
from betterset import BetterSet as S

from dftracer.analyzer.utils.collection_utils import is_set_like_series


def test_is_set_like_series_detects_betterset_values() -> None:
    series = pd.Series([None, S(["a"]), S(["b"])], dtype="object")
    assert is_set_like_series(series) is True


def test_is_set_like_series_ignores_plain_strings() -> None:
    series = pd.Series(["a", "b", None], dtype="object")
    assert is_set_like_series(series) is False

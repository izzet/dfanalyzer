import pandas as pd

from dftracer.analyzer.utils.pandas_utils import to_nullable_numeric


def test_to_nullable_numeric_preserves_integer_nullability() -> None:
    series = pd.Series([1, 2, 3], dtype="int64")
    out = to_nullable_numeric(series).where(pd.Series([True, False, True]))
    assert str(out.dtype) == "Int64"
    assert out.tolist() == [1, pd.NA, 3]


def test_to_nullable_numeric_preserves_float_nullability() -> None:
    series = pd.Series([1.5, 2.5, 3.5], dtype="float64")
    out = to_nullable_numeric(series).where(pd.Series([True, False, True]))
    assert str(out.dtype) == "Float64"
    assert out.tolist() == [1.5, pd.NA, 3.5]

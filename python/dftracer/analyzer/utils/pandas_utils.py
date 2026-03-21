import pandas as pd


def to_nullable_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_integer_dtype(numeric.dtype):
        return numeric.astype("Int64")
    if pd.api.types.is_float_dtype(numeric.dtype):
        return numeric.astype("Float64")
    return numeric

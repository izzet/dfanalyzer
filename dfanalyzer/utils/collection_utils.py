import pandas as pd
from typing import Iterable, List


def deepflatten(collection, ignore_types=(bytes, str)):
    for x in collection:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from deepflatten(x)
        else:
            yield x


def deepmerge(source: dict, destination: dict):
    for key, value in source.items():
        if isinstance(value, dict):
            deepmerge(value, destination.setdefault(key, {}))
        else:
            destination[key] = value
    return destination


def get_every_x_intervals(values: List[int]):
    every_x = pd.Series(sorted(values)).diff().value_counts()
    every_x_values = every_x[every_x >
                             every_x.std()].index.astype(int).astype(str)
    print(values, list(every_x_values))
    return '-'.join(sorted(every_x_values))


def get_intervals(values: list):
    series = pd.Series(sorted(values, reverse=True))
    grouped = series.groupby(series.diff().fillna(1).ne(-1).cumsum())
    output = grouped.apply(lambda x: str(x.iloc[0]) if len(
        x) == 1 else str(x.iloc[-1]) + '-' + str(x.iloc[0]))
    return list(reversed(output.tolist()))


def join_with_and(values: List[str]):
    if len(values) == 1:
        return values[0]
    elif len(values) == 2:
        return ' and '.join(values)
    else:
        return ', '.join(values[:-1]) + ', and ' + values[-1]

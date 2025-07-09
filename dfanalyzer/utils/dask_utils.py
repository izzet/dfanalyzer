import functools
import logging
import dask.dataframe as dd
from distributed import get_client
from .logger import ElapsedTimeLogger


class EventLogger(ElapsedTimeLogger):
    def __init__(self, key: str, message: str, level=logging.INFO, stacklevel=4):
        super().__init__(message, level, stacklevel=stacklevel)
        self.key = key

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        get_client().log_event(
            'elapsed_times',
            dict(
                elapsed_time=self.elapsed_time,
                key=str(self.key),
                message=self.message,
                start_time=self.start_time,
                end_time=self.end_time,
            ),
        )


def event_logger(key: str, message: str, level=logging.INFO):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with EventLogger(key, message, level, stacklevel=5):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def flatten_column_names(ddf: dd.DataFrame):
    ddf.columns = ['_'.join(tup).rstrip('_') for tup in ddf.columns.values]
    return ddf

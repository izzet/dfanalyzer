import functools
import logging
import logging.config
import structlog
import sys
import time
from contextlib import contextmanager
from rich.console import Console


console = Console()


def configure_logging(log_file: str, json_logs: bool = False, level: str = "info") -> None:
    
    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name
    ]

    logging_config = {
        'version': 1,
        'handlers': {
            
            'json_file': {
                'class': 'logging.FileHandler',
                'filename': log_file,
                'level': 'INFO',
                'formatter': 'json'
            },
        },
        'formatters': {
            'json': {
                '()': structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
                'foreign_pre_chain': pre_chain
            },
            
        },
        'loggers': {
            '': {
                'handlers': ['json_file'],
                'level': 'INFO',
                'propagate': False,
            }
        }
    }

    logging.config.dictConfig(logging_config)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )



@contextmanager
def console_block(message: str, level: str = "info", logger=None, **kwargs):
    """
    A context manager that logs the elapsed time of the block to the console.
    """
    logger = logger or structlog.get_logger()
    start = time.perf_counter()
    with console.status(f"{message}...", spinner="dots"):
        try:
            getattr(logger, level)(f"▶ {message}...", **kwargs)
            yield
        finally:
            elapsed = time.perf_counter() - start
            console.print(f"✓ {message} [i]({elapsed:.3f}s)[/i]")
            getattr(logger, level)(f"✓ {message}", elapsed=elapsed, **kwargs)


@contextmanager
def log_block(message: str, level: str = "info", logger=None, **kwargs):
    """
    A context manager that logs the elapsed time of the block.
    """
    logger = logger or structlog.get_logger()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if kwargs:
            logger = logger.bind(**kwargs)
        getattr(logger, level)(message, elapsed=elapsed)

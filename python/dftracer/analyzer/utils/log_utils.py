import logging
import logging.config
import structlog
import time
from contextlib import contextmanager
from pathlib import Path
from rich.console import Console

from .notebook_utils import IN_JUPYTER

console = Console()


def configure_logging(log_file: str, level: str = "info") -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
    ]

    logging_config = {
        "version": 1,
        # 2026-09-28 (judge batch 5 G4 root cause): dictConfig's own default is True, which
        # disables every ALREADY-CREATED logger not explicitly named in "loggers" below. This
        # process calls configure_logging() twice -- once for the analyzer's own setup, once for
        # the diagnoser's -- and dftracer.analyzer.fact_engine's logger is created and used
        # (facts.rules.loaded, during FactEngine.from_rule_file() inside build_analyzer()) BEFORE
        # the diagnoser's own configure_logging() call runs. That second call, with the implicit
        # default, silently set fact_engine's logger.disabled=True -- which short-circuits every
        # logging call on it (Logger.info()/.debug() check self.disabled before anything else,
        # including level), regardless of level/handlers/propagate. This is why
        # "facts.rules.loaded" always appeared (logged before the disable) while every later call
        # on the SAME logger -- every facts.rule.* line and patch 13's own diagnostic -- silently
        # never did, with NO exception and NO visible symptom other than total silence. Confirmed
        # directly: patch 296's file-persisted diagnostic recorded
        # fact_engine_logger_disabled=true on every one of 15 windows in the v4.5.9 1N run.
        # dftracer.analyzer.analyzer's own logger was unaffected only because its first real use
        # (per-window processing) happens to occur after both configure_logging() calls complete,
        # so it was never in the "already exists" set either dictConfig call could disable.
        "disable_existing_loggers": False,
        "handlers": {
            "json_file": {
                "class": "logging.FileHandler",
                "filename": log_file,
                "level": level.upper(),
                "formatter": "json",
            },
        },
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
                "foreign_pre_chain": pre_chain,
            },
        },
        "loggers": {
            "": {
                "handlers": ["json_file"],
                "level": level.upper(),
                "propagate": False,
            }
        },
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
    if IN_JUPYTER:
        yield
        elapsed = time.perf_counter() - start
        getattr(logger, level)(message, elapsed=elapsed, **kwargs)
        return
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

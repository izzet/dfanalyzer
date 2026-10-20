import glob
import os
from collections.abc import Iterable

import structlog
from dftracer.utils import TraceViewer
from dftracer.utils.dfanalyzer import index_path_for, resolve_trace_inputs

logger = structlog.get_logger()

AI_CATEGORIES = frozenset({"ai_root", "pipeline"})
DLIO_CATEGORIES = frozenset({"reader", "data_loader", "checkpoint", "ai_framework", "dlio_benchmark"})
POSIX_CATEGORIES = frozenset({"posix", "stdio"})


def preset_is_explicit(overrides: Iterable[str]) -> bool:
    return any(o.lstrip("+~").startswith("analyzer/preset") for o in overrides)


def classify_categories(categories: Iterable[str]) -> str:
    cats = {c.lower() for c in categories}
    if cats & AI_CATEGORIES:
        return "ai"
    if cats & DLIO_CATEGORIES:
        return "dlio"
    if cats & POSIX_CATEGORIES:
        return "posix"
    return "auto"


def list_trace_files(trace_path: str, trace_groups: list[str] | None = None) -> list[str]:
    directory, files = resolve_trace_inputs(trace_path, trace_groups)
    if files is not None:
        return list(files)
    return sorted(glob.glob(os.path.join(directory, "*.pfw")) + glob.glob(os.path.join(directory, "*.pfw.gz")))


def detect_preset(trace_path: str, trace_groups: list[str] | None = None) -> str:
    files = list_trace_files(trace_path, trace_groups)
    if not files:
        raise FileNotFoundError("No matching .pfw or .pfw.gz files found.")
    viewer = TraceViewer(files, index_path=index_path_for(trace_path))
    categories = viewer.unique("cat").collect().to_pandas()["cat"].dropna().astype(str)
    name = classify_categories(categories)
    logger.info("Detected analyzer preset", preset=name, categories=sorted(set(categories)))
    return name


def apply_detected_preset(cfg, overrides: Iterable[str]) -> None:
    """Set ``cfg.analyzer.preset`` from the trace unless the caller chose one."""
    from omegaconf import OmegaConf

    from ..config import PRESET_CONFIGS

    if preset_is_explicit(overrides) or OmegaConf.is_missing(cfg, "trace_path"):
        return
    name = detect_preset(cfg.trace_path, cfg.analyzer.get("trace_groups"))
    cfg.analyzer.preset = OmegaConf.structured(PRESET_CONFIGS[name])

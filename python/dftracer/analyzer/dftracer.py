import dask
import dask.bag as db
import dask.dataframe as dd
import glob
import json
import math
import numpy as np
import os
import pandas as pd
import portion as I
import re
import structlog
# dftracer-utils >=0.0.12 renamed Reader -> TraceReader. Support BOTH rather than pinning:
# pinning to 0.0.5 forces a SOURCE build, which fails on this cluster at the cpplogger
# git-clone step, whereas 0.0.12 ships a wheel and installs cleanly.
try:
    from dftracer.utils import Indexer, Reader
except ImportError:  # dftracer-utils >= 0.0.12
    from dftracer.utils import Indexer, TraceReader as Reader
from dask.distributed import wait
from typing import Callable, Dict, List, Optional, Tuple

from .analysis_utils import set_size_bins
from .analyzer import Analyzer
from .constants import (
    COL_ACC_PAT,
    COL_COUNT,
    COL_EPOCH,
    COL_FILE_HASH,
    COL_FILE_NAME,
    COL_FUNC_NAME,
    COL_HOST_HASH,
    COL_HOST_NAME,
    COL_IO_CAT,
    COL_PROC_NAME,
    COL_SIZE,
    COL_TIME,
    COL_TIME_END,
    COL_TIME_RANGE,
    COL_TIME_START,
    POSIX_IO_CAT_MAPPING,
    POSIX_METADATA_FUNCTIONS,
    IOCategory,
)
from .types import ReadTraceResult, ViewType
from .utils.log_utils import log_block

logger = structlog.get_logger()

# Warn once per process, not once per dask partition: the condition is a whole-run
# configuration mistake, and repeating it per partition would bury it.
_PURPOSE_MATCH_WARNED = False

CAT_POSIX = "POSIX"
CAT_STDIO = "STDIO"
IGNORED_FILE_PATTERNS = [
    "/dev/",
    "/etc/",
    "/gapps/python",
    "/lib/python",
    "/proc/",
    "/software/",
    "/sys/",
    "/usr/lib",
    "/usr/tce/backend",
    "/usr/tce/packages",
    "/venv",
    "__pycache__",
]
IGNORED_FUNC_NAMES = [
    "DLIOBenchmark.__init__",
    # 'DLIOBenchmark._train',
    "DLIOBenchmark.initialize",
    # 'DLIOBenchmark.run',
    "FileStorage.__init__",
    "TorchDataset.__init__",
    # "TorchDataset.worker_init",
]
IGNORED_FUNC_PATTERNS = [
    "Checkpointing.__init__",
    "Checkpointing.finalize",
    "Checkpointing.get_tensor",
    "DataLoader.__init__",
    "DataLoader.finalize",
    "DataLoader.get_tensor",
    "DataLoader.next",
    "Framework.get_loader",
    "Framework.init_loader",
    "Framework.is_nativeio_available",
    "Framework.trace_object",
    "Reader.__init__",
    "Reader.load_index",
    "Reader.next",
    "Reader.read_index",
    ".save_state",
    "checkpoint_end_",
    "checkpoint_start_",
]
TRACE_COL_MAPPING = {
    "dur": COL_TIME,
    "name": COL_FUNC_NAME,
    "te": COL_TIME_END,
    "trange": COL_TIME_RANGE,
    "ts": COL_TIME_START,
}
TYPE_EVENT = 0
TYPE_FILE_HASH = 1
TYPE_HOST_HASH = 2
TYPE_STRING_HASH = 3
TYPE_METADATA = 4
TYPE_PROC_METADATA = 5
TYPE_PROFILE = 6
TYPE_SYSTEM = 7
PROFILE_COLUMN_MAPPING = {
    "count": "Int64",
    "count_max": "Int64",
    "count_min": "Int64",
    "count_sum": "Int64",
    "dft_cnt": "Int64",
    "dur": "Int64",
    "dur_max": "Int64",
    "dur_min": "Int64",
    "dur_sum": "Int64",
    "epoch": "Int64",
    "flags": "Int64",
    "offset": "Int64",
    "ret": "Int64",
    "offset_max": "Int64",
    "offset_min": "Int64",
    "offset_sum": "Int64",
    "ret_max": "Int64",
    "ret_min": "Int64",
    "ret_sum": "Int64",
    "whence": "Int64",
    "whence_max": "Int64",
    "whence_min": "Int64",
    "whence_sum": "Int64",
}
PROFILE_OUTPUT_COLUMNS = {
    "cat": "string",
    COL_FUNC_NAME: "string",
    "pid": "Int64",
    "tid": "Int64",
    "epoch": "Int64",
    "step": "Int64",
    "file_hash": "string",
    "host_hash": "string",
    COL_FILE_NAME: "string",
    COL_HOST_NAME: "string",
    COL_PROC_NAME: "string",
    COL_IO_CAT: "Int8",
    COL_ACC_PAT: "Int8",
    COL_COUNT: "Int64",
    COL_TIME: "float64",
    COL_SIZE: "Int64",
    "time_min": "float64",
    "time_max": "float64",
    "size_min": "Int64",
    "size_max": "Int64",
    "offset_min": "Int64",
    "offset_max": "Int64",
    COL_TIME_RANGE: "Int64",
    COL_TIME_START: "Int64",
    COL_TIME_END: "Int64",
}
PROFILE_MEASURE_COLUMNS = [COL_COUNT, COL_TIME, COL_SIZE]
PROFILE_STAT_COLUMNS = ["time_min", "time_max", "size_min", "size_max", "offset_min", "offset_max"]
PROFILE_IDENTITY_COLUMNS = [
    col for col in PROFILE_OUTPUT_COLUMNS
    if col not in PROFILE_MEASURE_COLUMNS and col not in PROFILE_STAT_COLUMNS
]

# System metric columns extracted from cat="sys" ph="C" events
SYSTEM_CPU_METRICS = ["user_pct", "system_pct", "iowait_pct", "idle_pct", "irq_pct", "softirq_pct"]
SYSTEM_MEMORY_METRICS = ["MemAvailable", "MemFree", "Cached", "Dirty", "Active"]
SYSTEM_COLUMN_MAPPING = {
    **{m: "float64" for m in SYSTEM_CPU_METRICS},
    **{m: "float64" for m in SYSTEM_MEMORY_METRICS},
}
SYSTEM_OUTPUT_COLUMNS = {
    "host_hash": "string",
    COL_TIME_RANGE: "Int64",
    "sys_cpu_iowait_pct": "float64",
    "sys_cpu_iowait_pct_core_max": "float64",
    "sys_cpu_iowait_pct_core_p95": "float64",
    "sys_cpu_user_pct": "float64",
    "sys_cpu_system_pct": "float64",
    "sys_cpu_idle_pct": "float64",
    "sys_mem_dirty_pct": "float64",
    "sys_mem_cached_pct": "float64",
    "sys_mem_available_pct": "float64",
    "sys_mem_available_kb": "float64",
}


def create_index(filename):
    # dftracer-utils 0.0.12 changed the Indexer API. The old call was
    # Indexer(<trace file>, <index file>), but the new signature is
    # Indexer(directory="", files=None, index_dir="", ...) -- so the positional form passed a FILE
    # where a DIRECTORY is expected and died with
    #   DFTUtilsError: filesystem error: Path is not a directory: [...].pfw.gz
    # which made EVERY .pfw.gz trace unreadable by dfanalyzer. Since gzipped traces are how we keep
    # the trace corpus on disk (7-12x smaller, byte-identical), that incompatibility is what stands
    # between us and compressing the archive at all.
    #
    # ensure_indexed() rather than build(): the new indexer keeps its state in a `.dftindex` store
    # beside the traces, not in a `<file>.idx` sibling, so the old existence guard could never fire
    # and would have rebuilt the index on every call. ensure_indexed() resolves first and builds
    # only what is missing, which is idempotent and cheap on an already-indexed corpus.
    indexer = Indexer(files=[filename], checkpoint_size=32 * 1024 * 1024)
    indexer.ensure_indexed()
    logger.debug("Ensured index", filename=filename)
    return filename


def generate_batches(filename, max_bytes):
    batch_size = 4 * 1024 * 1024  # 4 MB
    for start in range(0, max_bytes, batch_size):
        # this range is intended since DFTracerJsonLinesBytesReader do
        # line boundary algorithm internally to chop incomplete line
        end = min(start + batch_size, max_bytes)
        logger.debug("Created batch", filename=filename, start=start, end=end)
        yield filename, start, end


def get_size(filename):
    size = 0
    if filename.endswith(".pfw"):
        size = os.stat(filename).st_size
    elif filename.endswith(".pfw.gz"):
        # dftracer-utils 0.0.12 moved get_max_bytes() off Indexer and onto the per-file
        # CheckpointIndexer, reached via Indexer.get_checkpoint_indexer(<path>). Verified to return
        # the exact UNCOMPRESSED byte count (753291066 for trace-3-of-4, byte-identical to the
        # original .pfw), which is what generate_batches needs to plan read batches.
        #
        # Deliberately NOT derived from the gzip ISIZE trailer: dftracer_pgzip writes MULTI-MEMBER
        # gzip (one independent member per 4 MB chunk, which is what makes parallel decode possible),
        # so the trailer describes only the final member, not the file.
        indexer = Indexer(files=[filename])
        size = indexer.get_checkpoint_indexer(filename).get_max_bytes()
    logger.debug("File has size", filename=filename, size=size / 1024**3)
    return filename, int(size)


def get_io_cat(func_name: str):
    if func_name in POSIX_METADATA_FUNCTIONS:
        return IOCategory.METADATA.value
    if func_name in POSIX_IO_CAT_MAPPING:
        return POSIX_IO_CAT_MAPPING[func_name].value
    return IOCategory.OTHER.value


def io_columns():
    columns = {
        "file_hash": "string",
        "host_hash": "string",
        "image_id": "Int64",
        "io_cat": "Int8",
        "size": "Int64",
        "offset": "Int64",
    }
    return columns


def io_function(json_dict: dict):
    d = {}
    d[COL_IO_CAT] = IOCategory.OTHER.value
    if "args" in json_dict:
        if "fhash" in json_dict["args"]:
            d["file_hash"] = str(json_dict["args"]["fhash"])
        if "size_sum" in json_dict["args"]:
            d["size"] = int(json_dict["args"]["size_sum"])
        elif json_dict.get("cat") in [CAT_POSIX, CAT_STDIO]:
            name = json_dict.get("name", "")
            io_cat = get_io_cat(name)
            if "ret" in json_dict["args"]:
                size = int(json_dict["args"]["ret"])
                if size > 0:
                    if io_cat in [IOCategory.READ.value, IOCategory.WRITE.value]:
                        d["size"] = size
            if "offset" in json_dict["args"]:
                offset = int(json_dict["args"]["offset"])
                if offset >= 0:
                    d["offset"] = offset
            d[COL_IO_CAT] = io_cat
        else:
            if "image_idx" in json_dict["args"]:
                image_id = int(json_dict["args"]["image_idx"])
                if image_id > 0:
                    d["image_id"] = image_id
            if "image_size" in json_dict["args"]:
                name = json_dict["name"].lower()
                # e.g. NPZReader.open image_size is not correct
                if "open" not in name:
                    size = int(json_dict["args"]["image_size"])
                    if size > 0:
                        d["size"] = size
    return d


def profile_function(json_dict: dict):
    args = json_dict.get("args", {})
    d = {}
    d[COL_IO_CAT] = IOCategory.OTHER.value
    if "fhash" in args:
        d["file_hash"] = str(args["fhash"])
    if "hhash" in args:
        d["host_hash"] = str(args["hhash"])
    if json_dict.get("cat") in [CAT_POSIX, CAT_STDIO]:
        d[COL_IO_CAT] = get_io_cat(json_dict["name"])
    for key in PROFILE_COLUMN_MAPPING:
        if key in args:
            d[key] = int(args[key])
    return d


def system_function(json_dict: dict):
    """Extract CPU/memory metric args from a cat='sys' ph='C' event."""
    args = json_dict.get("args", {})
    d = {}
    if "hhash" in args:
        d["host_hash"] = str(args["hhash"])
    for key in SYSTEM_COLUMN_MAPPING:
        if key in args:
            d[key] = float(args[key])
    return d


def _unflatten_args(rec):
    """Restore the nested ``args`` dict that dftracer-utils 0.0.12 flattens away.

    THE .pfw.gz "EMPTY ANALYSIS" BUG. read_lines_json() no longer returns the on-disk record
    shape. It hands back FLATTENED keys -- ``args.fhash``, ``args.hhash``, ``args.p_idx`` -- with
    the ``args`` key itself set to None, whereas the plain .pfw path (db.read_text -> json.loads)
    still yields the nested ``{"args": {...}}`` form.

    load_objects_dict() reads ``json_dict["args"]["fhash"]``. Against the flattened shape that
    lookup finds nothing and silently produces no file_hash/host_hash -- while name, cat and type
    keep working because they are top level. The result was an analysis that ran to completion and
    reported "Total Files 0 / Total Nodes 0" with an empty layer breakdown, against 6,146 files and
    1 node for the byte-identical uncompressed corpus. Silent, not an error, which is what made it
    dangerous.

    Normalising here -- at the reader boundary, where the schema difference actually is -- keeps
    both read paths feeding load_objects_dict() the same shape and leaves everything downstream
    untouched. None-valued keys are dropped rather than carried, because the nested form only ever
    contains keys that were present, and callers test membership (``"fhash" in args``).
    """
    if not isinstance(rec, dict) or rec.get("args") is not None:
        return rec
    args = {}
    flat = [k for k in rec if k.startswith("args.")]
    if not flat:
        return rec
    for k in flat:
        v = rec.pop(k)
        if v is not None:
            args[k[len("args."):]] = v
    rec["args"] = args
    return rec


def load_indexed_gzip_files(filename, start, end):
    # The Reader -> TraceReader rename is already handled by the import shim above, but 0.0.12 also
    # changed the CONSTRUCTOR and the read METHOD, which the shim does not cover:
    #   old: Reader(<trace>, <index file>).read_line_bytes_json(start, end)
    #   new: TraceReader(<trace>, index_dir=<DIR>).read_lines_json(start_byte=, end_byte=)
    # Passing the old `<file>.idx` as the second positional now lands in `index_dir`, i.e. a file
    # where a directory is expected -- the same trap as create_index/get_size. Omitting it lets the
    # reader find the `.dftindex` store beside the trace, which is where the indexer writes it.
    reader = Reader(filename)
    if hasattr(reader, "read_line_bytes_json"):  # dftracer-utils < 0.0.12
        json_lines = reader.read_line_bytes_json(start, end)
    else:                                        # dftracer-utils >= 0.0.12
        json_lines = reader.read_lines_json(start_byte=start, end_byte=end)
        json_lines = [_unflatten_args(r) for r in json_lines]
    logger.debug("Read json lines", filename=filename, start=start, end=end, num_lines=len(json_lines))
    return json_lines


def load_objects_dict(
    json_dict: dict,
    time_approximate: bool,
    extra_columns: Optional[Dict[str, str]],
    extra_columns_fn: Optional[Callable[[dict], dict]],
):
    final_dict = {}
    # logger.debug("Loading dict", json_dict=json_dict)
    if json_dict is not None:
        try:
            ph = json_dict.get("ph")
            if "name" in json_dict:
                final_dict["name"] = json_dict["name"]
            if "cat" in json_dict:
                final_dict["cat"] = json_dict["cat"].lower()
            if "pid" in json_dict:
                final_dict["pid"] = json_dict["pid"]
            if "tid" in json_dict:
                final_dict["tid"] = json_dict["tid"]
            if "args" in json_dict:
                if "hhash" in json_dict["args"]:
                    final_dict["host_hash"] = str(json_dict["args"]["hhash"])
                if (
                    "epoch" in json_dict["args"]
                    and json_dict["args"]["epoch"] != "train"
                    and json_dict["args"]["epoch"] != "valid"
                ):
                    epoch = int(json_dict["args"]["epoch"])
                    if epoch >= 0:
                        final_dict["epoch"] = epoch
                if "step" in json_dict["args"]:
                    step = int(json_dict["args"]["step"])
                    if step >= 0:
                        final_dict["step"] = step
            if "M" == ph:
                if "name" in final_dict:
                    if final_dict["name"] == "FH":
                        final_dict["type"] = TYPE_FILE_HASH
                        if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                            final_dict["name"] = json_dict["args"]["name"]
                            final_dict["hash"] = str(json_dict["args"]["value"])
                    elif final_dict["name"] == "HH":
                        final_dict["type"] = TYPE_HOST_HASH
                        if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                            final_dict["name"] = json_dict["args"]["name"]
                            final_dict["hash"] = str(json_dict["args"]["value"])
                    elif final_dict["name"] == "SH":
                        final_dict["type"] = TYPE_STRING_HASH
                        if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                            final_dict["name"] = json_dict["args"]["name"]
                            final_dict["hash"] = str(json_dict["args"]["value"])
                    elif final_dict["name"] == "PR":
                        final_dict["type"] = TYPE_PROC_METADATA
                        if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                            final_dict["name"] = json_dict["args"]["name"]
                            final_dict["hash"] = str(json_dict["args"]["value"])
                else:
                    final_dict["type"] = TYPE_METADATA
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["value"] = str(json_dict["args"]["value"])
            elif "C" == ph:
                is_system = json_dict.get("cat", "").lower() == "sys"
                final_dict["type"] = TYPE_SYSTEM if is_system else TYPE_PROFILE
                if "ts" in json_dict:
                    if type(json_dict["ts"]) is not int:
                        json_dict["ts"] = int(json_dict["ts"])
                    final_dict["ts"] = json_dict["ts"]
                if is_system:
                    final_dict.update(system_function(json_dict))
                else:
                    final_dict.update(profile_function(json_dict))
                    final_dict.update(extra_columns_fn(json_dict) if extra_columns_fn else {})
            else:
                final_dict["type"] = TYPE_EVENT
                if "dur" in json_dict and "ts" in json_dict:
                    if type(json_dict["dur"]) is not int:
                        json_dict["dur"] = int(json_dict["dur"])
                    if type(json_dict["ts"]) is not int:
                        json_dict["ts"] = int(json_dict["ts"])
                    final_dict["ts"] = json_dict["ts"]
                    final_dict["dur"] = json_dict["dur"]
                    final_dict["te"] = final_dict["ts"] + final_dict["dur"]
                    if not time_approximate:
                        final_dict["tinterval"] = I.to_string(
                            I.closed(json_dict["ts"], json_dict["ts"] + json_dict["dur"])
                        )
                final_dict.update(io_function(json_dict))
                if extra_columns and extra_columns_fn:
                    final_dict.update(extra_columns_fn(json_dict))
            # check if all extra columns are present
            if extra_columns and not all(col in final_dict for col in extra_columns):
                missing_cols = [col for col in extra_columns if col not in final_dict]
                raise ValueError(f"Missing extra columns: {missing_cols}")
            # logger.debug("Built a dictionary for dict", final_dict=final_dict)
            yield final_dict
        except ValueError as error:
            logger.error("Processing dict failed", dict=json_dict, error=error)
    return {}


def load_objects_str(
    line: str,
    time_approximate: bool,
    extra_columns: Optional[Dict[str, str]],
    extra_columns_fn: Optional[Callable[[dict], dict]],
):
    if line is not None and line != "" and len(line) > 0 and "[" != line[0] and "]" != line[0] and line != "\n":
        try:
            unicode_line = "".join([i if ord(i) < 128 else "#" for i in line])
            json_dict = json.loads(unicode_line, strict=False)
            yield from load_objects_dict(json_dict, time_approximate, extra_columns, extra_columns_fn)
        except ValueError as error:
            logger.error("Processing line failed", line=line, error=error)
    return {}


class DFTracerAnalyzer(Analyzer):
    def __init__(self, preset, assign_epochs=False, **kwargs):
        super().__init__(preset, **kwargs)
        self.assign_epochs = assign_epochs

    def read_trace(self, trace_path, extra_columns, extra_columns_fn):
        with log_block("glob_files"):
            pfw_pattern, pfw_gz_pattern = [], []
            if os.path.isdir(trace_path):
                pfw_pattern = glob.glob(os.path.join(trace_path, "*.pfw"))
                pfw_gz_pattern = glob.glob(os.path.join(trace_path, "*.pfw.gz"))
            elif trace_path.endswith(".pfw.gz"):
                pfw_gz_pattern = glob.glob(trace_path) if "*" in trace_path else [trace_path]
            elif trace_path.endswith(".pfw"):
                pfw_pattern = glob.glob(trace_path) if "*" in trace_path else [trace_path]
            all_files = pfw_pattern + pfw_gz_pattern
            if not all_files:
                raise FileNotFoundError("No matching .pfw or .pfw.gz files found.")
        logger.debug("Processing files", files=all_files)
        if len(pfw_gz_pattern) > 0:
            with log_block("create_index"):
                db.from_sequence(pfw_gz_pattern).map(create_index).compute()
                logger.info("Created index for files", num_files=len(pfw_gz_pattern))
        with log_block("sum_total_size"):
            sizes = db.from_sequence(all_files).map(get_size).compute()
            total_size = sum(size for _, size in sizes)
            logger.info("Total size of all files", total_size=total_size)
        gz_bag = None
        pfw_bag = None
        if len(pfw_gz_pattern) > 0:
            with log_block("gzip_index_and_batches"):
                logger.debug("Max bytes per file", sizes=sizes)
                json_line_delayed = []
                total_lines = 0
                # ONE task per gz file, NOT per 4 MB batch.
                #
                # The batched form silently DUPLICATED data. TraceReader.read_lines_json() in
                # dftracer-utils 0.0.12 streams from an internal cursor and ignores the
                # start_byte/end_byte it is handed; since each dask task constructs its own reader,
                # every batch restarted at byte 0 and returned the WHOLE file. Measured on a 20 MB
                # trace: 5 batches x 86,334 records = 431,670 records for a file that holds 86,334.
                # The 5x duplicate rows then broke the hash joins that resolve file and host names,
                # which is why the compressed run reported "Total Files 0" and an empty layer
                # breakdown while the uncompressed run reported 6,146 files -- same bytes, same
                # parser, different row multiplicity.
                #
                # Reading each file in a single call is the form that is actually verified: over the
                # full range the reader returns exactly the plain file's record count (3,131,587)
                # and metadata-record count (6,151 ph=="M"). Parallelism is preserved across FILES,
                # which is how the corpus is laid out anyway (one trace per rank), and the reader
                # already decodes gzip members in parallel internally.
                #
                # Only pfw_gz_pattern is walked here: `sizes` also carries plain .pfw entries, which
                # this branch must not claim -- those are handled by db.read_text below.
                _gz = set(pfw_gz_pattern)
                for filename, max_bytes in sizes:
                    if filename not in _gz:
                        continue
                    total_lines += max_bytes
                    json_line_delayed.append((filename, 0, max_bytes))

                logger.info(
                    "Loading batches",
                    num_batches=len(json_line_delayed),
                    num_files=len(pfw_gz_pattern),
                    total_lines=total_lines,
                )
                json_line_bags = []
                for filename, start, end in json_line_delayed:
                    json_line_bags.append(dask.delayed(load_indexed_gzip_files)(filename, start, end))
                json_lines = db.concat(json_line_bags)
            with log_block("parse_gzip_json_lines"):
                gz_bag = (
                    json_lines.map(
                        load_objects_dict,
                        time_approximate=self.time_approximate,
                        extra_columns=extra_columns,
                        extra_columns_fn=extra_columns_fn,
                    )
                    .flatten()
                    .filter(lambda x: "name" in x)
                )
        main_bag = None
        if len(pfw_pattern) > 0:
            with log_block("parse_json_lines"):
                pfw_bag = (
                    db.read_text(pfw_pattern)
                    .map(
                        load_objects_str,
                        time_approximate=self.time_approximate,
                        extra_columns=extra_columns,
                        extra_columns_fn=extra_columns_fn,
                    )
                    .flatten()
                    .filter(lambda x: "name" in x)
                )
        if len(pfw_gz_pattern) > 0 and len(pfw_pattern) > 0:
            main_bag = db.concat([pfw_bag, gz_bag])
        elif len(pfw_gz_pattern) > 0:
            main_bag = gz_bag
        elif len(pfw_pattern) > 0:
            main_bag = pfw_bag
        if main_bag:
            self._columns = self._get_columns(extra_columns)
            with log_block("to_dataframe"):
                raw_traces = main_bag.to_dataframe(meta=self._columns)
            with log_block("_handle_metadata"):
                _hash_tables = (
                    self._index_hash_tables(pfw_gz_pattern) if len(pfw_gz_pattern) > 0 else None
                )
                traces, profiles, system_events = self._handle_metadata(
                    raw_traces, hash_tables=_hash_tables
                )
            with log_block("compute_time_origin"):
                trace_min, profile_min, system_min = dask.compute(
                    traces["ts"].min(), profiles["ts"].min(), system_events["ts"].min()
                )
                time_origin_candidates = [
                    ts for ts in [trace_min, profile_min, system_min] if pd.notna(ts)
                ]
                time_origin = min(time_origin_candidates) if time_origin_candidates else 0
                has_profiles = pd.notna(profile_min)
                has_system = pd.notna(system_min)
                if has_profiles:
                    # DFTracer counter buckets are emitted on absolute 5s boundaries,
                    # while trace_min is arbitrary. Snap the shared origin down to the
                    # 5s profile grid so a single 5s profile bucket cannot straddle two
                    # analyzer bins and get assigned to only one time_range.
                    profile_grid_width = int(self.profile_time_granularity * self.time_resolution)
                    time_origin = (time_origin // profile_grid_width) * profile_grid_width
            self._npartitions = math.ceil(total_size / (128 * 1024**2))
            logger.debug(f"Number of partitions used are {self._npartitions}")
            with log_block("repartition+persist"):
                traces = traces.repartition(npartitions=self._npartitions).persist()
                if has_profiles:
                    profiles = profiles.repartition(npartitions=self._npartitions).persist()
                else:
                    profiles = None
            with log_block("normalize_records+persist"):
                traces = self._fix_time(traces, time_origin=time_origin).persist()
                if profiles is not None:
                    profiles = self._standardize_profiles(profiles, time_origin=time_origin).persist()
                if has_system:
                    system_metrics = self._standardize_system(system_events, time_origin=time_origin).persist()
                    logger.info("system_events.found", has_system=True)
                else:
                    system_metrics = None
                    logger.info("system_events.found", has_system=False)
            with log_block("wait_all"):
                if isinstance(traces, dd.DataFrame):
                    wait_list = [traces, self._file_hashes, self._host_hashes, self._string_hashes, self._metadata]
                    if profiles is not None:
                        wait_list.append(profiles)
                    if system_metrics is not None:
                        wait_list.append(system_metrics)
                    wait(wait_list)
        else:
            logger.error("Unable to load traces")
            exit(1)
        return ReadTraceResult(
            traces=self._rename_columns(traces),
            profiles=profiles,
            profile_time_granularity=self.profile_time_granularity if profiles is not None else None,
            system_metrics=system_metrics,
        )

    def postread_trace(self, traces, view_types):
        # print("Post-reading trace", traces)
        # print("Post-reading trace columns", traces.columns)
        is_dask = isinstance(traces, dd.DataFrame)

        if not is_dask and traces.empty:
            logger.warning("No traces found for postread_trace")
            return traces

        # Ignore redundant files
        with log_block("filter_files"):
            if COL_FILE_NAME in traces.columns:
                traces = traces[
                    traces[COL_FILE_NAME].isna()
                    | ~traces[COL_FILE_NAME].str.contains("|".join(IGNORED_FILE_PATTERNS), na=False)
                ]
            else:
                traces[COL_FILE_NAME] = traces[COL_FILE_HASH].astype(str).replace("nan", "")

        # Ignore redundant function calls
        with log_block("filter_functions"):
            traces = traces[~traces[COL_FUNC_NAME].isin(IGNORED_FUNC_NAMES)]
            traces = traces[~traces[COL_FUNC_NAME].str.contains("|".join(IGNORED_FUNC_PATTERNS))]

        # Set epochs
        with log_block("assign_epochs"):
            if self.assign_epochs:
                if "epoch" not in self.preset.layer_defs:
                    raise ValueError("Epoch layer definition is missing")
                epochs = traces.query(self.preset.layer_defs["epoch"])
                if is_dask:
                    epochs = epochs.compute()
                epochs_with_index = epochs.sort_values(["pid", "time_start"]).reset_index(drop=True)
                epochs_with_index["epoch"] = epochs_with_index.groupby("pid").cumcount() + 1
                epoch_boundaries = epochs_with_index[["pid", "time_start", "time_end", "epoch"]]
                if is_dask:
                    traces = traces.map_partitions(self._set_epochs, epoch_boundaries=epoch_boundaries)
                else:
                    traces = self._set_epochs(traces, epoch_boundaries=epoch_boundaries)

        with log_block("wait"):
            if is_dask:
                wait(traces)

        with log_block("set_basic_columns"):
            traces[COL_ACC_PAT] = 0
            traces[COL_COUNT] = 1

        if is_dask:
            traces = (
                traces.map_partitions(self._set_proc_names)
                .map_partitions(self._fix_file_posix_category, self._file_purposes())
                .map_partitions(self._sanitize_size)
            )
        else:
            traces = self._set_proc_names(traces)
            traces = self._fix_file_posix_category(traces, self._file_purposes())
            traces = self._sanitize_size(traces)

        return traces

    def normalize_trace_event(
        self,
        event: dict,
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
    ) -> dict:
        # logger.debug(
        #     "stream.normalize_input",
        #     name=event.get("name"),
        #     ph=event.get("ph"),
        #     args_name=event.get("args", {}).get("name"),
        # )
        normalized_event = next(
            load_objects_dict(
                event,
                time_approximate=self.time_approximate,
                extra_columns=extra_columns,
                extra_columns_fn=extra_columns_fn,
            )
        )
        return normalized_event

    def handle_stream_events(
        self,
        events: List[dict],
        view_types: List[ViewType],
        extra_columns: Optional[Dict[str, str]] = None,
    ):
        columns = self._get_columns(extra_columns)
        df = pd.DataFrame(events, columns=columns)
        traces, profile_events, system_events = self._handle_metadata(df)

        # Compute shared time_origin (same as batch read_trace) so that
        # traces, profiles, and system_metrics share aligned time_range values.
        ts_candidates = []
        if not traces.empty and "ts" in traces.columns:
            ts_candidates.append(traces["ts"].min())
        if not profile_events.empty and "ts" in profile_events.columns:
            ts_candidates.append(profile_events["ts"].min())
        if not system_events.empty and "ts" in system_events.columns:
            ts_candidates.append(system_events["ts"].min())
        ts_candidates = [ts for ts in ts_candidates if pd.notna(ts)]
        time_origin = int(min(ts_candidates)) if ts_candidates else 0

        # Snap time_origin to profile grid (same as batch read_trace)
        if not profile_events.empty:
            profile_grid_width = int(self.profile_time_granularity * self.time_resolution)
            if profile_grid_width > 0:
                time_origin = (time_origin // profile_grid_width) * profile_grid_width

        # Standardize traces
        traces = self._fix_time(traces, time_origin=time_origin)
        traces = self._rename_columns(traces)
        traces = set_size_bins(traces)
        traces = self.postread_trace(traces=traces, view_types=view_types)
        # THIS SUBCLASS OVERRIDES handle_stream_events, so the base class's call to
        # assign_window_epochs never ran and every POSIX read stayed epoch-less. Measured on
        # t4m_storagegate2: node_read_bytes and node_read_busy_s arrived at the planner on all 124
        # decisions and every one of them was 0.0, which is the same shape as the metric being
        # absent and is exactly the failure mode the whole barrier-attribution wire was audited
        # for. The call belongs on both paths into the frame, not on the one the streaming agent
        # does not take.
        traces = self.assign_window_epochs(traces)

        # Standardize profiles (same pipeline as batch _standardize_profiles,
        # but calling each step directly on pandas instead of map_partitions).
        profiles = None
        if not profile_events.empty:
            p = self._set_proc_names(profile_events)
            p = self._standardize_profile_partition(
                p,
                profile_time_granularity=self.profile_time_granularity,
                time_origin=time_origin,
                time_granularity=self.time_granularity,
                time_resolution=self.time_resolution,
            )
            if not p.empty:
                p = self._fix_file_posix_category(p, self._file_purposes())
                p = self._sanitize_size_offset(p)
                # Coalesce: collapse duplicate counter rows per canonical bucket
                coalesce_agg = {
                    COL_COUNT: "sum", COL_TIME: "sum", COL_SIZE: "sum",
                    "time_min": "min", "time_max": "max",
                    "size_min": "min", "size_max": "max",
                    "offset_min": "min", "offset_max": "max",
                }
                p = (
                    p.groupby(PROFILE_IDENTITY_COLUMNS, dropna=False)
                    .agg(coalesce_agg)
                    .reset_index()
                )
                p[COL_COUNT] = p[COL_COUNT].astype("Int64")
                p[COL_TIME] = p[COL_TIME].astype("float64")
                p[COL_SIZE] = p[COL_SIZE].replace(0, pd.NA).astype("Int64")
                profiles = p[list(PROFILE_OUTPUT_COLUMNS)]

        # Standardize system metrics (same static method the batch path uses)
        system_metrics = None
        if not system_events.empty:
            system_metrics = self._standardize_system_partition(
                system_events,
                time_origin=time_origin,
                time_granularity=self.time_granularity,
                time_resolution=self.time_resolution,
            )
            if system_metrics.empty:
                system_metrics = None

        return traces, profiles, system_metrics

    def get_job_time(self, traces):
        return super().get_job_time(traces) / self.time_resolution

    def get_unique_file_count(self, traces: dd.DataFrame):
        return traces["file_hash"].nunique()

    def get_unique_host_count(self, traces: dd.DataFrame):
        return traces["host_hash"].nunique()

    def get_unique_process_count(self, traces: dd.DataFrame):
        return traces["pid"].nunique()

    def _file_purposes(self) -> Dict[str, str]:
        """The preset's path-fragment -> category-suffix map, or a clear error if it was not set.

        The DL presets declare `file_purposes` as MISSING, so omegaconf raises on access when the
        caller forgot it. That exception names the config key but not the consequence, and the
        consequence is what matters: no purpose suffixes means the reader_posix layer is empty and
        every rule keyed on it is silently false. Say that here, once, at startup, instead of
        letting it surface as an unexplained zero six steps downstream.
        """
        try:
            purposes = self.preset.file_purposes
        except Exception as exc:  # omegaconf MissingMandatoryValue, among others
            raise ValueError(
                f"preset {getattr(self.preset, 'name', '?')!r} requires `file_purposes`, a map of "
                "path fragment -> category suffix such as "
                '{"/p/lustre1/me/dataset/train": "_reader", "/p/lustre1/me/ckpt": "_checkpoint"}. '
                "It has no default because only you know where this run's data lives, and a wrong "
                "guess empties the reader_posix layer without any error. Pass the run's actual "
                "data and checkpoint folders."
            ) from exc
        return dict(purposes) if purposes else {}

    @staticmethod
    def _fix_file_posix_category(df: pd.DataFrame, file_purposes: Dict[str, str]):
        base_condition = df["cat"].str.contains("posix|stdio") & ~df["file_name"].isna()

        # Step 1: Map file purpose suffixes first.
        #
        # `file_purposes` comes from the preset and is MANDATORY for the DL presets. It used to be
        # hardcoded here as {"/data": "_reader", "/checkpoint": "_checkpoint"}, which silently tied
        # the whole reader_posix layer to a dataset happening to live under a directory named
        # "data". See AnalyzerPresetConfigDLIO.file_purposes for the failure this caused.
        for path, suffix in (file_purposes or {}).items():
            mask = base_condition & df["file_name"].str.contains(path, regex=False)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix

        # A partition full of POSIX operations where NOT ONE matched a purpose is the exact shape
        # of the silent failure above: the layer is not wrong, it is empty, and every rule keyed on
        # it evaluates to false without a single log line. Say so, once per process.
        global _PURPOSE_MATCH_WARNED
        if not _PURPOSE_MATCH_WARNED and file_purposes:
            posix_rows = int(base_condition.sum())
            if posix_rows > 0:
                matched = int(
                    (
                        base_condition
                        & df["file_name"].str.contains(
                            "|".join(re.escape(p) for p in file_purposes), regex=True
                        )
                    ).sum()
                )
                if matched == 0:
                    _PURPOSE_MATCH_WARNED = True
                    logger.warning(
                        "posix.file_purpose.no_match",
                        posix_rows=posix_rows,
                        file_purposes=dict(file_purposes),
                        example_file=str(df.loc[base_condition, "file_name"].iloc[0]),
                        hint="no POSIX op matched any file_purposes path, so purpose layers such "
                        "as reader_posix will be EMPTY and every rule keyed on them will be "
                        "silently false. Check the paths passed for data/checkpoint folders.",
                    )

        # Step 2: Map filesystem suffixes
        filesystem_updates = {"/lustre": "_lustre", "/ssd": "_ssd"}

        for path, suffix in filesystem_updates.items():
            mask = base_condition & df["file_name"].str.contains(path)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix

        return df

    def _fix_time(self, traces: dd.DataFrame, time_origin: Optional[int] = None) -> dd.DataFrame:
        time_origin = traces["ts"].min() if time_origin is None else time_origin
        traces["ts"] = traces["ts"] - time_origin
        traces["te"] = traces["ts"] + traces["dur"]
        traces["trange"] = traces["ts"] // int(self.time_granularity * self.time_resolution)
        traces["ts"] = traces["ts"].astype("Int64")
        traces["te"] = traces["te"].astype("Int64")
        traces["trange"] = traces["trange"].fillna(0).astype("Int64")
        traces["dur"] = traces["dur"] / self.time_resolution
        logger.debug(
            "Fixed time columns",
            time_granularity=self.time_granularity,
            time_resolution=self.time_resolution,
        )
        return traces

    def _standardize_profiles(self, profiles: dd.DataFrame, time_origin: int) -> dd.DataFrame:
        profiles = profiles.map_partitions(self._set_proc_names)
        profiles = profiles.map_partitions(
            self._standardize_profile_partition,
            profile_time_granularity=self.profile_time_granularity,
            time_granularity=self.time_granularity,
            time_origin=time_origin,
            time_resolution=self.time_resolution,
            meta=PROFILE_OUTPUT_COLUMNS,
        )
        profiles = profiles.map_partitions(
            self._fix_file_posix_category, self._file_purposes()
        ).map_partitions(self._sanitize_size_offset)
        return self._coalesce_profiles(profiles)

    def _coalesce_profiles(self, profiles: dd.DataFrame) -> dd.DataFrame:
        # dft-agg-full can emit multiple counter rows for the same canonical
        # profile bucket. Collapse them here so `read_trace()` returns a stable
        # analyzer-native profile table.
        split_out = max(1, math.ceil(math.sqrt(profiles.npartitions)))
        coalesced = (
            profiles.groupby(PROFILE_IDENTITY_COLUMNS, dropna=False)
            .agg(
                {
                    COL_COUNT: "sum",
                    COL_TIME: "sum",
                    COL_SIZE: "sum",
                    "time_min": "min",
                    "time_max": "max",
                    "size_min": "min",
                    "size_max": "max",
                    "offset_min": "min",
                    "offset_max": "max",
                },
                split_out=split_out,
            )
            .reset_index()
        )
        coalesced[COL_COUNT] = coalesced[COL_COUNT].astype("Int64")
        coalesced[COL_TIME] = coalesced[COL_TIME].astype("float64")
        coalesced[COL_SIZE] = coalesced[COL_SIZE].replace(0, pd.NA).astype("Int64")
        return coalesced[list(PROFILE_OUTPUT_COLUMNS)]

    @staticmethod
    def _standardize_profile_partition(
        df: pd.DataFrame,
        profile_time_granularity: float,
        time_origin: int,
        time_granularity: float,
        time_resolution: float,
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in PROFILE_OUTPUT_COLUMNS.items()})

        df = df.copy()
        duration = df["dur_sum"].where(df["dur_sum"].notna(), df["dur"]).fillna(0)
        size = df["ret_sum"].where(df["ret_sum"].notna(), df["ret"])
        is_sized_io = df[COL_IO_CAT].isin([IOCategory.READ.value, IOCategory.WRITE.value]) & size.notna() & (size > 0)

        profile_df = pd.DataFrame(index=df.index)
        profile_df["cat"] = df["cat"].astype("string")
        profile_df[COL_FUNC_NAME] = df["name"].astype("string")
        profile_df["pid"] = df["pid"].astype("Int64")
        profile_df["tid"] = df["tid"].astype("Int64")
        profile_df["epoch"] = df["epoch"].astype("Int64")
        profile_df["step"] = df["step"].astype("Int64")
        profile_df["file_hash"] = df["file_hash"].astype("string")
        profile_df["host_hash"] = df["host_hash"].astype("string")
        profile_df[COL_FILE_NAME] = df[COL_FILE_NAME].astype("string")
        profile_df[COL_HOST_NAME] = df[COL_HOST_NAME].astype("string")
        profile_df[COL_PROC_NAME] = df[COL_PROC_NAME].astype("string")
        profile_df[COL_IO_CAT] = df[COL_IO_CAT].fillna(IOCategory.OTHER.value).astype("Int8")
        profile_df[COL_ACC_PAT] = pd.Series(0, index=df.index, dtype="Int8")
        profile_df[COL_COUNT] = df["dft_cnt"].fillna(0).astype("Int64")
        profile_df[COL_TIME] = duration.astype("float64") / time_resolution
        profile_df[COL_SIZE] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        profile_df.loc[is_sized_io, COL_SIZE] = size.loc[is_sized_io].astype("Int64")
        dur_min = df["dur_min"].where(df["dur_min"].notna(), df["dur"])
        profile_df["time_min"] = dur_min.astype("float64") / time_resolution
        dur_max = df["dur_max"].where(df["dur_max"].notna(), df["dur"])
        profile_df["time_max"] = dur_max.astype("float64") / time_resolution
        profile_df["size_min"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        profile_df["size_max"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        profile_df.loc[is_sized_io, "size_min"] = (
            df["ret_min"].where(df["ret_min"].notna(), df["ret"]).loc[is_sized_io].astype("Int64")
        )
        profile_df.loc[is_sized_io, "size_max"] = (
            df["ret_max"].where(df["ret_max"].notna(), df["ret"]).loc[is_sized_io].astype("Int64")
        )
        profile_df["offset_min"] = df["offset_min"].where(df["offset_min"].notna(), df["offset"]).astype("Int64")
        profile_df["offset_max"] = df["offset_max"].where(df["offset_max"].notna(), df["offset"]).astype("Int64")
        profile_df[COL_TIME_START] = (df["ts"] - time_origin).astype("Int64")
        profile_df[COL_TIME_END] = profile_df[COL_TIME_START] + int(profile_time_granularity * time_resolution)
        profile_df[COL_TIME_RANGE] = (
            profile_df[COL_TIME_START] // int(time_granularity * time_resolution)
        ).astype("Int64")
        return profile_df[list(PROFILE_OUTPUT_COLUMNS)]

    @staticmethod
    def _standardize_system_partition(
        df: pd.DataFrame,
        time_origin: int,
        time_granularity: float,
        time_resolution: float,
    ) -> pd.DataFrame:
        """Aggregate raw system events into per-time_range system metric rows."""
        empty = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in SYSTEM_OUTPUT_COLUMNS.items()})
        if df.empty:
            return empty

        df = df.copy()
        bucket_width_us = int(time_granularity * time_resolution)
        df[COL_TIME_RANGE] = ((df["ts"] - time_origin) // bucket_width_us).astype("Int64")

        group_keys = ["host_hash", COL_TIME_RANGE]

        # Aggregate CPU (name == "cpu"): mean of samples per bucket
        # Source values are 0-100; normalize to 0-1 fractions.
        agg_cpu = df[df["name"] == "cpu"]
        cpu_agg = pd.DataFrame()
        if not agg_cpu.empty:
            agg_dict = {}
            for m, out in [("iowait_pct", "sys_cpu_iowait_pct"),
                           ("user_pct", "sys_cpu_user_pct"),
                           ("system_pct", "sys_cpu_system_pct"),
                           ("idle_pct", "sys_cpu_idle_pct")]:
                if m in agg_cpu.columns:
                    agg_dict[out] = (m, "mean")
            if agg_dict:
                cpu_agg = agg_cpu.groupby(group_keys).agg(**agg_dict).reset_index()
                for col in ["sys_cpu_iowait_pct", "sys_cpu_user_pct",
                            "sys_cpu_system_pct", "sys_cpu_idle_pct"]:
                    if col in cpu_agg.columns:
                        cpu_agg[col] = cpu_agg[col] / 100.0

        # Per-core cross-core stats (name starts with "cpu-")
        # Source values are 0-100; normalize to 0-1 fractions.
        per_core = df[df["name"].str.startswith("cpu-")]
        core_agg = pd.DataFrame()
        if not per_core.empty and "iowait_pct" in per_core.columns:
            core_agg = per_core.groupby(group_keys).agg(
                sys_cpu_iowait_pct_core_max=("iowait_pct", "max"),
                sys_cpu_iowait_pct_core_p95=("iowait_pct", lambda x: x.quantile(0.95)),
            ).reset_index()
            for col in ["sys_cpu_iowait_pct_core_max", "sys_cpu_iowait_pct_core_p95"]:
                if col in core_agg.columns:
                    core_agg[col] = core_agg[col] / 100.0

        # Memory (name == "memory"): mean of samples per bucket
        # MemAvailable is raw KB — convert to fraction of MemTotal.
        # Cached/Dirty are % of MemAvailable (0-100) — normalize to 0-1.
        mem = df[df["name"] == "memory"]
        mem_agg = pd.DataFrame()
        if not mem.empty:
            mem_dict = {}
            for m, out in [("Dirty", "sys_mem_dirty_pct"),
                           ("Cached", "sys_mem_cached_pct"),
                           ("MemAvailable", "sys_mem_available_kb")]:
                if m in mem.columns:
                    mem_dict[out] = (m, "mean")
            if mem_dict:
                mem_agg = mem.groupby(group_keys).agg(**mem_dict).reset_index()
                # Normalize Dirty/Cached from 0-100 to 0-1
                for col in ["sys_mem_dirty_pct", "sys_mem_cached_pct"]:
                    if col in mem_agg.columns:
                        mem_agg[col] = mem_agg[col] / 100.0
                # Derive sys_mem_available_pct from KB value
                if "sys_mem_available_kb" in mem_agg.columns:
                    try:
                        with open("/proc/meminfo") as _f:
                            for _line in _f:
                                if _line.startswith("MemTotal:"):
                                    mem_total_kb = int(_line.split()[1])
                                    break
                            else:
                                mem_total_kb = None
                    except OSError:
                        mem_total_kb = None
                    if mem_total_kb and mem_total_kb > 0:
                        mem_agg["sys_mem_available_pct"] = mem_agg["sys_mem_available_kb"] / mem_total_kb
                    else:
                        mem_agg["sys_mem_available_pct"] = pd.NA

        # Merge all on (host_hash, time_range)
        dfs = [d for d in [cpu_agg, core_agg, mem_agg] if not d.empty]
        if not dfs:
            return empty

        result = dfs[0]
        for d in dfs[1:]:
            result = result.merge(d, on=group_keys, how="outer")

        for col, dtype in SYSTEM_OUTPUT_COLUMNS.items():
            if col not in result.columns:
                result[col] = pd.Series(dtype=dtype)
            result[col] = result[col].astype(dtype)
        return result[list(SYSTEM_OUTPUT_COLUMNS)]

    def _standardize_system(self, system_events: dd.DataFrame, time_origin: int) -> dd.DataFrame:
        """Standardize raw system events into per-time_range metrics."""
        meta = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in SYSTEM_OUTPUT_COLUMNS.items()})
        return system_events.map_partitions(
            self._standardize_system_partition,
            time_origin=time_origin,
            time_granularity=self.time_granularity,
            time_resolution=self.time_resolution,
            meta=meta,
        )

    def _get_columns(self, extra_columns: Optional[Dict[str, str]]):
        columns = {
            "name": "string",
            "cat": "string",
            "type": "Int8",
            "pid": "Int64",
            "tid": "Int64",
            "ts": "Int64",
            "te": "Int64",
            "dur": "Int64",
            "window": "Int64",
            "epoch": "Int64",
            "step": "Int64",
            "tinterval": "Int64" if self.time_approximate else "string",
            "trange": "Int64",
            # Neither trace nesting nor a live control-window identity is
            # bounded by an 8-bit signed integer.  Keep the trace schema wide
            # enough that the streaming path cannot fail once either exceeds
            # 127.
            "level": "Int32",
        }
        metadata_columns = {
            "hash": "string",
            "host_hash": "string",
            "value": "string",
        }
        columns.update(io_columns())
        columns.update(PROFILE_COLUMN_MAPPING)
        columns.update(SYSTEM_COLUMN_MAPPING)
        columns.update(metadata_columns)
        columns.update(extra_columns or {})
        logger.debug("get_columns", columns=columns)
        return columns

    @staticmethod
    def _index_hash_tables(gz_files):
        """Read the file/host/string hash tables straight out of the C++ index.

        Ported from upstream dfanalyzer (origin/main, "feat(analyzer): integrate to new
        dftracer-utils"), which builds these from `Indexer.get_hash_table(...)` rather than by
        joining the "M" metadata records out of the JSON stream.

        WHY. On the .pfw.gz path the M-record join yields nothing, so `file_name` and `host_name`
        never resolve: a compressed corpus analysed to "Total Files 0 / Total Nodes 0" and an empty
        layer breakdown, where the byte-identical uncompressed corpus reported 6,146 files and 1
        node. The index already holds these mappings -- verified on a 20 MB trace: 6,147 file
        entries and 1 host entry, matching the uncompressed run -- so taking them from there is both
        correct and cheaper than re-deriving them from the event stream.

        Returns (file_hashes, host_hashes, string_hashes) as hash-indexed frames, or None.
        """
        try:
            ix = Indexer(files=list(gz_files))
            ix.ensure_indexed()

            def _frame(d):
                return pd.DataFrame(
                    {"name": list(d.values())},
                    index=pd.Index(list(d.keys()), name="hash", dtype="string"),
                )

            return (
                _frame(ix.get_hash_table("file")),
                _frame(ix.get_hash_table("host")),
                _frame(ix.get_hash_table("string")),
            )
        except Exception as e:  # index unavailable -> fall back to the M-record join
            logger.warning("index hash tables unavailable", error=str(e))
            return None

    def _handle_metadata(self, raw_traces: dd.DataFrame, hash_tables=None) -> Tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        is_dask = isinstance(raw_traces, dd.DataFrame)

        # `DataFrame.query` rebuilds the frame's column resolvers on every call -- about 10 ms
        # against this ~30-column frame -- and this method makes SEVEN of them on the same column.
        # On the streaming path that is ~50 ms of every window spent parsing seven expressions
        # whose answer is one integer comparison. The dask path keeps query, where the expression
        # is compiled once into the graph rather than per call.
        if is_dask:
            def _of_type(type_id):
                return raw_traces.query(f"type == {type_id}")
        else:
            _types = raw_traces["type"]

            def _of_type(type_id):
                return raw_traces[(_types == type_id).fillna(False).to_numpy()]

        traces = _of_type(TYPE_EVENT)
        profile_events = _of_type(TYPE_PROFILE)
        system_events = _of_type(TYPE_SYSTEM)
        file_hashes = _of_type(TYPE_FILE_HASH)[["name", "hash"]].groupby("hash").first()
        host_hashes = _of_type(TYPE_HOST_HASH)[["name", "hash"]].groupby("hash").first()
        string_hashes = _of_type(TYPE_STRING_HASH)[["name", "hash"]].groupby("hash").first()
        metadata = _of_type(TYPE_METADATA)[["name", "value"]]
        file_hashes.index = file_hashes.index.astype(str)
        host_hashes.index = host_hashes.index.astype(str)
        string_hashes.index = string_hashes.index.astype(str)
        if is_dask:
            file_hashes = file_hashes.persist()
            host_hashes = host_hashes.persist()
            string_hashes = string_hashes.persist()
            metadata = metadata.persist()
        # Index-derived tables win over the M-record join when the join came back empty, which is
        # what happens for .pfw.gz input (see _index_hash_tables). Only substituted when the join
        # produced nothing, so the plain .pfw path is bit-for-bit unchanged.
        if hash_tables is not None:
            idx_file, idx_host, idx_string = hash_tables

            def _prefer_index(existing, idx, label):
                """Index table wins; anything only the M-record join knows about is kept.

                The index is authoritative -- upstream's C++ reader builds these tables from it and
                does no M-record join at all. Substituting only when the join looked EMPTY was not
                enough: on the gz path the join returns rows whose key does not match the traces'
                `file_hash`, so `file_name` stayed unresolved while the table looked populated.
                Union rather than replace so a mixed .pfw + .pfw.gz corpus keeps the hashes that
                only the plain files carry.
                """
                if idx is None or not len(idx.index):
                    return existing
                try:
                    cur = existing.compute() if hasattr(existing, "compute") else existing
                except Exception:
                    cur = None
                idx = idx.copy()
                idx.index = idx.index.astype(str)
                if cur is None or len(cur.index) == 0:
                    logger.info(f"using index-derived {label} hashes", count=len(idx.index))
                    return idx
                cur.index = cur.index.astype(str)
                extra = cur[~cur.index.isin(idx.index)]
                out = pd.concat([idx, extra]) if len(extra.index) else idx
                logger.info(
                    f"using index-derived {label} hashes",
                    count=len(idx.index),
                    kept_from_json=len(extra.index),
                )
                return out

            file_hashes = _prefer_index(file_hashes, idx_file, "file")
            host_hashes = _prefer_index(host_hashes, idx_host, "host")
            string_hashes = _prefer_index(string_hashes, idx_string, "string")
        # THE HASH TABLES MUST SURVIVE THE WINDOW, on the streaming path only.
        #
        # dftracer emits an FH record ONCE, when a file is first touched. Measured on
        # t4m_storagegate4: all 504 FH records of a lane arrive in the first 7179 lines and the
        # remaining 342,252 preads carry only a hash. A per-window drain is therefore a frame full
        # of reads whose file_hash resolves to nothing, `file_name` stays NA, and
        # `_fix_file_posix_category` cannot tag a single row -- so `reader_posix` is EMPTY in every
        # window after the first, which is why node_read_bytes reached the planner as 0.0 on all
        # 125 windows of that run while the broad `posix` layer carried 11.56 GB per window.
        #
        # The batch path never saw this because it reads the whole trace at once. Accumulating is
        # the minimum that makes a window-at-a-time reader equivalent to it. The tables are small
        # (one row per file, per host, per string) and bounded by the corpus, not by the run.
        if not is_dask:
            file_hashes = self._merge_seen_hashes("_seen_file_hashes", file_hashes)
            host_hashes = self._merge_seen_hashes("_seen_host_hashes", host_hashes)
            string_hashes = self._merge_seen_hashes("_seen_string_hashes", string_hashes)

        # DFTracer builds that write the host hash ONLY in the per-process HH/thread_name metadata
        # records (the PyPI dftracer 1.0.15 in the SFNO venv; events carry no `args` at all) leave
        # every event's host_hash NA, so unique_host_count is 0 and all views collapse to zeros
        # (the "still OPEN" host-metadata failure documented in _assert_metadata_resolved).
        # Events that already carry args.hhash (the project's SHM/STDIO build) are untouched: only
        # NA host_hash values are filled, from the host hash of the SAME pid's metadata records.
        pid_hosts = self._pid_host_table(raw_traces, is_dask)
        traces = self._fill_host_hash_from_pid(traces, pid_hosts)
        profile_events = self._fill_host_hash_from_pid(profile_events, pid_hosts)
        traces = self._attach_metadata(traces, file_hashes=file_hashes, host_hashes=host_hashes)
        profile_events = self._attach_metadata(profile_events, file_hashes=file_hashes, host_hashes=host_hashes)
        self._file_hashes = file_hashes
        self._host_hashes = host_hashes
        self._string_hashes = string_hashes
        self._metadata = metadata
        self._profile_events = profile_events
        self._system_events = system_events
        return traces, profile_events, system_events

    def _merge_seen_hashes(self, attr: str, table):
        """Union this window's hash table with every one seen before, newest winning.

        Streaming only. A hash maps to one name for the life of a run, so a union cannot conflict;
        the new table is preferred on a collision purely so a re-registered hash takes the fresher
        name. Kept per analyzer instance, which is per node agent, which is the same lifetime as
        the run it is watching.
        """
        prev = getattr(self, attr, None)
        if table is None or not len(getattr(table, "index", ())):
            merged = prev if prev is not None else table
        elif prev is None or not len(prev.index):
            merged = table
        else:
            extra = prev[~prev.index.isin(table.index)]
            merged = pd.concat([table, extra]) if len(extra.index) else table
        setattr(self, attr, merged)
        return merged

    @staticmethod
    def _pid_host_table(raw_traces, is_dask: bool):
        """pid -> host_hash from metadata (M) records that carry args.hhash, as a pandas Series."""
        if "host_hash" not in raw_traces.columns or "pid" not in raw_traces.columns:
            return None
        meta_types = [TYPE_FILE_HASH, TYPE_HOST_HASH, TYPE_STRING_HASH, TYPE_METADATA, TYPE_PROC_METADATA]
        m = raw_traces[raw_traces["type"].isin(meta_types)][["pid", "host_hash"]]
        m = m.dropna()
        if is_dask:
            m = m.compute()
        if not len(m.index):
            return None
        m = m.assign(host_hash=m["host_hash"].astype("string")).drop_duplicates("pid", keep="first")
        return m.set_index(m["pid"].astype("int64"))["host_hash"]

    @staticmethod
    def _fill_host_hash_from_pid(records, pid_hosts):
        """Fill NA host_hash of `records` from pid_hosts; rows that already have one keep it."""
        if pid_hosts is None or "host_hash" not in records.columns or "pid" not in records.columns:
            return records
        records = records.copy()
        host = records["host_hash"].astype("string")
        mapped = records["pid"].astype("int64").map(pid_hosts).astype("string") if not isinstance(records, dd.DataFrame) \
            else records["pid"].astype("int64").map(pid_hosts, meta=("host_hash", "string")).astype("string")
        records["host_hash"] = host.fillna(mapped)
        return records

    @staticmethod
    def _attach_metadata(records: dd.DataFrame, file_hashes: dd.DataFrame, host_hashes: dd.DataFrame):
        # Ensure merge key dtypes match (file_hash/host_hash may be float64
        # when all values are NaN, e.g. AI-only events with POSIX disabled).
        #
        # USE THE NULLABLE "string" DTYPE, NOT `astype(str)`.
        #
        # `astype(str)` is version-dependent for missing values and it silently changed underneath
        # us (diagnosed 2026-08-28):
        #   pandas 2.x : NaN -> the LITERAL STRING 'nan'  (a real value: merges, groups, counts)
        #   pandas 3.0 : NaN -> stays NA                  (does not match, excluded from nunique)
        # This code was written for the 2.x behaviour -- forcing an all-NaN float column into a
        # usable string merge key. Under 3.0 the column stayed NA, `unique_host_count` went 1 -> 0,
        # and every host-scoped groupby produced zero groups, so the analyzer emitted a
        # schema-valid metrics.json full of zeros while exiting 0. Same traces, same code, silent.
        #
        # The 2.x behaviour was ALSO wrong, just invisibly: it reported `unique_host_count=1` on a
        # 16-node/64-rank job because it was counting the string 'nan' as one host. So NA must be
        # preserved as NA -- a missing hash is missing, and must not masquerade as a value.
        # `astype("string")` has stable NA semantics across pandas 1.x-3.x and merges correctly
        # (NA keys simply do not match, which is the intended outcome for a left join).
        #
        # NOTE: this fixes the DTYPE HANDLING only. It does not explain why host_hash is NA on some
        # corpora in the first place -- that is a separate, still-open defect that this coercion was
        # masking. `_assert_metadata_resolved()` in analyzer.py now fails loudly on the outcome.
        if "file_hash" in records.columns:
            records["file_hash"] = records["file_hash"].astype("string")
        if "host_hash" in records.columns:
            records["host_hash"] = records["host_hash"].astype("string")
        # Both sides of the merge must share the key dtype, or the join silently matches nothing.
        file_hashes = file_hashes.copy()
        host_hashes = host_hashes.copy()
        file_hashes.index = file_hashes.index.astype("string")
        host_hashes.index = host_hashes.index.astype("string")
        records = records.merge(
            file_hashes.rename(columns={"name": COL_FILE_NAME}),
            how="left",
            left_on="file_hash",
            right_index=True,
        )
        records = records.merge(
            host_hashes.rename(columns={"name": COL_HOST_NAME}),
            how="left",
            left_on="host_hash",
            right_index=True,
        )
        return records

    @staticmethod
    def _rename_columns(traces: dd.DataFrame) -> dd.DataFrame:
        return traces.rename(columns=TRACE_COL_MAPPING)

    @staticmethod
    def _sanitize_size(df: pd.DataFrame):
        df["size"] = df["size"].replace(0, pd.NA)
        return df

    @staticmethod
    def _set_epochs(df: pd.DataFrame, epoch_boundaries: pd.DataFrame):
        df["epoch"] = pd.NA

        # Iterate over each epoch boundary to find matching events
        for _, epoch_boundary in epoch_boundaries.iterrows():
            pid = epoch_boundary["pid"]
            start = epoch_boundary["time_start"]
            end = epoch_boundary["time_end"]

            # Find rows in the partition that match the pid and fall within the time interval
            mask = (df["pid"] == pid) & (df["time_start"] >= start) & (df["time_start"] < end)

            # Assign the epoch number to the matching rows
            df.loc[mask, "epoch"] = epoch_boundary["epoch"]

        return df

    @staticmethod
    def _set_proc_names(df: pd.DataFrame):
        host_component = df[COL_HOST_NAME] if COL_HOST_NAME in df.columns else pd.Series(pd.NA, index=df.index)
        if "host_hash" in df.columns:
            host_component = host_component.fillna(df["host_hash"])
        df[COL_PROC_NAME] = (
            "app#"
            + host_component.fillna("unknown").astype(str)
            + "#"
            + df["pid"].astype(str)
            + "#"
            + df["tid"].astype(str)
        )
        return df

    @staticmethod
    def _set_steps(df: pd.DataFrame, step_time_ranges: pd.DataFrame):
        mapped_traces = df.copy()

        for pid in df["pid"].unique():
            pid_trace_cond = mapped_traces["pid"] == pid
            pid_traces = mapped_traces[pid_trace_cond]
            pid_step_ranges = step_time_ranges[step_time_ranges["pid"] == pid]

            # Sort step ranges by start timestamp
            pid_step_ranges_sorted = pid_step_ranges.sort_values("ts")

            # Create bins and labels
            bins = pid_step_ranges_sorted["ts"].tolist()
            if len(bins) > 0:
                bins.append(pid_step_ranges_sorted["te"].max())
            # print(pid, bins)
            steps = pid_step_ranges_sorted["step"].tolist()

            # Use np.digitize to find bin indices
            bin_indices = np.digitize(pid_traces["ts"], bins=bins) - 1

            # Map indices to steps, leaving as None for out-of-range timestamps
            mapped_traces.loc[pid_trace_cond, "step"] = [
                steps[idx] if 0 <= idx < len(steps) else pd.NA for idx in bin_indices
            ]

        return mapped_traces

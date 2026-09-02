import dataclasses as dc
import logging
import socket
from hydra.core.config_store import ConfigStore
from hydra.conf import HelpConf, JobConf
from omegaconf import MISSING
from typing import Any, Dict, List, Optional

from .constants import COL_TIME_RANGE
from .types import ViewMetricBoundaries
from .utils.env_utils import get_bool_env_var, get_int_env_var


CHECKPOINT_VIEWS = get_bool_env_var("DFANALYZER_CHECKPOINT_VIEWS", False)
CLUSTER_RESTART_TIMEOUT_SECONDS = get_int_env_var("DFANALYZER_CLUSTER_RESTART_TIMEOUT_SECONDS", 120)
DEFAULT_HLM_FIELDS = ["cat", "io_cat", "acc_pat", "func_name", "host_hash"]
DERIVED_POSIX_METRICS = {
    'data': 'io_cat == 1 or io_cat == 2',
    'read': 'io_cat == 1',
    'write': 'io_cat == 2',
    'metadata': 'io_cat == 3',
    'close': 'io_cat == 3 and func_name.str.contains("close") and ~func_name.str.contains("dir")',
    'open': 'io_cat == 3 and func_name.str.contains("open") and ~func_name.str.contains("dir")',
    'seek': 'io_cat == 3 and func_name.str.contains("seek")',
    'stat': 'io_cat == 3 and func_name.str.contains("stat")',
    'other': 'io_cat == 6',
    'sync': 'io_cat == 7',
}
DERIVED_POSIX_SIZE_METRICS = ('data', 'read', 'write')
HASH_CHECKPOINT_NAMES = get_bool_env_var("DFANALYZER_HASH_CHECKPOINT_NAMES", False)


@dc.dataclass
class AnalyzerPresetConfig:
    additional_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(default_factory=dict)
    async_layers: Optional[List[str]] = dc.field(default_factory=list)
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(default_factory=dict)
    hlm_fields: List[str] = dc.field(default_factory=lambda: list(DEFAULT_HLM_FIELDS))
    layer_defs: Dict[str, Optional[str]] = MISSING
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(default_factory=dict)
    logical_views: Optional[Dict[str, Dict[str, Optional[str]]]] = dc.field(default_factory=dict)
    name: str = MISSING
    # Path fragment -> category suffix, e.g. {"/p/lustre1/.../train": "_reader"}. This is what
    # SPLITS POSIX operations by purpose, and layers like `reader_posix` are defined as
    # `cat.str.contains("posix|stdio") & cat.str.contains("_reader")`. A preset with no purpose
    # layers wants {} here; see the DL presets below, where this is MANDATORY.
    file_purposes: Dict[str, str] = dc.field(default_factory=dict)
    size_derived_metrics: Optional[Dict[str, List[str]]] = dc.field(default_factory=dict)
    size_layers: Optional[List[str]] = dc.field(default_factory=list)
    time_boundary_layer: str = MISSING
    unscored_metrics: Optional[List[str]] = dc.field(default_factory=list)


@dc.dataclass
class AnalyzerPresetConfigPOSIX(AnalyzerPresetConfig):
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(
        default_factory=lambda: {
            'posix': DERIVED_POSIX_METRICS,
        }
    )
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: {
            'posix': 'cat.str.contains("posix|stdio")',
        }
    )
    logical_views: Optional[Dict[str, Dict[str, Optional[str]]]] = dc.field(
        default_factory=lambda: {
            'file_name': {
                'file_dir': None,
                'file_pattern': None,
            },
            'proc_name': {
                'host_name': 'proc_name.str.split("#").str[1]',
                'proc_id': 'proc_name.str.split("#").str[2]',
                'thread_id': 'proc_name.str.split("#").str[3]',
            },
        }
    )
    name: str = "posix"
    size_derived_metrics: Optional[Dict[str, List[str]]] = dc.field(
        default_factory=lambda: {
            'posix': list(DERIVED_POSIX_SIZE_METRICS),
        }
    )
    size_layers: Optional[List[str]] = dc.field(default_factory=lambda: ['posix'])
    time_boundary_layer: str = "posix"


@dc.dataclass
class AnalyzerPresetConfigDLIO(AnalyzerPresetConfig):
    # MANDATORY for every deep-learning preset, and inherited by DLIOAILogging, Stormer,
    # StormerXLayer and Halo3d. There is deliberately NO default.
    #
    # It used to be a hardcoded {"/data": "_reader", "/checkpoint": "_checkpoint"} inside
    # _fix_file_posix_category, which meant the entire reader_posix layer depended on the dataset
    # happening to be stored under a directory called "data". A corpus at
    # /l/ssd/izzet/foundation_data produced ZERO reader_posix rows -- "_data" is not "/data" -- so
    # every reader_posix rule was structurally unable to fire, silently, no matter how many events
    # reached the window. The era5 dataset matched only because "/datasets" contains "/data", which
    # is luck rather than design.
    #
    # A default here cannot be right: only the caller knows where its data lives. Requiring the
    # value converts a silent wrong answer into a startup error naming the field.
    file_purposes: Dict[str, str] = MISSING
    async_layers: Optional[List[str]] = dc.field(
        default_factory=lambda: [
            'data_loader',
            'data_loader_fork',
            'reader',
            'reader_posix',
            'posix',
            # 'reader_posix_lustre',
            # 'reader_posix_ssd',
            # 'checkpoint_posix_lustre',
            # 'checkpoint_posix_ssd',
        ]
    )
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(
        default_factory=lambda: {
            'app': {},
            'training': {},
            'epoch': {},
            'window_wall': {},
            'compute': {},
            'fetch_data': {},
            'checkpoint': {},
            'data_loader': {
                'init': 'func_name.str.contains("init")',
                'item': 'func_name.str.contains("item")',
            },
            'data_loader_fork': {},
            'reader': {
                'close': 'func_name.str.contains(".close")',
                'open': 'func_name.str.contains(".open")',  # e.g. NPZReader.open
                'preprocess': 'func_name.str.contains(".preprocess")',
                'sample': 'func_name.str.contains(".get_sample")',
            },
            'posix': DERIVED_POSIX_METRICS,
            'reader_posix': DERIVED_POSIX_METRICS,
            # 'reader_posix_lustre': DERIVED_POSIX_METRICS,
            # 'reader_posix_ssd': DERIVED_POSIX_METRICS,
            'checkpoint_posix': DERIVED_POSIX_METRICS,
            # 'checkpoint_posix_lustre': DERIVED_POSIX_METRICS,
            # 'checkpoint_posix_ssd': DERIVED_POSIX_METRICS,
            'other_posix': DERIVED_POSIX_METRICS,
            # 'other_posix_lustre': DERIVED_POSIX_METRICS,
            # 'other_posix_ssd': DERIVED_POSIX_METRICS,
        }
    )
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: {
            'app': 'func_name.isin(["DLIOBenchmark.initialize", "DLIOBenchmark.run"])',
            'training': 'func_name == "DLIOBenchmark.run"',
            'epoch': 'func_name == "DLIOBenchmark._train"',
            'compute': 'cat == "ai_framework"',
            'fetch_data': 'func_name.isin(["<module>.iter", "fetch-data.iter", "loop.iter"])',
            'checkpoint': 'cat == "checkpoint"',
            'data_loader': 'cat == "data_loader" & ~func_name.isin(["loop.iter", "loop.yield"])',
            'data_loader_fork': 'cat == "posix" & func_name == "fork"',
            'reader': 'cat == "reader"',
            'posix': 'cat.str.contains("posix|stdio")',
            'reader_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader")',
            'checkpoint_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint")',
        }
    )
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(
        default_factory=lambda: {
            'app': None,
            'training': 'app',
            'epoch': 'training',
            'compute': 'epoch',
            'fetch_data': 'epoch',
            'checkpoint': 'epoch',
            'data_loader': 'fetch_data',
            'data_loader_fork': 'fetch_data',
            'reader': 'data_loader',
            'posix': None,
            'reader_posix': 'reader',
            'checkpoint_posix': 'checkpoint',
        }
    )
    logical_views: Optional[Dict[str, Dict[str, Optional[str]]]] = dc.field(
        default_factory=lambda: {
            'file_name': {
                'file_dir': None,
                'file_pattern': None,
            },
            'proc_name': {
                'host_name': 'proc_name.str.split("#").str[1]',
                'proc_id': 'proc_name.str.split("#").str[2]',
                'thread_id': 'proc_name.str.split("#").str[3]',
            },
        }
    )
    name: str = "dlio-prev"
    size_derived_metrics: Optional[Dict[str, List[str]]] = dc.field(
        default_factory=lambda: {
            'posix': list(DERIVED_POSIX_SIZE_METRICS),
            'reader_posix': list(DERIVED_POSIX_SIZE_METRICS),
            'checkpoint_posix': list(DERIVED_POSIX_SIZE_METRICS),
        }
    )
    size_layers: Optional[List[str]] = dc.field(
        default_factory=lambda: [
            'posix',
            'reader_posix',
            'checkpoint_posix',
        ]
    )
    time_boundary_layer: str = "app"


@dc.dataclass
class AnalyzerPresetConfigDLIOAILogging(AnalyzerPresetConfigDLIO):
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(
        default_factory=lambda: {
            'app': {},
            'training': {},
            'epoch': {},
            'compute': {},
            'fetch_iter': {},
            'fetch_block': {},
            'data_loader': {
                'init': 'func_name.str.contains("init")',
                'item': 'func_name.str.contains("item")',
            },
            'data_loader_fork': {},
            'reader': {
                'close': 'func_name.str.contains(".close")',
                'open': 'func_name.str.contains(".open")',
                'preprocess': 'func_name.str.contains(".preprocess")',
                'sample': 'func_name.str.contains(".get_sample")',
            },
            'posix': DERIVED_POSIX_METRICS,
            'reader_posix': DERIVED_POSIX_METRICS,
            'checkpoint': {},
            'checkpoint_posix': DERIVED_POSIX_METRICS,
        }
    )
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: {
            'app': 'cat == "ai_root" & func_name == "ai_root"',
            'training': 'cat == "pipeline" & func_name == "train"',
            'epoch': 'cat == "pipeline" & func_name.str.startswith("epoch")',
            # One duration-bearing window.stop event per analysis window.  This
            # is the measured wall objective.  It must not be reconstructed as
            # the sum of domain layers because that makes the residual zero by
            # definition and hides missing attribution.
            'window_wall': 'cat == "window" & func_name == "window.stop"',
            'fetch_iter': 'cat == "dataloader" & func_name == "fetch.iter"',
            'fetch_block': 'cat == "dataloader" & func_name == "fetch.block"',
            'compute': 'cat == "compute" & func_name == "compute"',   # PARENT only:
            # `cat == "compute"` sums EVERY level of the hierarchy, so the moment a child
            # region (compute.forward/backward) is added it silently double counts. Selecting
            # the parent cannot double count at any depth. Requires the app to emit a region
            # NAMED `compute` -- see liveflow_shm/ANNOTATION-SPEC.md sec 4: parents land
            # BEFORE this narrowing, or compute_time_proc_max drops to 0 in silence.
            'data_loader': 'cat.isin(["data", "data_loader", "dataloader"])',
            'data_loader_fork': 'cat == "posix" & func_name == "fork"',
            'reader': 'cat == "reader" or func_name == "preprocess"',
            'posix': 'cat.str.contains("posix|stdio")',
            'reader_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader")',
            'checkpoint': 'cat == "checkpoint"',
            'checkpoint_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint")',
        }
    )
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(
        default_factory=lambda: {
            'app': None,
            'training': 'app',
            'epoch': 'training',
            'window_wall': None,
            'fetch_iter': 'epoch',
            'fetch_block': 'epoch',
            'compute': 'fetch_block',
            'data_loader': 'fetch_iter',
            'data_loader_fork': 'fetch_iter',
            'reader': 'data_loader',
            'posix': None,
            'reader_posix': 'reader',
            'checkpoint': 'epoch',
            'checkpoint_posix': 'checkpoint',
        }
    )
    name: str = "dlio"


@dc.dataclass
class AnalyzerPresetConfigStormer(AnalyzerPresetConfigDLIOAILogging):
    """Preset for StorMer (weather transformer) workload.

    Inherits the standard dft_ai layer definitions from the DLIO AI-logging
    preset.  Key differences from DLIO:
    - No per-step pipeline events — compute/fetch hang directly off epoch.
    - fetch_data includes fetch.block (the real I/O wait with prefetch workers)
      in addition to fetch.iter, so the analyzer correctly measures I/O time
      even when DataLoader prefetch hides it from the main process.
    """
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: {
            'app': 'cat == "ai_root" & func_name == "ai_root"',
            'training': 'cat == "pipeline" & func_name == "train"',
            'epoch': 'cat == "pipeline" & func_name.str.startswith("epoch")',
            'window_wall': 'cat == "window" & func_name == "window.stop"',
            'fetch_iter': 'cat == "dataloader" & func_name == "fetch.iter"',
            'fetch_block': 'cat == "dataloader" & func_name == "fetch.block"',
            'compute': 'cat == "compute" & func_name == "compute"',
            'compute_forward': 'cat == "compute" & func_name == "forward"',
            'compute_backward': 'cat == "compute" & func_name == "backward"',
            'comm': 'cat == "comm"',
            'comm_all_reduce': 'cat == "comm" & func_name == "all_reduce"',
            'data_loader': 'cat == "data" & func_name.isin(["data.init", "item"])',
            'data_loader_fork': 'cat == "posix" & func_name == "fork"',
            'reader': 'cat == "data" & func_name.isin(["preprocess"])',
            'posix': 'cat.str.contains("posix|stdio")',
            'reader_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader")',
        }
    )
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(
        default_factory=lambda: {
            'app': None,
            'training': 'app',
            'epoch': 'training',
            'window_wall': None,
            'fetch_iter': 'epoch',
            'fetch_block': 'epoch',
            'compute': 'fetch_block',
            'compute_forward': 'compute',
            'compute_backward': 'compute',
            'comm': 'compute_backward',
            'comm_all_reduce': 'comm',
            'data_loader': 'fetch_iter',
            'data_loader_fork': 'fetch_iter',
            'reader': 'data_loader',
            'posix': None,
            'reader_posix': 'reader',
        }
    )
    name: str = "stormer"


@dc.dataclass
class AnalyzerPresetConfigStormerXLayer(AnalyzerPresetConfigStormer):
    """StorMer with the DECODE domain exposed (LIVEFLOW-STATUS.md sec 36/37).

    Identical to the stormer preset EXCEPT it adds one layer:
        reader_decode : the `dft_ai.data.decode` region emitted by the two-stage reader
                        (STORMER_TWOSTAGE=1), i.e. the zlib inflate of the per-variable chunks.
    Everything else is inherited, so every existing stormer metric keeps its meaning and no
    derived_metrics entry can reference an undefined layer (the KeyError ->
    analysis_skipped_missing_column trap that silently skipped every window before).

    WHY NOT DERIVE STORAGE FROM POSIX. The stormer preset's `reader_posix` layer requires POSIX
    syscall interception, and that does NOT reliably reach forked DataLoader workers: measured,
    a workers=1 run yields 130 reader_posix-scoped findings while workers=8 runs yield ZERO
    (the sec 16.5 fork gap). So the three-way split here is built from APP-LEVEL regions we emit
    ourselves, which do work in workers:
        decode  = reader_decode_time_proc_max          (explicit region)
        input   = fetch_iter_time_proc_max - decode    (the remainder: queueing + raw chunk I/O)
        comm    = comm_time_proc_max
    reader_posix is left defined and still used opportunistically when it IS populated.

    Selected with DFANALYZER_PRESET=stormer_xlayer. Running the SAME trace under `stormer` gives
    the 2-domain view and under `stormer_xlayer` the 3-domain view -- a free attribution ablation.
    """
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: dict(
            AnalyzerPresetConfigStormer().layer_defs,
            **{'reader_decode': 'cat == "data" & func_name == "decode"'}
        )
    )
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(
        default_factory=lambda: dict(
            AnalyzerPresetConfigStormer().layer_deps,
            # decode runs in the forked workers and overlaps the main-process fetch wait through
            # the prefetch queue -> child of fetch_iter AND async (see async_layers below).
            **{'reader_decode': 'fetch_iter'}
        )
    )
    async_layers: List[str] = dc.field(
        default_factory=lambda: list(AnalyzerPresetConfigStormer().async_layers) + ['reader_decode']
    )
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(
        default_factory=lambda: dict(
            AnalyzerPresetConfigStormer().derived_metrics or {}, **{'reader_decode': {}}
        )
    )
    name: str = "stormer_xlayer"


@dc.dataclass
class AnalyzerPresetConfigHalo3d(AnalyzerPresetConfigStormer):
    """Preset for halo3d (LiveFlow three-way storage<->decode<->comm).

    The cross-layer JOIN splits the input pole into two reader-domain layers (sibling to how DLIO
    scopes reader_posix):
    - 'reader_posix'  : the POSIX read latency on cube_*.zz (syscall layer) -> STORAGE domain,
    - 'reader_decode' : the data.decode region (zlib inflate; emitted from the MAIN thread because the
                        MOFKA writer is unsafe for daemon-thread emission) -> DECODE domain,
    while 'comm' (halo) and 'compute' (conv+backward) are the main-thread app regions. The analyzer
    emits reader_posix_time_proc_max / reader_decode_time_proc_max / comm_time_proc_max /
    compute_time_proc_max / fetch_iter_time_proc_max per window; clbo_planner maps read_threads->storage
    (reader_posix), decode_threads->decode (reader_decode), halo_width->comm.
    NOTE: async_layers / derived_metrics / size_derived_metrics are OVERRIDDEN so they reference ONLY
    the layers this preset defines -- the inherited Stormer/DLIO values name data_loader/reader/... which
    do not exist here and cause a hard KeyError (analysis_skipped_missing_column) in set_cross_layer_metrics.
    (POSIX cat matched case-insensitively: some dftracer builds emit "POSIX", others "posix".)"""
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: {
            'app': 'cat == "ai_root" & func_name == "ai_root"',
            'training': 'cat == "pipeline" & func_name == "train"',
            'fetch_iter': 'cat == "dataloader" & func_name == "fetch.iter"',  # was bare `cat == "dataloader"`, which would also swallow fetch.block                 # dataloader.fetch (app-level fetch)
            'comm': 'cat == "comm"',                             # halo exchange (COMM domain)
            'compute': 'cat == "compute" & func_name == "compute"',  # PARENT only; halo3d emits a region named `compute`                       # conv + backward (COMPUTE domain)
            'posix': 'cat.str.contains("posix|stdio", case=False)',
            'reader_posix': 'cat.str.contains("posix|stdio", case=False) & func_name == "read"',  # STORAGE
            'reader_decode': 'cat == "data"',                    # data.decode (DECODE domain)
        }
    )
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(
        default_factory=lambda: {
            'app': None,
            'training': 'app',
            'fetch_iter': 'training',
            'comm': 'training',
            'compute': 'training',
            'posix': None,
            'reader_posix': 'fetch_iter',    # reader-domain, overlaps fetch via the prefetch queue
            'reader_decode': 'fetch_iter',
        }
    )
    # Async (overlapping) reader-domain layers -- MUST only name layers defined above.
    async_layers: List[str] = dc.field(
        default_factory=lambda: ['reader_posix', 'reader_decode', 'posix'])
    # No extra derived/size metrics needed (the fact rule uses the per-layer *_time_proc_max directly);
    # override to only-defined layers so nothing references data_loader/reader/checkpoint.
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(
        default_factory=lambda: {k: {} for k in
                                 ['app', 'training', 'fetch_iter', 'comm', 'compute',
                                  'posix', 'reader_posix', 'reader_decode']})
    size_derived_metrics: Optional[Dict[str, List[str]]] = dc.field(default_factory=lambda: {})
    name: str = "halo3d"


@dc.dataclass
class AnalyzerConfig:
    checkpoint: Optional[bool] = False
    checkpoint_dir: Optional[str] = "${hydra:run.dir}/checkpoints"
    profile_distribution: Optional[str] = "uniform"
    profile_time_granularity: Optional[float] = 5
    preset: Optional[AnalyzerPresetConfig] = MISSING
    quantile_stats: Optional[bool] = False
    time_approximate: Optional[bool] = True
    time_granularity: Optional[float] = MISSING
    time_resolution: Optional[float] = MISSING
    time_sliced: Optional[bool] = False


@dc.dataclass
class DarshanAnalyzerConfig(AnalyzerConfig):
    _target_: str = "dftracer.analyzer.darshan.DarshanAnalyzer"
    time_granularity: Optional[float] = 1
    time_resolution: Optional[float] = 1e3


@dc.dataclass
class DFTracerAnalyzerConfig(AnalyzerConfig):
    _target_: str = "dftracer.analyzer.dftracer.DFTracerAnalyzer"
    assign_epochs: Optional[bool] = False
    time_granularity: Optional[float] = 1
    time_resolution: Optional[float] = 1e6


@dc.dataclass
class RecorderAnalyzerConfig(AnalyzerConfig):
    _target_: str = "dftracer.analyzer.recorder.RecorderAnalyzer"
    time_granularity: Optional[float] = 1
    time_resolution: Optional[float] = 1e7


@dc.dataclass
class ClusterConfig:
    local_directory: Optional[str] = "/tmp/${hydra:job.name}-${oc.env:USER}/${oc.select:hydra.job.id,0}"


@dc.dataclass
class ExternalClusterConfig(ClusterConfig):
    _target_: str = "dftracer.analyzer.cluster.ExternalCluster"
    restart_on_connect: Optional[bool] = False
    scheduler_address: Optional[str] = MISSING


@dc.dataclass
class JobQueueClusterSchedulerConfig:
    dashboard_address: Optional[str] = None
    host: Optional[str] = dc.field(default_factory=socket.gethostname)


@dc.dataclass
class JobQueueClusterConfig(ClusterConfig):
    cores: int = 16  # ncores
    death_timeout: Optional[int] = 60
    job_directives_skip: Optional[List[str]] = dc.field(default_factory=list)
    job_extra_directives: Optional[List[str]] = dc.field(default_factory=list)
    log_directory: Optional[str] = ""
    memory: Optional[str] = None
    processes: Optional[int] = 1  # nnodes
    scheduler_options: Optional[JobQueueClusterSchedulerConfig] = dc.field(
        default_factory=JobQueueClusterSchedulerConfig
    )


@dc.dataclass
class LocalClusterConfig(ClusterConfig):
    _target_: str = "dask.distributed.LocalCluster"
    host: Optional[str] = None
    memory_limit: Optional[int] = None
    n_workers: Optional[int] = None
    # MUST be settable (LIVEFLOW-STATUS.md sec 54). Without this field the value cannot be
    # overridden, and dask then derives it as
    #     threads_per_worker = ceil(CPU_COUNT / n_workers)
    # so the streaming analyzer's `n_workers=1` yields a NINETY-SIX thread worker on a 96-core
    # node -- one per node, fired at every control-window boundary, on the same cores as the
    # training ranks and their ncclProxyProgress threads. That burst starves NCCL's CPU-side proxy,
    # the GPU stream never drains, and the job dies in hipDeviceSynchronize. n_workers alone bounds
    # workers, NOT threads, which is why "--workers 1" looked safe and was not.
    threads_per_worker: Optional[int] = None
    processes: Optional[bool] = True
    silence_logs: Optional[int] = logging.CRITICAL


@dc.dataclass
class LSFClusterConfig(JobQueueClusterConfig):
    _target_: str = "dask_jobqueue.LSFCluster"
    use_stdin: Optional[bool] = True


@dc.dataclass
class PBSClusterConfig(JobQueueClusterConfig):
    _target_: str = "dask_jobqueue.PBSCluster"


@dc.dataclass
class SLURMClusterConfig(JobQueueClusterConfig):
    _target_: str = "dask_jobqueue.SLURMCluster"


@dc.dataclass
class InputConfig:
    pass


@dc.dataclass
class FileInputConfig(InputConfig):
    _target_: str = "dftracer.analyzer.input.FileInput"
    path: str = MISSING


@dc.dataclass
class ZMQInputConfig(InputConfig):
    _target_: str = "dftracer.analyzer.input.ZMQInput"
    address: str = MISSING


@dc.dataclass
class MofkaInputConfig(InputConfig):
    _target_: str = "dftracer.analyzer.input.MofkaInput"
    group_file: str = MISSING
    topic_name: str = MISSING
    control_topic_name: Optional[str] = None


@dc.dataclass
class OutputConfig:
    pass


@dc.dataclass
class FactsConfig:
    enabled: bool = False
    rule_file: str = ""
    emit_flat_views: bool = True
    emit_analysis_facts: bool = False
    strict_time_semantics: bool = True
    allow_mixed_time_aggregates: bool = False


@dc.dataclass
class FileOutputConfig(OutputConfig):
    compact: Optional[bool] = False
    name: Optional[str] = ""
    root_only: Optional[bool] = True
    view_names: Optional[List[str]] = dc.field(default_factory=list)


@dc.dataclass
class ConsoleOutputConfig(OutputConfig):
    _target_: str = "dftracer.analyzer.output.ConsoleOutput"
    show_debug: Optional[bool] = False
    show_header: Optional[bool] = True


@dc.dataclass
class JSONOutputConfig(OutputConfig):
    _target_: str = "dftracer.analyzer.output.JSONOutput"
    file_path: Optional[str] = ""


@dc.dataclass
class CSVOutputConfig(OutputConfig):
    _target_: str = "dftracer.analyzer.output.CSVOutput"


@dc.dataclass
class SQLiteOutputConfig(OutputConfig):
    _target_: str = "dftracer.analyzer.output.SQLiteOutput"
    run_db_path: Optional[str] = ""


@dc.dataclass
class ZMQOutputConfig(OutputConfig):
    _target_: str = "dftracer.analyzer.output.ZMQOutput"
    address: str = MISSING


@dc.dataclass
class MofkaOutputConfig(OutputConfig):
    _target_: str = "dftracer.analyzer.output.MofkaOutput"
    group_file: str = MISSING
    topic_name: str = MISSING
    source_node: str = ""


@dc.dataclass
class CustomJobConfig(JobConf):
    name: str = "dftracer.analyzer"


@dc.dataclass
class CustomHelpConfig(HelpConf):
    app_name: str = "DFAnalyzer"
    header: str = "${hydra:help.app_name}: Data Flow Analyzer"
    footer: str = dc.field(
        default_factory=lambda: """
Powered by Hydra (https://hydra.cc)

Use --hydra-help to view Hydra specific help
    """.strip()
    )
    template: str = dc.field(
        default_factory=lambda: """
${hydra:help.header}

== Configuration groups ==

Compose your configuration from those groups (group=option)

$APP_CONFIG_GROUPS
== Config ==

Override anything in the config (foo.bar=value)

$CONFIG
${hydra:help.footer}
    """.strip()
    )


@dc.dataclass
class Config:
    defaults: List[Any] = dc.field(
        default_factory=lambda: [
            {"analyzer": "dftracer"},
            {"analyzer/preset": "posix"},
            {"hydra/job": "custom"},
            {"cluster": "local"},
            {"input": "file"},
            {"output": "console"},
            "_self_",
            {"override hydra/help": "custom"},
        ]
    )
    analyzer: AnalyzerConfig = MISSING
    cluster: ClusterConfig = MISSING
    debug: Optional[bool] = False
    exclude_characteristics: Optional[List[str]] = dc.field(default_factory=list)
    facts: FactsConfig = dc.field(default_factory=FactsConfig)
    input: InputConfig = MISSING
    logical_view_types: Optional[bool] = False
    metric_boundaries: Optional[ViewMetricBoundaries] = dc.field(default_factory=dict)
    output: OutputConfig = MISSING
    verbose: Optional[bool] = False
    view_types: Optional[List[str]] = dc.field(default_factory=lambda: [COL_TIME_RANGE])


def init_hydra_config_store() -> ConfigStore:
    cs = ConfigStore.instance()
    cs.store(group="hydra/help", name="custom", node=dc.asdict(CustomHelpConfig()))
    cs.store(group="hydra/job", name="custom", node=CustomJobConfig)
    cs.store(name="config", node=Config)
    cs.store(group="analyzer", name="darshan", node=DarshanAnalyzerConfig)
    cs.store(group="analyzer", name="dftracer", node=DFTracerAnalyzerConfig)
    cs.store(group="analyzer", name="recorder", node=RecorderAnalyzerConfig)
    cs.store(group="analyzer/preset", name="posix", node=AnalyzerPresetConfigPOSIX)
    cs.store(group="analyzer/preset", name="dlio-prev", node=AnalyzerPresetConfigDLIO)
    cs.store(group="analyzer/preset", name="dlio", node=AnalyzerPresetConfigDLIOAILogging)
    cs.store(group="analyzer/preset", name="stormer", node=AnalyzerPresetConfigStormer)
    cs.store(group="analyzer/preset", name="halo3d", node=AnalyzerPresetConfigHalo3d)
    cs.store(group="analyzer/preset", name="stormer_xlayer", node=AnalyzerPresetConfigStormerXLayer)
    cs.store(group="cluster", name="external", node=ExternalClusterConfig)
    cs.store(group="cluster", name="local", node=LocalClusterConfig)
    cs.store(group="cluster", name="lsf", node=LSFClusterConfig)
    cs.store(group="cluster", name="pbs", node=PBSClusterConfig)
    cs.store(group="cluster", name="slurm", node=SLURMClusterConfig)
    cs.store(group="input", name="file", node=FileInputConfig)
    cs.store(group="input", name="zmq", node=ZMQInputConfig)
    cs.store(group="input", name="mofka", node=MofkaInputConfig)
    cs.store(group="output", name="console", node=ConsoleOutputConfig)
    cs.store(group="output", name="json", node=JSONOutputConfig)
    cs.store(group="output", name="csv", node=CSVOutputConfig)
    cs.store(group="output", name="sqlite", node=SQLiteOutputConfig)
    cs.store(group="output", name="zmq", node=ZMQOutputConfig)
    cs.store(group="output", name="mofka", node=MofkaOutputConfig)
    return cs

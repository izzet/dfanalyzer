import hydra
import signal
from dask_jobqueue import LSFCluster, PBSCluster, SLURMCluster
from dataclasses import asdict, dataclass, field
from distributed import LocalCluster, wait
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING
from typing import Any, List, Optional, Union

from .config import (
    ClusterConfig,
    CustomHelpConfig,
    CustomJobConfig,
    InProcessClusterConfig,
    LocalClusterConfig,
    LSFClusterConfig,
    PBSClusterConfig,
    SLURMClusterConfig,
)


@dataclass
class ExternalCluster:
    restart_on_connect: Optional[bool]
    scheduler_address: str
    local_directory: Optional[str]


@dataclass
class InProcessCluster:
    """Marker for "no cluster at all" -- see `InProcessClusterConfig`.

    Carries no scheduler, so `dfanalyzer-cluster` cannot serve it; it exists
    only so `cluster=none` composes like any other cluster choice.
    """

    local_directory: Optional[str] = None

    def close(self) -> None:
        """Nothing was started, so nothing needs stopping."""


# Keyword arguments `Client.submit` accepts to place and schedule a task. They
# describe *where* work runs, which is meaningless once it runs here, so
# NullClient drops them rather than passing them to the function.
_SCHEDULING_KWARGS = frozenset(
    {
        "actor",
        "actors",
        "allow_other_workers",
        "fifo_timeout",
        "key",
        "priority",
        "pure",
        "resources",
        "retries",
        "workers",
    }
)


class NullClient:
    """A `Client`-shaped object that runs work here and now.

    The analyzer talks to a Dask client in a handful of places -- submitting
    scans, gathering partials, scheduling checkpoint writes. Rather than teach
    each of those to ask "is there a cluster?", this satisfies the same calls
    synchronously, so one code path serves both modes.

    `submit` returns the function's result rather than a future. That is what
    lets the shim disappear: Dask resolves futures to values before calling a
    function, so a caller that submits a "future" here and passes it to another
    `submit` sees exactly what it would have seen from a real client. `gather`
    is then the identity.

    `is_distributed` is the one thing callers may legitimately branch on --
    reading a trace, for instance, has to scan locally instead of fanning the
    work out to workers that do not exist.
    """

    is_distributed = False

    def submit(self, fn, *args, **kwargs):
        for key in _SCHEDULING_KWARGS:
            kwargs.pop(key, None)
        return fn(*args, **kwargs)

    def gather(self, futures, **kwargs):
        """Identity: `submit` already returned values, not futures."""
        if isinstance(futures, (list, tuple)):
            return list(futures)
        return futures

    def compute(self, collections, sync=True, **kwargs):
        """Compute now, whatever `sync` says -- there is nowhere to defer to."""
        import dask

        return dask.compute(collections)[0]

    def cancel(self, *args, **kwargs) -> None:
        """Work is finished by the time anything could cancel it."""

    def nthreads(self, *args, **kwargs) -> dict:
        """No workers, so no per-worker thread counts."""
        return {}

    def run(self, fn, *args, **kwargs) -> dict:
        """Run a worker-setup function against this process instead."""
        return {"in-process": fn(*args, **kwargs)}

    def restart(self, *args, **kwargs) -> None:
        """No workers to restart."""

    def close(self, *args, **kwargs) -> None:
        """Nothing to disconnect from."""

    def shutdown(self, *args, **kwargs) -> None:
        """Nothing to shut down."""


def wait_for(client, items) -> None:
    """Block until `items` are finished, where finishing means anything.

    This asks the *client*, not the frames. `distributed.wait` needs a
    scheduler tracking futures, and whether one exists is a property of how
    execution was set up -- not of whether a given collection is a Dask
    collection. The two come apart exactly here: under `cluster=none` the
    frames are still genuine Dask DataFrames, so anything that dispatched on
    the frame would conclude "Dask, therefore wait" and raise `No clients
    found`. There is nothing to wait for either way, because `persist` and
    `compute` have already run to completion inline.
    """
    if getattr(client, "is_distributed", True):
        wait(items)


@dataclass
class Config:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"hydra/job": "custom"},
            {"cluster": "local"},
            "_self_",
            {"override hydra/help": "custom"},
        ]
    )
    cluster: ClusterConfig = MISSING
    debug: Optional[bool] = False
    verbose: Optional[bool] = False


cs = ConfigStore.instance()
cs.store(group="hydra/help", name="custom", node=asdict(CustomHelpConfig()))
cs.store(group="hydra/job", name="custom", node=CustomJobConfig)
cs.store(name="config", node=Config)
cs.store(group="cluster", name="local", node=LocalClusterConfig)
cs.store(group="cluster", name="none", node=InProcessClusterConfig)
cs.store(group="cluster", name="lsf", node=LSFClusterConfig)
cs.store(group="cluster", name="pbs", node=PBSClusterConfig)
cs.store(group="cluster", name="slurm", node=SLURMClusterConfig)


ClusterType = Union[ExternalCluster, InProcessCluster, LocalCluster, LSFCluster, PBSCluster, SLURMCluster]


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    cluster: ClusterType = instantiate(cfg.cluster)
    print(cluster.scheduler.address, flush=True)
    try:
        signal.pause()
    except KeyboardInterrupt:
        print("Shutting down the Dask cluster...")
    finally:
        cluster.close()
        print("Dask cluster is shut down.")


if __name__ == "__main__":
    main()

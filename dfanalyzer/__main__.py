import hydra
from distributed import Client
from hydra.utils import instantiate
from omegaconf import OmegaConf

from . import AnalyzerType, ClusterType, OutputType
from .config import CLUSTER_RESTART_TIMEOUT_SECONDS, Config, init_hydra_config_store
from .cluster import ExternalCluster


init_hydra_config_store()


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    cluster: ClusterType = instantiate(cfg.cluster)
    if isinstance(cluster, ExternalCluster):
        client = Client(cluster.scheduler_address)
        if cluster.restart_on_connect:
            client.restart(timeout=CLUSTER_RESTART_TIMEOUT_SECONDS)
    else:
        client = Client(cluster)
    analyzer: AnalyzerType = instantiate(
        cfg.analyzer,
        debug=cfg.debug,
        verbose=cfg.verbose,
    )
    result = analyzer.analyze_trace(
        exclude_characteristics=cfg.exclude_characteristics,
        logical_view_types=cfg.logical_view_types,
        metric_boundaries=OmegaConf.to_object(cfg.metric_boundaries),
        percentile=cfg.percentile,
        threshold=cfg.threshold,
        trace_path=cfg.trace_path,
        unoverlapped_posix_only=cfg.unoverlapped_posix_only,
        view_types=cfg.view_types,
    )
    output: OutputType = instantiate(cfg.output)
    output.handle_result(result=result)
    client.close()
    if not isinstance(cluster, ExternalCluster):
        cluster.close()  # type: ignore


if __name__ == "__main__":
    main()

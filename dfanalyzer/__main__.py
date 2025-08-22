import hydra
import structlog
from distributed import Client
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from . import AnalyzerType, ClusterType, OutputType
from .config import CLUSTER_RESTART_TIMEOUT_SECONDS, Config, init_hydra_config_store
from .cluster import ExternalCluster
from .utils.log_utils import configure_logging, console_block, log_block
from .utils.warning_utils import filter_warnings

filter_warnings()
init_hydra_config_store()

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    # Configure structlog + stdlib logging
    hydra_config = HydraConfig.get()
    log_file = f"{hydra_config.runtime.output_dir}/{hydra_config.job.name}.log"
    # print(OmegaConf.to_yaml(hydra_config))
    configure_logging(log_file=log_file, json_logs=False, level="info")
    log = structlog.get_logger()
    log.info("Starting dfanalyzer")

    with console_block("Cluster setup"):
        cluster: ClusterType = instantiate(cfg.cluster)
        if isinstance(cluster, ExternalCluster):
            client = Client(cluster.scheduler_address)
            if cluster.restart_on_connect:
                client.restart(timeout=CLUSTER_RESTART_TIMEOUT_SECONDS)
        else:
            client = Client(cluster)

    with log_block("Configuring logging on all Dask workers"):
        client.run(configure_logging, log_file=log_file, json_logs=False, level="info")

    analyzer: AnalyzerType = instantiate(
        cfg.analyzer,
        debug=cfg.debug,
        verbose=cfg.verbose,
    )
    result = analyzer.analyze_trace(
        exclude_characteristics=cfg.exclude_characteristics,
        logical_view_types=cfg.logical_view_types,
        metric_boundaries=OmegaConf.to_object(cfg.metric_boundaries),
        trace_path=cfg.trace_path,
        unoverlapped_posix_only=cfg.unoverlapped_posix_only,
        view_types=cfg.view_types,
    )

    with console_block("Output"):
        output: OutputType = instantiate(cfg.output)
        output.handle_result(result=result)

    with console_block("Cluster teardown"):
        client.close()
        if not isinstance(cluster, ExternalCluster):
            cluster.close()  # type: ignore


if __name__ == "__main__":
    main()

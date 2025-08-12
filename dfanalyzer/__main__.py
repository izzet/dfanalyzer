import hydra
import json
import signal
from distributed import Client
from hydra.utils import instantiate
from omegaconf import OmegaConf

from . import AnalyzerType, ClusterType, OutputType
from .config import Config, FileInputConfig, ZMQInput, init_hydra_config_store
from .cluster import ExternalCluster
from .types import Rule


init_hydra_config_store()


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    cluster: ClusterType = instantiate(cfg.cluster)

    if isinstance(cluster, ExternalCluster):
        client = Client(cluster.scheduler_address)
        if cluster.restart_on_connect:
            client.restart()
    else:
        client = Client(cluster)

    analyzer: AnalyzerType = instantiate(
        cfg.analyzer,
        debug=cfg.debug,
        verbose=cfg.verbose,
    )
    input = instantiate(cfg.input)
    output: OutputType = instantiate(cfg.output)
    if isinstance(input, FileInputConfig):
        result = analyzer.analyze_file(
            exclude_characteristics=cfg.exclude_characteristics,
            logical_view_types=cfg.logical_view_types,
            metric_boundaries=OmegaConf.to_object(cfg.metric_boundaries),
            percentile=cfg.percentile,
            threshold=cfg.threshold,
            path=cfg.input.path,
            unoverlapped_posix_only=cfg.unoverlapped_posix_only,
            view_types=cfg.view_types,
        )
        output.handle_result(result=result)
    elif isinstance(input, ZMQInput):
        print(f"Starting stream analysis from: {input.address}")
        analysis_stream = analyzer.analyze_zmq(
            address=input.address,
            exclude_characteristics=cfg.exclude_characteristics,
            logical_view_types=cfg.logical_view_types,
            metric_boundaries=OmegaConf.to_object(cfg.metric_boundaries),
            percentile=cfg.percentile,
            threshold=cfg.threshold,
            unoverlapped_posix_only=cfg.unoverlapped_posix_only,
            view_types=cfg.view_types,
        )
        analysis_stream = analysis_stream.map(lambda result: result.flat_views[('epoch',)].to_json(orient='index'))
        analysis_stream.sink(print)
        analysis_stream.to_zmq(output.address)
        analysis_stream.visualize('analysis')
        analysis_stream.start()
        print("Streaming analysis started. Press Ctrl+C to exit.")
        try:
            signal.pause()
        except KeyboardInterrupt:
            print("\nShutting down streaming analysis...")
    else:
        raise ValueError(f"Unsupported input configuration type: {type(cfg.input)}")

    print("Closing Dask client and cluster...")
    client.close()
    cluster.close()  # type: ignore
    print("Shutdown complete.")


if __name__ == "__main__":
    main()

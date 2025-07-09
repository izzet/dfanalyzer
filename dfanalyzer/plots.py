import dask.dataframe as dd
import dataclasses
import hydra
import math
import random
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category20 as colors
from bokeh.plotting import figure, show
from distributed import Client
from enum import Enum
from hydra.utils import instantiate
from omegaconf import MISSING
from typing import List

from .__main__ import AnalyzerType, ClusterType
from .cluster import ExternalCluster
from .config import Config, init_hydra_config_store


def plot_bottleneck_summary(
    flat_bottlenecks: dd.DataFrame,
    groupby: tuple,
    view_depth=1,
    xaxis_major_label_orientation=math.pi / 2,
    y_range_start=0,
):
    bot_cols = get_bottleneck_columns(flat_bottlenecks)
    df = flat_bottlenecks.query(f"view_depth == {view_depth}")
    group_df = df.groupby(list(groupby))[bot_cols].sum().compute()
    group_df = group_df.reset_index()
    group_df['group_id'] = list(map(str, zip(*(group_df[col] for col in groupby))))
    x_range = list(group_df['group_id'])
    p = figure(x_range=x_range, tools="hover", tooltips="$name: @$name")
    p.vbar_stack(
        bot_cols,
        x='group_id',
        width=0.8,
        fill_color=colors[len(bot_cols)],
        line_color='black',
        source=ColumnDataSource(group_df),
        legend_label=bot_cols,
    )
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = xaxis_major_label_orientation
    p.xgrid.grid_line_color = None
    p.y_range.start = y_range_start
    return p


def get_bottleneck_columns(bottlenecks: dd.DataFrame) -> List[str]:
    return [
        col for col in bottlenecks.columns if 'b_' in col and not col[-1].isnumeric()
    ]


class PlotType(Enum):
    LAYER_BOTTLENECKS = 0


@dataclasses.dataclass
class PlotConfig(Config):
    plot: PlotType = PlotType.LAYER_BOTTLENECKS


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
    bottlenecks = analyzer.read_bottlenecks()


if __name__ == "__main__":
    cs = init_hydra_config_store()
    cs.store(name="config", node=PlotConfig)
    main()

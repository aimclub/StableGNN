from stable_gnn.visualization.config.parameters.defaults import Defaults
from stable_gnn.visualization.config.types import TEdgeList


def calc_arrow_head_width(edge_line_width: list, show_arrow: bool, edge_list: TEdgeList):
    return [Defaults.arrow_multiplier * w for w in edge_line_width] if show_arrow else [0] * len(edge_list)

from dataclasses import dataclass

from stable_gnn.visualization.config.parameters.defaults import Defaults
from stable_gnn.visualization.config.types import TEdgeList


@dataclass
class SizeConstructorContract:
    vertex_num: int
    edge_list: TEdgeList | None = None
    vertex_size: float | list = Defaults.vertex_size
    vertex_line_width: float | list = Defaults.vertex_line_width
    edge_line_width: float | list = Defaults.edge_line_width
    font_size: float = Defaults.font_size

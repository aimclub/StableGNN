from dataclasses import dataclass

from stable_gnn.visualization.config.parameters.defaults import Defaults


@dataclass
class SizeConstructorContract:
    vertex_num: int
    edges_list: list[tuple]
    vertex_size: float | list = Defaults.vertex_size
    vertex_line_width: float | list = Defaults.vertex_line_width
    edge_line_width: float | list = Defaults.edge_line_width
    font_size: float = Defaults.font_size

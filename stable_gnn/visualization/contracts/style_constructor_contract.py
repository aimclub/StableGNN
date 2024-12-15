from dataclasses import dataclass

from stable_gnn.visualization.config.parameters.defaults import Defaults


@dataclass
class GraphStyleConstructorContract:
    vertex_num: int
    edges_num: int
    vertex_color: str | list = Defaults.vertex_color
    edge_color: str | list = Defaults.edge_color
    edge_fill_color: str | list = Defaults.edge_fill_color

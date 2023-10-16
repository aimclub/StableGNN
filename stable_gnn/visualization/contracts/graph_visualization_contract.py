from dataclasses import dataclass

from stable_gnn.graph import Graph
from stable_gnn.visualization.config.parameters.defaults import Defaults


@dataclass
class GraphVisualizationContract:
    graph: Graph
    edge_style: str = Defaults.edge_style
    edge_color: str | list = Defaults.edge_color
    vertex_label: list | None = None
    vertex_size: float | list = Defaults.vertex_size
    vertex_color: str | list = Defaults.vertex_color
    vertex_line_width: float | list = Defaults.vertex_line_width
    font_size: float = Defaults.font_size
    font_family: str = Defaults.font_family
    push_vertex_strength: float = Defaults.push_vertex_strength
    push_edge_strength: float = Defaults.push_edge_strength
    pull_edge_strength: float = Defaults.pull_edge_strength
    pull_center_strength: float = Defaults.pull_center_strength

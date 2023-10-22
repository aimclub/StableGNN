from dataclasses import dataclass

from stable_gnn.visualization.config.types import TGraphEdgeList


@dataclass
class GraphContract:
    vertex_num: int
    edge_num: int
    edge_list: TGraphEdgeList | None = None
    edge_weights: float | list[float] | None = None

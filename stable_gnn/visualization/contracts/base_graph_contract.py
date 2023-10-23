from dataclasses import dataclass

from stable_gnn.visualization.config.types import TGraphEdgeList


@dataclass
class BaseGraphContract:
    vertex_num: int
    edge_list: TGraphEdgeList | None = None


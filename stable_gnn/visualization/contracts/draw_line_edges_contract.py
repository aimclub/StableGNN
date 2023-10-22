import numpy as np
from dataclasses import dataclass

from stable_gnn.visualization.contracts.size_constructor_contract import TEdgeList


@dataclass
class DrawLineEdgesContract:
    vertex_coordinates: np.array
    vertex_size: list
    edge_list: TEdgeList
    show_arrow: bool
    edge_color: list
    edge_line_width: list

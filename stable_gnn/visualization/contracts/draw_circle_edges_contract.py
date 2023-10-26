from dataclasses import dataclass

from stable_gnn.visualization.config.types import TVectorCoordinates


@dataclass
class DrawEdgesContract:
    vertex_coordinates: TVectorCoordinates
    vertex_size: list
    edge_list: list[tuple] | list[list[int]]
    edge_color: list
    edge_fill_color: list
    edge_line_width: list

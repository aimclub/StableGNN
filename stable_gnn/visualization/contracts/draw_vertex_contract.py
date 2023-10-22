from dataclasses import dataclass

from stable_gnn.visualization.config.types import TVectorCoordinates


@dataclass
class DrawVertexContract:
    vertex_coordinates: TVectorCoordinates
    vertex_label: list[str] | None
    font_size: int
    font_family: str
    vertex_size: list
    vertex_color: list
    vertex_line_width: list

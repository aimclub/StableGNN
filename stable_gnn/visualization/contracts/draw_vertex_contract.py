import matplotlib
from dataclasses import dataclass


@dataclass
class DrawVertexContract:
    axes: matplotlib.axes.Axes
    vertex_coordinates: list[tuple[float, float]]
    vertex_label: list[str] | None
    font_size: int
    font_family: str
    vertex_size: list
    vertex_color: list
    vertex_line_width: list

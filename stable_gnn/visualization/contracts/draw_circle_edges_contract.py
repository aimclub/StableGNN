import matplotlib
from dataclasses import dataclass


@dataclass
class DrawEdgesContract:
    axes: matplotlib.axes.Axes
    vertex_coordinates: list[tuple[float, float]]
    vertex_size: list
    edge_list: list[tuple] | list[list[int]]
    edge_color: list
    edge_fill_color: list
    edge_line_width: list

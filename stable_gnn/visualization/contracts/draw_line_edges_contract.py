import matplotlib
import numpy as np
from dataclasses import dataclass


@dataclass
class DrawLineEdgesContract:
    axes: matplotlib.axes.Axes
    vertex_coordinates: np.array
    vertex_size: list
    edge_list: list[tuple] | list[list[int]]
    show_arrow: bool
    edge_color: list
    edge_line_width: list

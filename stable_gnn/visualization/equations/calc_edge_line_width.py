import numpy as np

from stable_gnn.visualization.config.parameters.defaults import Defaults
from stable_gnn.visualization.utils.cached import cached


@cached()
def calculate_edge_line_width(edge_list_length: int):
    return Defaults.edge_line_width_multiplier * np.exp(-edge_list_length / Defaults.edge_line_width_divider)

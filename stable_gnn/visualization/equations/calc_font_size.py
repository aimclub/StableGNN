import numpy as np

from stable_gnn.visualization.config.parameters.defaults import Defaults
from stable_gnn.visualization.utils.cached import cached


@cached()
def calculate_font_size(vertex_num):
    return Defaults.font_size_multiplier * np.exp(-vertex_num / Defaults.font_size_divider)

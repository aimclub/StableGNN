import numpy as np


def init_position(vertex_num: int, center: tuple[float, float] = (0, 0), scale: float = 1.0):
    return (np.random.rand(vertex_num, 2) * 2 - 1) * scale + center

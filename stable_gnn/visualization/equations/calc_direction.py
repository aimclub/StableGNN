import numpy as np


def calc_direction(direction):
    return direction / np.linalg.norm(direction)

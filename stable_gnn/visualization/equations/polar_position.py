import math
import numpy as np


def polar_position(r, theta, start_point):
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return np.array([x, y]) + start_point

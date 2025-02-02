import math

import numpy as np


def polar_position(r, theta, start_point):
    return np.array([r * math.cos(theta), r * math.sin(theta)]) + start_point

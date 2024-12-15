import numpy as np


def calc_edge_center(H, position):
    return np.matmul(H.T, position) / H.sum(axis=0).reshape(-1, 1)

from itertools import chain

import numpy as np


def edge_list_to_incidence_matrix(vertex_num: int, edge_list: list[tuple]) -> np.ndarray:
    vertex_indexes = list(chain(*edge_list))
    edge_indexes = [[idx] * len(e) for idx, e in enumerate(edge_list)]
    edge_indexes = list(chain(*edge_indexes))
    H = np.zeros((vertex_num, len(edge_list)))
    H[vertex_indexes, edge_indexes] = 1
    return H

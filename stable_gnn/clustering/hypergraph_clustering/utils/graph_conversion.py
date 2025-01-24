import networkx as nx
import numpy as np

def hypergraph_to_incidence_matrix(hyperedges):
    """
    Преобразует гиперграф в матрицу инцидентности.
    
    :param hyperedges: Список гиперребер (каждое гиперребро — список вершин).
    :return: np.ndarray, матрица инцидентности.
    """
    nodes = sorted(set(node for edge in hyperedges for node in edge))
    node_index = {node: i for i, node in enumerate(nodes)}
    incidence_matrix = np.zeros((len(nodes), len(hyperedges)))
    
    for j, edge in enumerate(hyperedges):
        for node in edge:
            incidence_matrix[node_index[node], j] = 1
            
    return incidence_matrix

def incidence_to_adjacency(incidence_matrix):
    """
    Преобразует матрицу инцидентности в матрицу смежности графа.
    
    :param incidence_matrix: np.ndarray, матрица инцидентности.
    :return: np.ndarray, матрица смежности.
    """
    adjacency_matrix = np.dot(incidence_matrix, incidence_matrix.T)
    np.fill_diagonal(adjacency_matrix, 0)  # Убираем петли
    return adjacency_matrix

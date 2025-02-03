import json
import os

import pytest
from hypergraph_clustering.utils.graph_conversion import hypergraph_to_incidence_matrix, incidence_to_adjacency

# Путь к JSON-файлам
DATA_DIR = "data"


@pytest.mark.parametrize("filename", ["biological_network.json", "electric_network.json"])
def test_hypergraph_to_incidence_matrix(filename):
    # Загружаем гиперграф из JSON
    with open(os.path.join(DATA_DIR, filename), "r") as f:
        hypergraph = json.load(f)

    # Преобразуем гиперграф в матрицу инцидентности
    incidence_matrix = hypergraph_to_incidence_matrix(hypergraph["hyperedges"])

    # Проверки
    assert incidence_matrix.shape[0] == hypergraph["num_nodes"]
    assert incidence_matrix.shape[1] == len(hypergraph["hyperedges"])
    assert incidence_matrix.sum() == sum(len(edge) for edge in hypergraph["hyperedges"])


@pytest.mark.parametrize("filename", ["biological_network.json", "electric_network.json"])
def test_incidence_to_adjacency(filename):
    # Загружаем гиперграф из JSON
    with open(os.path.join(DATA_DIR, filename), "r") as f:
        hypergraph = json.load(f)

    # Преобразуем гиперграф в матрицу инцидентности и затем в матрицу смежности
    incidence_matrix = hypergraph_to_incidence_matrix(hypergraph["hyperedges"])
    adjacency_matrix = incidence_to_adjacency(incidence_matrix)

    # Проверки
    assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
    assert adjacency_matrix.sum() > 0
    assert (adjacency_matrix.diagonal() == 0).all()

import pytest
import torch
from torch_geometric.utils import add_remaining_self_loops
import numpy as np
import networkx as nx
from unittest.mock import MagicMock
from stable_gnn.generation.graph_generator.core.graph_builder import GraphBuilder
from stable_gnn.embedding.models.model_factory import ModelFactory
from stable_gnn.analytics.gh_graph_metrics import GHGraphMetrics
from stable_gnn.clustering.hypergraph_clustering.clustering.agglomerative import AgglomerativeHypergraphClustering
from stable_gnn.generation.graph_generator.core.llm_client import LLMClient

# Мокируем LLMClient
@pytest.fixture
def mock_llm_client():
    """Fixture для создания мокированного экземпляра LLMClient."""
    mock = MagicMock(spec=LLMClient)
    mock.generate_graph_description = MagicMock(return_value={"A": ["B", "C"], "B": ["A"], "C": ["A"]})
    return mock

@pytest.fixture
def adjacency_matrix():
    """Fixture для создания тестовой матрицы смежности."""
    matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return matrix

@pytest.fixture
def edge_index(adjacency_matrix):
    """Fixture для создания edge_index из матрицы смежности."""
    return torch.tensor(np.transpose(np.nonzero(adjacency_matrix)), dtype=torch.long)

def test_graph_generation(mock_llm_client):
    """Тестирование генерации графа."""
    builder = GraphBuilder(mock_llm_client)
    text = "Тестовое описание графа"
    graph = builder.build_graph(text)
    assert len(graph.nodes) > 0, "Граф должен содержать хотя бы один узел."
    assert len(graph.edges) > 0, "Граф должен содержать хотя бы одно ребро."

def test_graph_metrics(edge_index, adjacency_matrix):
    """Тестирование вычисления метрик графа."""
    metrics = GHGraphMetrics()
    num_nodes = adjacency_matrix.shape[0]  # Определяем количество узлов
    if edge_index.size(0) == 0:
        pytest.skip("Граф не содержит ребер, пропускаем тест.")
    try:
        centrality = metrics.calculate_centrality(edge_index, num_nodes)
        sparsity_loss = metrics.calculate_sparsity_loss(edge_index, num_nodes)  # Передаем edge_index вместо adjacency_matrix
    except Exception as e:
        pytest.fail(f"Ошибка при вычислении метрик графа: {e}")
    assert centrality is not None, "Центральность не должна быть None."
    assert sparsity_loss is not None, "Потеря разреженности не должна быть None."


def test_clustering(adjacency_matrix):
    """Тестирование кластеризации."""
    assert adjacency_matrix.size > 1, "Матрица смежности должна содержать больше одного элемента."
    clustering = AgglomerativeHypergraphClustering(n_clusters=2, linkage="ward")
    try:
        clusters = clustering.fit(adjacency_matrix)
    except Exception as e:
        pytest.fail(f"Ошибка при кластеризации: {e}")
    assert len(np.unique(clusters)) == 2, "Кластеризация должна создать два кластера."

def test_pipeline_integration(mock_llm_client):
    """Интеграционный тест для всего пайплайна."""
    builder = GraphBuilder(mock_llm_client)
    text = "Тестовое описание графа"
    graph = builder.build_graph(text)
    adjacency_matrix = nx.to_numpy_array(graph)
    edge_index = torch.tensor(np.transpose(np.nonzero(adjacency_matrix)), dtype=torch.long).t()
    num_nodes = adjacency_matrix.shape[0]

    # Отладочный вывод перед добавлением петель
    print(f"edge_index before adding self-loops: {edge_index}")
    print(f"num_nodes: {num_nodes}")

    # Добавляем оставшиеся петли для узлов
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)

    # Отладочный вывод после добавления петель
    print(f"edge_index after adding self-loops: {edge_index}")

    # Проверяем кластеризацию
    clustering = AgglomerativeHypergraphClustering(n_clusters=2, linkage="ward")
    try:
        clusters = clustering.fit(adjacency_matrix)
    except Exception as e:
        pytest.fail(f"Ошибка в кластеризации на этапе интеграции: {e}")

    # Проверяем обучение модели
    model = ModelFactory().build_model(
        conv="GCN",
        loss_function={"loss var": "Random Walks"},
        device="cpu",
        num_features=adjacency_matrix.shape[0],
        hidden_layer=64,
        out_layer=32,
        num_layers=2,
        dropout=0.1,
        heads=1,
    )

    # Подготовка данных для forward
    x = torch.eye(num_nodes, dtype=torch.float32)  # Фичи узлов как единичная матрица
    adjs = [(edge_index, None, (num_nodes, num_nodes))]

    try:
        # Подаем `x` без выделения x_target
        for i, (edge_index, _, size) in enumerate(adjs):
            x = model.convs[i](x, edge_index)  # Прямая передача в GCNConv
            if i != model.num_layers - 1:
                x = torch.relu(x)
        assert x is not None, "Модель должна возвращать результат."
    except Exception as e:
        pytest.fail(f"Ошибка в обучении модели на этапе интеграции: {e}")
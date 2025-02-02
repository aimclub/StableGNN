import networkx as nx
import torch

from stable_gnn.analytics.gh_graph_metrics import GHGraphMetrics
from stable_gnn.embedding.models.model_factory import ModelFactory
from stable_gnn.generation.graph_generator.core.graph_builder import GraphBuilder
from stable_gnn.generation.graph_generator.core.llm_client import LLMClient


def run_gln_pipeline(graph_text: str, num_clusters: int = 2):
    """
    Запуск пайплайна GLN: генерация графа, обучение модели, расчет метрик и кластеризация.

    :param graph_text: str: Текстовое описание графа.
    :param num_clusters: int: Количество кластеров для агломеративной кластеризации.
    """
    print("Step 1: Generating GH-graph...")
    llm_client = LLMClient()  # Инстанс LLMClient
    builder = GraphBuilder(llm_client)

    # Генерация графа
    graph = builder.build_graph(graph_text)
    adjacency_matrix = nx.to_numpy_array(graph)

    print(f"Generated graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

    # Визуализация графа
    builder.visualize_graph(graph)

    print("Step 2: Training model on generated graph...")
    model = ModelFactory().build_model(
        conv="GCN",
        loss_function={"loss var": "Random Walks"},
        device="cuda",
        num_features=adjacency_matrix.shape[0],
        hidden_layer=64,
        out_layer=32,
        num_layers=2,
        dropout=0.1,
        heads=1,
    )
    model.train(adjacency_matrix)

    print("Step 3: Calculating graph metrics...")
    metrics = GHGraphMetrics()

    # Преобразуем граф в edge_index и передаем количество узлов
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes)}
    edge_index = (
        torch.tensor([(node_mapping[u], node_mapping[v]) for u, v in graph.edges], dtype=torch.long).t().contiguous()
    )

    centrality = metrics.calculate_centrality(edge_index, num_nodes=len(graph.nodes))
    print(f"Calculated centrality: {centrality}")

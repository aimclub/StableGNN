import numpy as np
import networkx as nx
from stable_gnn.generation.graph_generator.core.graph_builder import GraphBuilder
from stable_gnn.embedding.models.model_factory import ModelFactory
from stable_gnn.analytics.gh_graph_metrics import GHGraphMetrics
from stable_gnn.clustering.hypergraph_clustering.clustering.agglomerative import AgglomerativeHypergraphClustering


def run_gln_pipeline(graph_text: str, num_clusters: int = 2):
    """
    Запуск пайплайна GLN: генерация графа, обучение модели, расчет метрик и кластеризация.

    :param graph_text: str: Текстовое описание графа.
    :param num_clusters: int: Количество кластеров для агломеративной кластеризации.
    """
    print("Step 1: Generating GH-graph...")
    llm_client = None  # Инстанс LLMClient
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
    centrality = metrics.calculate_centrality(adjacency_matrix)
    sparsity_loss = metrics.calculate_sparsity_loss(adjacency_matrix)

    print(f"Centrality: {centrality}")
    print(f"Sparsity Loss: {sparsity_loss}")

    print("Step 4: Running clustering...")
    clustering = AgglomerativeHypergraphClustering(n_clusters=num_clusters, linkage="ward")
    clusters = clustering.fit(adjacency_matrix)
    clustering.plot_clusters(adjacency_matrix)

    print(f"Clusters: {clusters}")

    print("Pipeline execution completed successfully.")

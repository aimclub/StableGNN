import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_hypergraph(hyperedges, node_colors=None, title="Hypergraph Visualization"):
    """
    Визуализирует гиперграф.

    :param hyperedges: list of lists, гиперребра гиперграфа.
    :param node_colors: dict, цвета для узлов (по умолчанию одинаковые для всех).
    :param title: str, заголовок графика.
    """
    G = nx.Graph()

    # Добавляем узлы и гиперребра
    for idx, edge in enumerate(hyperedges):
        for node in edge:
            G.add_node(node)
        G.add_edges_from([(edge[i], edge[j]) for i in range(len(edge)) for j in range(i + 1, len(edge))])

    # Определяем цвета узлов
    if node_colors is None:
        node_colors = {node: "blue" for node in G.nodes}
    colors = [node_colors.get(node, "blue") for node in G.nodes]

    # Рисуем график
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color="gray", node_size=500, font_size=10)
    plt.title(title)
    plt.show()

def plot_clusters(adjacency_matrix, labels, title="Cluster Visualization"):
    """
    Визуализирует кластеры, полученные из кластеризации.

    :param adjacency_matrix: np.ndarray, матрица смежности графа.
    :param labels: list, метки кластеров.
    :param title: str, заголовок графика.
    """
    G = nx.Graph()
    nodes = range(adjacency_matrix.shape[0])

    # Добавляем узлы и ребра
    for i in nodes:
        for j in nodes:
            if adjacency_matrix[i, j] > 0:
                G.add_edge(i, j, weight=adjacency_matrix[i, j])

    # Определяем цвета кластеров
    cluster_colors = plt.cm.rainbow(np.linspace(0, 1, len(set(labels))))
    node_colors = {node: cluster_colors[label] for node, label in enumerate(labels)}
    colors = [node_colors[node] for node in G.nodes]

    # Рисуем график
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color="gray", node_size=500, font_size=10)
    plt.title(title)
    plt.show()

import numpy as np
from hypergraph_clustering.clustering.agglomerative import AgglomerativeHypergraphClustering

# Пример графа
adjacency_matrix = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]])

# Кластеризация
clustering = AgglomerativeHypergraphClustering(n_clusters=2, linkage="average")
labels = clustering.fit(adjacency_matrix)

# Оценка
metrics = clustering.evaluate(adjacency_matrix)
print("Metrics:", metrics)

# Визуализация
clustering.plot_clusters(adjacency_matrix)

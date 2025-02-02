import numpy as np
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


class AgglomerativeHypergraphClustering:
    def __init__(self, n_clusters=2, linkage="ward", normalize=True):
        """
        Класс для агломеративной кластеризации гиперграфов.

        :param n_clusters: Количество кластеров.
        :param linkage: Метод связи ('ward', 'complete', 'average', 'single').
        :param normalize: Нормализовать ли матрицу смежности перед кластеризацией.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.normalize = normalize
        self.model = None

    def normalize_adjacency(self, adjacency_matrix):
        """
        Нормализует матрицу смежности с использованием нормализации строки.

        :param adjacency_matrix: np.ndarray, матрица смежности.
        :return: np.ndarray, нормализованная матрица.
        """
        row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
        normalized_matrix = adjacency_matrix / np.maximum(row_sums, 1e-10)
        return normalized_matrix

    def compute_distance_matrix(self, adjacency_matrix):
        """
        Преобразует матрицу смежности в матрицу расстояний.

        :param adjacency_matrix: np.ndarray, матрица смежности.
        :return: np.ndarray, матрица расстояний.
        """
        max_distance = np.max(adjacency_matrix) + 1
        distance_matrix = np.where(
            adjacency_matrix > 0,
            1 / adjacency_matrix,  # Инвертируем веса для положительных значений
            max_distance,  # Устанавливаем большое значение для недостижимых узлов
        )
        np.fill_diagonal(distance_matrix, 0)  # Убираем петли
        return distance_matrix

    def compute_laplacian(self, adjacency_matrix):
        """
        Вычисляет матрицу Лапласа графа.

        :param adjacency_matrix: np.ndarray, матрица смежности.
        :return: np.ndarray, матрица Лапласа.
        """
        degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
        laplacian_matrix = degree_matrix - adjacency_matrix
        return laplacian_matrix

    def fit(self, adjacency_matrix):
        """
        Выполняет кластеризацию на основе матрицы смежности.

        :param adjacency_matrix: np.ndarray, матрица смежности графа.
        :return: np.ndarray, метки кластеров.
        """
        max_distance = np.max(adjacency_matrix) + 1

        # Исправляем проблему деления на ноль
        distance_matrix = np.zeros_like(adjacency_matrix, dtype=float)
        non_zero_indices = adjacency_matrix > 0
        distance_matrix[non_zero_indices] = 1 / adjacency_matrix[non_zero_indices]
        distance_matrix[~non_zero_indices] = max_distance
        np.fill_diagonal(distance_matrix, 0)

        if self.linkage == "ward":
            # Для ward используем матрицу признаков (смежности)
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
            labels = self.model.fit_predict(adjacency_matrix)
        else:
            # Для остальных методов используем метрику "precomputed"
            condensed_distance_matrix = squareform(distance_matrix)
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters, metric="precomputed", linkage=self.linkage)
            labels = self.model.fit_predict(squareform(condensed_distance_matrix))

        self.labels_ = labels
        return labels

    def evaluate(self, adjacency_matrix):
        """
        Оценивает качество кластеризации с использованием различных метрик.

        :param adjacency_matrix: np.ndarray, матрица смежности графа.
        :return: dict, значения метрик.
        """
        if not hasattr(self, "labels_"):
            raise ValueError("Кластеризация не выполнена. Сначала вызовите fit().")

        metrics = {
            "silhouette_score": silhouette_score(adjacency_matrix, self.labels_, metric="precomputed"),
            "calinski_harabasz_score": calinski_harabasz_score(adjacency_matrix, self.labels_),
            "davies_bouldin_score": davies_bouldin_score(adjacency_matrix, self.labels_),
        }
        return metrics

    def spectral_embedding(self, adjacency_matrix, n_components=2):
        """
        Выполняет спектральное вложение на основе матрицы Лапласа.

        :param adjacency_matrix: np.ndarray, матрица смежности.
        :param n_components: int, количество компонент вложения.
        :return: np.ndarray, спектральные вложения.
        """
        laplacian = self.compute_laplacian(adjacency_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        return eigenvectors[:, :n_components]

    def plot_clusters(self, adjacency_matrix):
        """
        Визуализирует кластеры с использованием спектрального вложения.

        :param adjacency_matrix: np.ndarray, матрица смежности.
        """
        import matplotlib.pyplot as plt

        embeddings = self.spectral_embedding(adjacency_matrix)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=self.labels_, cmap="viridis", s=50)
        plt.title("Clusters Visualization")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(label="Cluster")
        plt.show()

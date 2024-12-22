import unittest
import numpy as np
import networkx as nx
from stable_gnn.pipelines.gln_pipeline import run_gln_pipeline
from stable_gnn.generation.graph_generator.core.graph_builder import GraphBuilder
from stable_gnn.analytics.gh_graph_metrics import GHGraphMetrics
from stable_gnn.clustering.hypergraph_clustering.clustering.agglomerative import AgglomerativeHypergraphClustering
from stable_gnn.embedding.models.model_factory import ModelFactory


class TestGLNPipeline(unittest.TestCase):
    def setUp(self):
        """
        Настройка перед тестами: создание фиктивного графа и матрицы смежности.
        """
        # Текст для фиктивного графа
        self.graph_text = """
        Иван, Аня и Петр гуляли в парке. Иван и Аня — друзья, а Петр познакомился с ними в кафе.
        """
        llm_client = None  # Убедитесь, что здесь используется реальный LLMClient
        self.builder = GraphBuilder(llm_client)

        # Генерация графа
        self.graph = self.builder.build_graph(self.graph_text)
        self.adjacency_matrix = nx.to_numpy_array(self.graph)

    def test_graph_generation(self):
        """
        Проверка генерации графа.
        """
        self.assertTrue(len(self.graph.nodes) > 0, "Граф не должен быть пустым.")
        self.assertTrue(len(self.graph.edges) > 0, "Граф должен содержать связи.")
        print("Graph generation test passed!")

    def test_model_training(self):
        """
        Проверка обучения модели на сгенерированном графе.
        """
        model = ModelFactory().build_model(
            conv="GCN",
            loss_function={"loss var": "Random Walks"},
            device="cpu",
            num_features=self.adjacency_matrix.shape[0],
            hidden_layer=64,
            out_layer=32,
            num_layers=2,
            dropout=0.1,
            heads=1,
        )
        try:
            model.train(self.adjacency_matrix)
        except Exception as e:
            self.fail(f"Model training failed with error: {e}")
        print("Model training test passed!")

    def test_graph_metrics(self):
        """
        Проверка расчета метрик графа.
        """
        metrics = GHGraphMetrics()
        centrality = metrics.calculate_centrality(self.adjacency_matrix)
        sparsity_loss = metrics.calculate_sparsity_loss(self.adjacency_matrix)

        self.assertTrue(isinstance(centrality, np.ndarray), "Центральность должна быть массивом.")
        self.assertTrue(sparsity_loss > 0, "Потери от разреженности должны быть больше нуля.")
        print("Graph metrics test passed!")

    def test_clustering(self):
        """
        Проверка выполнения агломеративной кластеризации.
        """
        clustering = AgglomerativeHypergraphClustering(n_clusters=2, linkage="ward")
        clusters = clustering.fit(self.adjacency_matrix)

        self.assertTrue(len(clusters) == self.adjacency_matrix.shape[0], "Кластеры должны соответствовать числу узлов.")
        print("Clustering test passed!")

    def test_pipeline_execution(self):
        """
        Проверка выполнения полного пайплайна GLN.
        """
        try:
            run_gln_pipeline(graph_text=self.graph_text, num_clusters=2)
        except Exception as e:
            self.fail(f"Pipeline execution failed with error: {e}")
        print("Pipeline execution test passed!")


if __name__ == "__main__":
    unittest.main()
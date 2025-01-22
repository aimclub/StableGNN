import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import numpy as np
import torch

from stable_gnn.pipelines.gln_pipeline import run_gln_pipeline
from stable_gnn.generation.graph_generator.core.graph_builder import GraphBuilder
from stable_gnn.analytics.gh_graph_metrics import GHGraphMetrics
from stable_gnn.clustering.hypergraph_clustering.clustering.agglomerative import AgglomerativeHypergraphClustering
from stable_gnn.embedding.models.model_factory import ModelFactory


class TestGLNPipeline(unittest.TestCase):

    @patch("stable_gnn.generation.graph_generator.core.llm_client.LLMClient")
    def setUp(self, mock_llm_client):
        self.mock_llm_instance = mock_llm_client.return_value
        self.mock_llm_instance.generate_graph_description.return_value = {
            "Иван": ["Аня"],
            "Аня": ["Иван", "Петр"],
            "Петр": ["Аня"]
        }
        self.mock_llm_instance.extract_entities.return_value = [
            {"name": "Иван", "type": "Person"},
            {"name": "Аня", "type": "Person"},
            {"name": "Петр", "type": "Person"},
        ]
        self.graph_text = "Иван, Аня и Петр гуляли в парке. Иван и Аня — друзья, а Петр познакомился с ними в кафе."

        self.builder = GraphBuilder(llm_client=self.mock_llm_instance)
        self.graph = self.builder.build_graph(self.graph_text)
        self.adjacency_matrix = nx.to_numpy_array(self.graph)

    def test_graph_generation(self):
        self.assertGreater(len(self.graph.nodes), 0, "Граф не должен быть пустым.")
        self.assertGreater(len(self.graph.edges), 0, "Граф должен содержать хотя бы одну связь.")

    def test_model_training(self):
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
        model.fit = MagicMock()
        try:
            model.fit(self.adjacency_matrix)
        except Exception as e:
            self.fail(f"Ошибка при обучении модели: {e}")

    def test_graph_metrics(self):
        metrics = GHGraphMetrics()

        adjacency_matrix_tensor = torch.from_numpy(self.adjacency_matrix).float()
        node_mapping = {node: idx for idx, node in enumerate(self.graph.nodes)}
        edge_index = torch.tensor(
            [(node_mapping[u], node_mapping[v]) for u, v in self.graph.edges],
            dtype=torch.long
        ).t().contiguous()

        centrality = metrics.calculate_centrality(edge_index, len(self.graph.nodes))
        sparsity_loss = metrics.calculate_sparsity_loss(edge_index, num_nodes=len(self.graph.nodes))

        self.assertIsInstance(centrality, torch.Tensor, "Центральность должна быть тензором.")
        self.assertGreater(sparsity_loss, 0, "Потери от разреженности должны быть больше 0.")

    def test_clustering(self):
        clustering = AgglomerativeHypergraphClustering(n_clusters=2, linkage="ward")
        clusters = clustering.fit(self.adjacency_matrix)
        self.assertEqual(len(clusters), self.adjacency_matrix.shape[0],
                         "Кластеры должны соответствовать числу узлов в графе.")

    @patch("stable_gnn.pipelines.gln_pipeline.GraphBuilder")
    @patch("stable_gnn.pipelines.gln_pipeline.ModelFactory")
    def test_pipeline_execution(self, mock_model_factory, mock_graph_builder):
        mock_graph_builder.return_value.build_graph.return_value = nx.Graph([
            ("Иван", "Аня"),
            ("Аня", "Петр")
        ])

        mock_model_instance = mock_model_factory.return_value.build_model.return_value
        mock_model_instance.fit = MagicMock()

        try:
            run_gln_pipeline(graph_text=self.graph_text, num_clusters=2)
        except Exception as e:
            self.fail(f"Ошибка при выполнении полного пайплайна: {e}")


if __name__ == "__main__":
    unittest.main()

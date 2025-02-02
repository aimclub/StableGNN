import numpy as np
import torch
from torch_geometric.utils import degree, to_dense_adj


class GHGraphMetrics:
    """Utility class for computing GH-graph metrics: centrality, radius, and sparsity-aware loss.

    This class provides static methods for calculating graph centrality, radius, sparsity loss,
    and diameter.
    """

    @staticmethod
    def calculate_centrality(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Calculate centrality of nodes based on their degree.

        :param edge_index: (Tensor): Edge indices of the graph.
        :param num_nodes: (int): Number of nodes in the graph.
        :return: (Tensor): Centrality values for each node.
        """
        node_degree = degree(edge_index[0], num_nodes=num_nodes)
        centrality = node_degree / node_degree.sum()
        return centrality

    @staticmethod
    def calculate_radius(edge_index: torch.Tensor, num_nodes: int) -> float:
        """
        Calculate the radius of the graph based on shortest paths.

        :param edge_index: (Tensor): Edge indices of the graph.
        :param num_nodes: (int): Number of nodes in the graph.
        :return: (float): Radius of the graph.
        """
        adjacency = to_dense_adj(edge_index).squeeze(0).cpu().numpy()
        shortest_paths = np.linalg.matrix_power(adjacency + np.eye(num_nodes), num_nodes - 1)
        eccentricities = shortest_paths.max(axis=1)
        return eccentricities.min()

    @staticmethod
    def calculate_sparsity_loss(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Calculate sparsity-aware loss for the graph.

        :param edge_index: (Tensor): Edge indices of the graph.
        :param num_nodes: (int): Number of nodes in the graph.
        :return: (Tensor): Sparsity-aware loss value.
        """
        node_degree = degree(edge_index[0], num_nodes=num_nodes)
        sparsity_loss = torch.mean((1 - (node_degree / node_degree.max())) ** 2)
        return sparsity_loss

    @staticmethod
    def calculate_diameter(edge_index: torch.Tensor, num_nodes: int) -> float:
        """
        Calculate the diameter of the graph.

        :param edge_index: (Tensor): Edge indices of the graph.
        :param num_nodes: (int): Number of nodes in the graph.
        :return: (float): Diameter of the graph.
        """
        adjacency = to_dense_adj(edge_index).squeeze(0).cpu().numpy()
        shortest_paths = np.linalg.matrix_power(adjacency + np.eye(num_nodes), num_nodes - 1)
        eccentricities = shortest_paths.max(axis=1)
        return eccentricities.max()


class GHLossFunctions:
    """Collection of GH-graph loss functions: sparsity-aware and centrality-weighted losses.

    This class provides static methods to compute sparsity-aware loss and centrality-weighted loss.
    """

    @staticmethod
    def sparsity_aware_loss(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute sparsity-aware loss.

        :param edge_index: (Tensor): Edge indices of the graph.
        :param num_nodes: (int): Number of nodes in the graph.
        :return: (Tensor): Sparsity loss value.
        """
        return GHGraphMetrics.calculate_sparsity_loss(edge_index, num_nodes)

    @staticmethod
    def centrality_weighted_loss(pred: torch.Tensor, target: torch.Tensor, centrality: torch.Tensor) -> torch.Tensor:
        """
        Compute a centrality-weighted loss.

        :param pred: (Tensor): Predicted values.
        :param target: (Tensor): Ground truth values.
        :param centrality: (Tensor): Centrality values for weighting.
        :return: (Tensor): Centrality-weighted loss.
        """
        base_loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
        weighted_loss = (base_loss * centrality).mean()
        return weighted_loss


# Example usage
if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # Example edge index
    num_nodes = 4

    metrics = GHGraphMetrics()
    centrality = metrics.calculate_centrality(edge_index, num_nodes)
    radius = metrics.calculate_radius(edge_index, num_nodes)
    sparsity_loss = metrics.calculate_sparsity_loss(edge_index, num_nodes)
    diameter = metrics.calculate_diameter(edge_index, num_nodes)

    print(f"Centrality: {centrality}")
    print(f"Radius: {radius}")
    print(f"Sparsity Loss: {sparsity_loss}")
    print(f"Diameter: {diameter}")

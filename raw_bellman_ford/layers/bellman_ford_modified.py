import torch
import torch.nn as nn

class BellmanFordLayerModified(nn.Module):
    def __init__(self, num_nodes, num_features):
        super(BellmanFordLayerModified, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features

        self.edge_weights = nn.Parameter(torch.rand(num_nodes, num_nodes))
        self.node_embedding = nn.Embedding(num_nodes, num_features)

    def forward(self, adj_matrix, source_node):
        distances = torch.full((self.num_nodes,), float('inf'))
        predecessors = torch.full((self.num_nodes,), -1)
        distances[source_node] = 0

        for _ in range(self.num_nodes - 1):
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d and adj_matrix[s][d] != float('inf'):
                        if distances[s] + adj_matrix[s][d] < distances[d]:
                            distances[d] = distances[s] + adj_matrix[s][d]
                            predecessors[d] = s

        graph_diameter = torch.max(distances).item()
        graph_eccentricity = torch.max(distances[source_node]).item()
        
        node_features = self.node_embedding(torch.arange(self.num_nodes))
        node_features = torch.cat([node_features, distances.unsqueeze(1)], dim=1)

        return node_features, graph_diameter, graph_eccentricity

if __name__ == "__main__":
    num_nodes_1 = 4
    adj_matrix_1 = torch.tensor([[0, 2, float('inf'), 1],
                            [float('inf'), 0, -1, float('inf')],
                            [float('inf'), float('inf'), 0, -2],
                            [float('inf'), float('inf'), float('inf'), 0]])
    source_node_1 = 0

    bellman_ford_layer_1 = BellmanFordLayerModified(num_nodes_1, num_features=5)
    node_features_1, diameter_1, eccentricity_1 = bellman_ford_layer_1(adj_matrix_1, source_node_1)

    print("Example 1:")
    print("Node Features:")
    print(node_features_1)
    print("Graph Diameter:", diameter_1)
    print("Graph Eccentricity:", eccentricity_1)

    num_nodes_2 = 4
    adj_matrix_2 = torch.tensor([[0, 2, 1, float('inf')],
                            [float('inf'), 0, -1, float('inf')],
                            [float('inf'), float('inf'), 0, -2],
                            [float('inf'), float('inf'), float('inf'), 0]])
    source_node_2 = 0

    bellman_ford_layer_2 = BellmanFordLayerModified(num_nodes_2, num_features=5)
    node_features_2, diameter_2, eccentricity_2 = bellman_ford_layer_2(adj_matrix_2, source_node_2)

    print("\nExample 2:")
    print("Node Features:")
    print(node_features_2)
    print("Graph Diameter:", diameter_2)
    print("Graph Eccentricity:", eccentricity_2)

    num_nodes_3 = 4
    adj_matrix_3 = torch.tensor([[0, 2, 1, 3],
                            [-1, 0, -1, 4],
                            [5, 2, 0, -2],
                            [2, 3, 1, 0]])
    source_node_3 = 0

    bellman_ford_layer_3 = BellmanFordLayerModified(num_nodes_3, num_features=5)
    node_features_3, diameter_3, eccentricity_3 = bellman_ford_layer_3(adj_matrix_3, source_node_3)

    print("\nExample 3:")
    print("Node Features:")
    print(node_features_3)
    print("Graph Diameter:", diameter_3)
    print("Graph Eccentricity:", eccentricity_3)

    num_nodes_4 = 4
    adj_matrix_4 = torch.tensor([[0, 2, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 2],
                            [0, 0, 0, 0]])
    source_node_4 = 0

    bellman_ford_layer_4 = BellmanFordLayerModified(num_nodes_4, num_features=5)
    node_features_4, diameter_4, eccentricity_4 = bellman_ford_layer_4(adj_matrix_4, source_node_4)

    print("\nExample 4:")
    print("Node Features:")
    print(node_features_4)
    print("Graph Diameter:", diameter_4)
    print("Graph Eccentricity:", eccentricity_4)

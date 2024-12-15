import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.bellman_ford_orig import BellmanFordLayer

class GNNWithBellmanFord(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes):
        super(GNNWithBellmanFord, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes

        self.bellman_ford_layer = BellmanFordLayer(num_nodes)
        self.node_embedding = nn.Embedding(num_nodes, num_features)
        self.fc = nn.Linear(num_features + num_nodes, num_classes)

    def forward(self, adj_matrix, source_node):
        distances, predecessors, has_negative_cycle = self.bellman_ford_layer(adj_matrix, source_node)

        node_features = self.node_embedding(torch.arange(self.num_nodes))
        node_features = torch.cat([node_features, distances], dim=1)

        output = self.fc(node_features)

        return output, has_negative_cycle 

if __name__ == "__main__":

    num_nodes = 4
    adj_matrix = torch.tensor([[0, 2, float('inf'), 1],
                                [float('inf'), 0, -1, float('inf')],
                                [float('inf'), float('inf'), 0, -2],
                                [float('inf'), float('inf'), float('inf'), 0]])
    source_node = 0

    gnn_model = GNNWithBellmanFord(num_nodes, num_features=5, num_classes=2)
    output, has_negative_cycle = gnn_model(adj_matrix, source_node)

    if has_negative_cycle:
        print("Example 1: The graph contains a negative weight cycle")
    else:
        print("Example 1: GNN output:", output)

    num_nodes = 5
    adj_matrix = torch.tensor([[0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 1, 0, 1],
                                [0, 0, 0, 1, 0]])
    source_node = 2

    gnn_model = GNNWithBellmanFord(num_nodes, num_features=4, num_classes=3)
    output, has_negative_cycle = gnn_model(adj_matrix, source_node)

    if has_negative_cycle:
        print("Example 2: The graph contains a negative weight cycle")
    else:
        print("Example 2: GNN output:", output)

    num_nodes = 3
    adj_matrix = torch.tensor([[0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]])
    source_node = 0

    gnn_model = GNNWithBellmanFord(num_nodes, num_features=4, num_classes=2)
    output, has_negative_cycle = gnn_model(adj_matrix, source_node)

    if has_negative_cycle:
        print("Example 3: The graph contains a negative weight cycle")
    else:
        print("Example 3: GNN output:", output)

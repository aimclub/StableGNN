import torch
import torch.nn as nn

class BellmanFordLayer(nn.Module):
    def __init__(self, num_nodes):
        super(BellmanFordLayer, self).__init__()
        self.num_nodes = num_nodes

    def forward(self, adj_matrix, source_node):
        distances = torch.full((self.num_nodes, self.num_nodes), float('inf'))
        predecessors = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.long)

        distances[source_node, 0] = 0

        for i in range(1, self.num_nodes):
            for u in range(self.num_nodes):
                for v in range(self.num_nodes):
                    w = adj_matrix[u, v]
                    if distances[u, i - 1] + w < distances[v, i]:
                        distances[v, i] = distances[u, i - 1] + w
                        predecessors[v, i] = u

        has_negative_cycle = False
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                w = adj_matrix[u, v]
                if distances[u, self.num_nodes - 1] + w < distances[v, self.num_nodes - 1]:
                    has_negative_cycle = True
                    break
            if has_negative_cycle:
                break

        return distances, predecessors, has_negative_cycle

if __name__ == "__main__":
    num_nodes = 4
    source_node = 0

    adj_matrix1 = torch.tensor([[0, 2, 1, float('inf')],
                               [float('inf'), 0, -1, float('inf')],
                               [float('inf'), float('inf'), 0, -2],
                               [float('inf'), float('inf'), float('inf'), 0]])

    adj_matrix2 = torch.tensor([[0, 2, float('inf'), 1],
                               [1, 0, -1, float('inf')],
                               [float('inf'), float('inf'), 0, -2],
                               [float('inf'), 1, float('inf'), 0]])

    adj_matrix3 = torch.tensor([[0, 2, 1, float('inf')],
                               [float('inf'), 0, -1, float('inf')],
                               [3, float('inf'), 0, -2],
                               [float('inf'), 1, float('inf'), 0]])

    bellman_ford_layer = BellmanFordLayer(num_nodes)

    distances1, predecessors1, has_negative_cycle1 = bellman_ford_layer(adj_matrix1, source_node)
    distances2, predecessors2, has_negative_cycle2 = bellman_ford_layer(adj_matrix2, source_node)
    distances3, predecessors3, has_negative_cycle3 = bellman_ford_layer(adj_matrix3, source_node)

    if has_negative_cycle1:
        print("Example 1: The graph contains a negative weight cycle")
    else:
        print("Example 1: Shortest distances from the source node:", distances1[source_node, -1])

    if has_negative_cycle2:
        print("Example 2: The graph contains a negative weight cycle")
    else:
        print("Example 2: Shortest distances from the source node:", distances2[source_node, -1])

    if has_negative_cycle3:
        print("Example 3: The graph contains a negative weight cycle")
    else:
        print("Example 3: Shortest distances from the source node:", distances3[source_node, -1])

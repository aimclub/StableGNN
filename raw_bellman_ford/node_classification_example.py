import torch
import torch.nn as nn
import torch.optim as optim
from layers.bellman_ford_orig import BellmanFordLayer

class NodeClassificationGNN(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes):
        super(NodeClassificationGNN, self).__init__()
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
    source_node = 0

    adj_matrix = torch.tensor([[0, 2, 0, 1],
                               [0, 0, -1, 0],
                               [0, 0, 0, -2],
                               [0, 0, 0, 0]])

    gnn_model = NodeClassificationGNN(num_nodes, num_features=5, num_classes=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, has_negative_cycle = gnn_model(adj_matrix, source_node)

        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Negative Cycle: {has_negative_cycle}')

    test_adj_matrix = torch.tensor([[0, 1, 1, 0],
                                   [0, 0, 0, 1],
                                   [1, 0, 0, 0],
                                   [0, 0, 1, 0]])

    test_source_node = 0

    test_output, has_negative_cycle = gnn_model(test_adj_matrix, test_source_node)
    print("Test Output:", test_output, "Negative Cycle:", has_negative_cycle)

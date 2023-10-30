import torch
import torch.nn as nn
import torch.optim as optim
from layers.bellman_ford_modified import BellmanFordLayerModified

class GraphPathPredictionModel(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_dim):
        super(GraphPathPredictionModel, self).__init__()
        
        self.bellman_ford_layer = BellmanFordLayerModified(num_nodes, num_features)        
        self.linear = nn.Linear(num_features + 1, hidden_dim)
        
    def forward(self, adj_matrix, source_node):
        node_features, _, _ = self.bellman_ford_layer(adj_matrix, source_node)
        
        predictions = self.linear(node_features)
        
        return predictions

if __name__ == "__main__":
    num_nodes = 6
    num_features = 5
    hidden_dim = 2
    
    adj_matrix = torch.tensor([[0, 1, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]])
    source_node = 0
    
    model = GraphPathPredictionModel(num_nodes, num_features, hidden_dim)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    labels = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.float32).view(-1, 1).repeat(1, 2)
    
    num_epochs = 1000
    for epoch in range(num_epochs):
        predictions = model(adj_matrix, source_node)
        
        predictions = torch.sigmoid(predictions)
        
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    with torch.no_grad():
        predictions = model(adj_matrix, source_node)
        predicted_labels = (predictions > 0.5).type(torch.float32)
        print("Predicted Labels:", predicted_labels)

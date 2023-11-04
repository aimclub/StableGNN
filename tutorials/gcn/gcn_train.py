import json
import torch
from torch_geometric.data import Data
from stable_gnn.gcn import GCN
from torch.optim import Adam
from torch_geometric.utils import negative_sampling

# Parse each graph and add to a global graph
nodes = set()
links = []

with open('user_graphs.jsonl', 'r') as f:
    for line in f:
        data_dict = json.loads(line)
        nodes.update(node['id'] for node in data_dict['nodes'])
        links.extend((link['source'], link['target']) for link in data_dict['links'])

# Create a mapping for node names to integer indices
node_to_idx = {node: i for i, node in enumerate(nodes)}
idx_to_node = {i: node for node, i in node_to_idx.items()}

# Convert links to a format suitable for PyTorch Geometric
edge_index = [[node_to_idx[source], node_to_idx[target]] for source, target in links]
edge_index = torch.tensor(edge_index, dtype=torch.long)

# Create a PyTorch Geometric data object
data = Data(x=torch.randn(len(nodes), 1), edge_index=edge_index.t().contiguous())

# Create GCN model and optimizer
model = GCN(data.num_node_features, 32, 1)
optimizer = Adam(model.parameters(), lr=0.01)

# Define Loss Function
criterion = torch.nn.BCEWithLogitsLoss()


# Training function
def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    pos_out = out[data.edge_index[0]]
    pos_loss = criterion(pos_out, torch.ones(data.edge_index.shape[1], ).view(-1, 1))
    # Negative sampling
    neg_edge_index = negative_sampling(data.edge_index, num_neg_samples=data.edge_index.shape[1])
    neg_out = out[neg_edge_index[0]]
    neg_loss = criterion(neg_out, torch.zeros(neg_edge_index.shape[1], ).view(-1, 1))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss


# Train the model
for epoch in range(10000):
    loss = train(data, model, optimizer)
    print(f"Epoch: {epoch}, Loss: {loss}")

# Save the model and mappings
torch.save(model.state_dict(), 'model.pth')
torch.save(data, 'data.pth')

# Save all extracted node names
with open('node_to_idx.json', 'w') as f:
    json.dump(node_to_idx, f)
with open('idx_to_node.json', 'w') as f:
    json.dump(idx_to_node, f)

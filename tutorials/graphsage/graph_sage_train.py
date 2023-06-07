# First, you need to download the dataset from the remote server
# > wget https://raw.githubusercontent.com/ZhongTr0n/JD_Analysis/main/jd_data2.json -O large_data.json

# Then create a symlink to the stable_gnn directory
# > ln -s ../../stable_gnn

import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch.optim import Adam
from stable_gnn.graph_sage import GraphSAGE

# Load dataset from file
with open('large_data.json', 'r') as f:
    data_dict = json.load(f)

# Extract node and links from dataset
nodes = data_dict['nodes']
links = data_dict['links']

# Parse large dataset
node_list = [node['id'] for node in nodes]
edge_index = [[node_list.index(link['source']), node_list.index(link['target'])] for link in links]
edge_index = torch.tensor(edge_index, dtype=torch.long)

# Create an object PyTorch Geometric by Dataset
data = Data(x=torch.randn(len(node_list), 1), edge_index=edge_index.t().contiguous())

# Create a model object
model = GraphSAGE(data.num_node_features, 32, 1)

# Обучение модели
optimizer = Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
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


for epoch in range(10000):
    loss = train()
    print(f"Epoch: {epoch}, Loss: {loss}")

# Save node names
with open('node_names.json', 'w') as f:
    json.dump(node_list, f)

# Saving model
torch.save(model.state_dict(), 'model.pth')
torch.save(data, 'data.pth')

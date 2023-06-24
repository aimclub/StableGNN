import json
import torch
import numpy as np
from torch_geometric.data import Data
from stable_gnn.graph_sage import GraphSAGE
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Avoid random
torch.manual_seed(42)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('user_json', type=str, help='JSON file with user technologies', default='user_data.json')
parser.add_argument('num_nodes', type=int, help='Number of nodes to add', default=1)
args = parser.parse_args()

# Load an object PyTorch Geometric by Dataset
data = torch.load('data.pth')

# Load a model
model = GraphSAGE(data.num_node_features, 32, 1)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Loading test Graph for prediction
with open(args.user_json, 'r') as f:
    user_data_dict = json.load(f)

# Prepare user's data
user_nodes = user_data_dict['nodes']
user_links = user_data_dict['links']
user_node_list = [node['id'] for node in user_nodes]
user_edge_index = [[user_node_list.index(link['source']), user_node_list.index(link['target'])] for link in user_links]
user_edge_index = torch.tensor(user_edge_index, dtype=torch.long)
user_data = Data(x=torch.randn(len(user_node_list), 1), edge_index=user_edge_index.t().contiguous())

# Make prediction
user_output = model(user_data.x, user_data.edge_index)
output = model(data.x, data.edge_index)

# Compute similarities between user's graph and large dataset
similarities = cosine_similarity(output.detach().numpy(), user_output.detach().numpy())

# Load node names
with open('node_names.json', 'r') as f:
    node_list = json.load(f)

added_nodes = []

for _ in range(args.num_nodes):
    # Prepare user's data
    user_nodes = user_data_dict['nodes']
    user_links = user_data_dict['links']
    user_node_list = [node['id'] for node in user_nodes]
    user_edge_index = [
        [user_node_list.index(link['source']), user_node_list.index(link['target'])]
        for link in user_links
    ]
    user_edge_index = torch.tensor(user_edge_index, dtype=torch.long)
    user_data = Data(
        x=torch.randn(len(user_node_list), 1),
        edge_index=user_edge_index.t().contiguous()
    )

    # Make prediction
    user_output = model(user_data.x, user_data.edge_index)
    output = model(data.x, data.edge_index)

    # Compute similarities between user's graph and large dataset
    similarities = cosine_similarity(output.detach().numpy(), user_output.detach().numpy())

    # Recommend a node
    recommended_node_index = np.argmax(similarities)
    recommended_node_name = node_list[recommended_node_index]

    # Check if the node already exists in user's graph
    if recommended_node_name in user_node_list:
        print(f"The node {recommended_node_name} already exists in the user's graph. Skipping...")
        continue

    # Save the added node
    added_nodes.append(recommended_node_name)

    # Add the node and edge to the user graph
    user_data_dict['nodes'].append({'id': recommended_node_name})

    # Find the closest node to the recommended node in the user's graph
    similarities_with_recommended = cosine_similarity(
        user_output.detach().numpy(),
        output.detach().numpy()[recommended_node_index].reshape(1, -1)
    )
    closest_node_in_user_graph_index = np.argmax(similarities_with_recommended)
    closest_node_in_user_graph_name = user_node_list[closest_node_in_user_graph_index]

    user_data_dict['links'].append({'source': closest_node_in_user_graph_name, 'target': recommended_node_name})

    print(f"The recommended node for the user is {recommended_node_name}")
    print(f"The recommended edge for the user is from {closest_node_in_user_graph_name} to {recommended_node_name}")

# Save the updated user graph
with open('updated_' + args.user_json, 'w') as f:
    json.dump(user_data_dict, f)

# Save the sequence of added nodes
with open('added_nodes.json', 'w') as f:
    json.dump(added_nodes, f)

print(f"Updated user graph saved as 'updated_{args.user_json}'")
print(f"Sequence of added nodes saved as 'added_nodes.json'")

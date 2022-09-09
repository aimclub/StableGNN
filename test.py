import torch
import torch_geometric
from StableGNN.Graph import Graph

graph = Graph('Cora',d=64)
print(graph.num_nodes,graph.x,graph.x.shape, graph.y,graph.edge_index)
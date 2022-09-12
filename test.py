import torch
import torch_geometric
from StableGNN.Graph import Graph
import matplotlib.pyplot as plt

graph = Graph('Cora', d=64)
a_genuine = graph.adjust()
print(torch.sum(a_genuine),graph.edge_index.shape)
#print(graph.num_nodes,graph.x,graph.x.shape, graph.y,graph.edge_index)

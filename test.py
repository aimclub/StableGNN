import torch
import torch_geometric
from StableGNN.Graph import Graph
import matplotlib.pyplot as plt
from datetime import datetime
from TrainModel import TrainModel
import random

dt = datetime.now()
#graph = Graph('Cora2', d=64)

#print(graph.edge_index.shape)
#graph.adjust()
#print(graph.edge_index.shape)
#print(graph.num_nodes,graph.x,graph.x.shape, graph.y,graph.edge_index)
#print(datetime.now()-dt)


#######
#MO = TrainModelOptuna(name=name, conv=conv, device=device, loss_function=loss, ')
#best_values = MO.run(number_of_trials=500)

M = TrainModel(name='Cora2', conv='GAT', device='cuda')
best_values={'hidden_layer': 32, 'dropout': 0.0, 'size of network, number of convs': 3, 'lr': 0.001}
train_acc_mi, test_acc_mi, train_acc_ma, test_acc_ma = M.run(best_values)

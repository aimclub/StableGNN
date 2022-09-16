import torch
import torch_geometric
from StableGNN.Graph import Graph
import matplotlib.pyplot as plt
from datetime import datetime
from TrainModel import TrainModel
import random

dt = datetime.now()

name = 'Cora2'
conv = 'GAT'
device = 'cuda'
adjust = True

#######
#MO = TrainModelOptuna(name=name, conv=conv, device=device,adjust = adjust)
#best_values = MO.run(number_of_trials=500)

M = TrainModel(name=name, conv=conv, device=device,adjust_flag = adjust)
best_values = {'hidden_layer': 32, 'dropout': 0.0, 'size of network, number of convs': 3, 'lr': 0.001,"number of negative samples for graph.adjust":5}
train_acc_mi, test_acc_mi, train_acc_ma, test_acc_ma = M.run(best_values)

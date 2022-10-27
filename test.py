import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj

from TrainModel import TrainModel, TrainModelOptuna
from StableGNN.Explain import Explain
from StableGNN.Graph import Graph


dt = datetime.now()

name = "Cora2"
conv = "GAT"
device = "cuda"
adjust = False
root = "DataValidation/"
####
data = Graph(
    name, root=root + str(name), transform=T.NormalizeFeatures(), ADJUST_FLAG=adjust
)[0]
# TODO number of negative samples for graph.adjust было неоптимизировано поскольку каждый раз данные считывались из одной и той же папки processed

#######
TRAIN_FLAG = False
if TRAIN_FLAG:
    MO = TrainModelOptuna(data=data, conv=conv, device=device, ADJUST_FLAG=adjust)
    best_values = MO.run(number_of_trials=200)

    M = TrainModel(data=data, conv=conv, device=device, ADJUST_FLAG = adjust)
    best_values = {'hidden_layer': 32, 'dropout': 0.0, 'size of network, number of convs': 3, 'lr': 0.001,"number of negative samples for graph.adjust":5}
    model, train_acc_mi, test_acc_mi, train_acc_ma, test_acc_ma = M.run(best_values)
    torch.save(model, 'model.pt')
model = torch.load("model.pt")


print(model)


MODEL_FLAG = False
if MODEL_FLAG:
    ori_pred = 3
    num_layers = len(model.convs)
    X = np.load(root + name + "/X.npy")
    try:
        A = np.load(root + name + "/A.npy")
    except:
        A = torch.squeeze(to_dense_adj(data.edge_index)).numpy()

    explainer = Explain(
        model=model,
        A=A,
        X=X,
        ori_pred=ori_pred,
        num_layers=num_layers,
        mode=0,
        print_result=1,
    )

    data, neighbors = explainer.DataGeneration(708)
    pgm_nodes, pgm_stats = explainer.VariableSelection(data,neighbors,708)
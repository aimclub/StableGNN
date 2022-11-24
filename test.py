import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj

from TrainModel import TrainModel, TrainModelOptuna
from TrainModel import TrainModel, TrainModelOptuna
from StableGNN.Explain import Explain
from StableGNN.Graph import Graph


dt = datetime.now()

name = "wisconsin"

conv = "GAT"
device = torch.device("cuda",0)
ADJUST_FLAG = False
SSL_FLAG = True
EXTRAPOLATE_FLAG = True
root = "DataValidation/"
####
data = Graph(
    name, root=root + str(name), transform=T.NormalizeFeatures(), ADJUST_FLAG=ADJUST_FLAG
)[0]
# TODO number of negative samples for graph.adjust было неоптимизировано поскольку каждый раз данные считывались из одной и той же папки processed

#######
TRAIN_FLAG = True
if TRAIN_FLAG:
   # MO = TrainModelOptuna(data=data, conv=conv, device=device, SSL = SSL_FLAG, EXTRAPOLATE = EXTRAPOLATE_FLAG)

    best_values ={'hidden_layer': 256, 'dropout': 0.2, 'size of network, number of convs': 1, 'lr': 0.007596327641645819}#MO.run(number_of_trials=36)

    M = TrainModel(data=data, conv=conv, device=device, SSL=SSL_FLAG)
    best_values = {'hidden_layer': 32, 'dropout': 0.0, 'size of network, number of convs': 3, 'lr': 0.001,"number of negative samples for graph.adjust":5}
    model, test_acc_mi, test_acc_ma = M.run(best_values)
    torch.save(model, 'model.pt')
model = torch.load("model.pt")


print(model)


EXPLAIN_FLAG = False
if EXPLAIN_FLAG:
    num_layers = len(model.convs)
    X = np.load(root + name + "/X.npy")
    try:
        A = np.load(root + name + "/A.npy")
    except:
        A = torch.squeeze(to_dense_adj(data.edge_index)).numpy()

    explainer = Explain(
        model = model,
        A = A,
        X = X,
        num_layers = num_layers,
        mode = 0,
        print_result = 1,
    )

    data, neighbors = explainer.DataGeneration(node_idx=28, num_samples=20)
    subnodes, data, pgm_stats = explainer.VariableSelection(data,neighbors,28)
    pgm_explanation = explainer.StructureLearning(28, data, subnodes,child='yes')
    print('explanations is 1', pgm_explanation.nodes(), pgm_explanation.edges())
    pgm_explanation = explainer.StructureLearning_bamt(28, data, subnodes)
    print('explanations is',pgm_explanation.nodes,pgm_explanation.edges)


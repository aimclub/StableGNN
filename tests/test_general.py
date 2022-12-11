from datetime import datetime

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj

from stable_gnn.explain import Explain
from stable_gnn.graph import Graph
from tutorials.train_model import TrainModel, TrainModelOptuna

dt = datetime.now()

name = "wisconsin"
conv = "GAT"
task = "NC"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adjust_flag = False

ssl_flag = False
extrapolate_flag = False
root = "../data_validation/"
####

data = Graph(name, root=root + str(name), transform=T.NormalizeFeatures(), adjust_flag=adjust_flag)
print("i have read data")

#######
train_flag = True
if train_flag:
    optuna_training = TrainModelOptuna(
        data=data,
        dataset_name=name,
        conv=conv,
        device=device,
        ssl_flag=ssl_flag,
        extrapolate_flag=extrapolate_flag,
        task=task,
    )
    best_values = optuna_training.run(number_of_trials=3)
    model_training = TrainModel(
        data=data,
        conv=conv,
        dataset_name=name,
        device=device,
        ssl_flag=ssl_flag,
        task=task,
        extrapolate_flag=extrapolate_flag,
    )

    # best_values = {'hidden_layer': 32, 'dropout': 0.0, 'size of network, number of convs': 3, 'lr': 0.001,"number of negative samples for graph.adjust":5}
    model, test_acc_mi, test_acc_ma = model_training.run(best_values)
    torch.save(model, "model.pt")
model = torch.load("model.pt")
print(model)

explain_flag = False
if explain_flag:

    features = np.load(root + name + "/X.npy")
    try:
        adj_matrix = np.load(root + name + "/A.npy")
    except:
        A = torch.squeeze(to_dense_adj(data.edge_index)).numpy()

    explainer = Explain(model=model, adj_matrix=adj_matrix, features=features)

    pgm_explanation = explainer.structure_learning(28)
    print("explanations is 1", pgm_explanation.nodes(), pgm_explanation.edges())
    pgm_explanation = explainer.structure_learning_bamt(28)
    print("explanations is", pgm_explanation.nodes, pgm_explanation.edges)

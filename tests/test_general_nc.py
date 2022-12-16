import collections
import os

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj

from stable_gnn.explain import Explain
from stable_gnn.graph import Graph
from stable_gnn.train_model_pipeline import TrainModelNC, TrainModelOptunaNC


def test_general_nc():

    name = "wisconsin"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adjust_flag = False
    loss_name = "APP"  # APP, LINE, HOPE_AA, VERSE_Adj

    ssl_flag = False
    root = "../data_validation/"
    ####

    dataset = Graph(name, root=root + str(name), transform=T.NormalizeFeatures(), adjust_flag=adjust_flag)
    data = dataset[0]
    assert (max(float(data.edge_index[0].max()), float(data.edge_index[1].max())) + 1) == len(data.x) == data.num_nodes
    assert len(collections.Counter((data.y).tolist())) == 5
    assert data.x.shape[1] == 1703

    #######
    train_flag = True
    if train_flag:
        optuna_training = TrainModelOptunaNC(data=dataset, device=device, ssl_flag=ssl_flag, loss_name=loss_name)

        best_values = optuna_training.run(number_of_trials=10)
        model_training = TrainModelNC(data=dataset, device=device, ssl_flag=ssl_flag, loss_name=loss_name)

        model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma = model_training.run(best_values)
        torch.save(model, root + str(name) + "/model.pt")
        print(train_acc_mi, test_acc_mi)
        assert train_acc_mi > test_acc_mi
        assert np.isclose(
            test_acc_mi, 0.5, atol=0.1
        )  # это для loss_name=APP, для остальных там другие значения, меньше
        assert np.isclose(train_acc_mi, 0.9, atol=0.1)

    model = torch.load(root + str(name) + "/model.pt")
    explain_flag = True
    if explain_flag:
        features = np.load(root + name + "/X.npy")
        if os.path.exists(root + name + "/A.npy"):
            adj_matrix = np.load(root + name + "/A.npy")
        else:
            adj_matrix = torch.squeeze(to_dense_adj(data.edge_index.cpu())).numpy()

        explainer = Explain(model=model, adj_matrix=adj_matrix, features=features)
        pgm_explanation = explainer.structure_learning(2)
        assert len(pgm_explanation.nodes) >= 2
        assert len(pgm_explanation.edges) >= 1
    print("explanations is", pgm_explanation.nodes, pgm_explanation.edges)

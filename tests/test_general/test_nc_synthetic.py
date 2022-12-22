import os
import pathlib

import numpy as np
import pytest
import torch
from torch_geometric.utils import to_dense_adj

from stable_gnn.explain import Explain
from stable_gnn.graph import Graph
from stable_gnn.pipelines.node_classification_pipeline import TrainModelNC
from tests.data_generators import generate_star_graphs

root = str(pathlib.Path(__file__).parent.resolve().joinpath("data_validation/")) + "/"
generate_star_graphs(root, 5)


@pytest.mark.parametrize("ssl_flag", [False, True])
@pytest.mark.parametrize("conv", ["SAGE", "GAT", "GCN"])
@pytest.mark.parametrize("loss_name", ["APP", "LINE", "HOPE_AA", "VERSE_Adj"])
@pytest.mark.parametrize("adjust_flag", [False, True])
def test_explain(ssl_flag: bool, conv: str, loss_name: str, adjust_flag: bool) -> None:

    name = "stars"
    data = Graph(root=root + name + "/", name=name, adjust_flag=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #######
    best_values = {"hidden_layer": 64, "size of network, number of convs": 3, "dropout": 0.0, "lr": 0.01, "coef": 10}
    model_training = TrainModelNC(data=data, device=device, ssl_flag=ssl_flag, loss_name=loss_name, emb_conv=conv)

    model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma = model_training.run(best_values)

    features = np.load(root + name + "/X.npy")
    if os.path.exists(root + name + "/A.npy"):
        adj_matrix = np.load(root + name + "/A.npy")
    else:
        adj_matrix = torch.squeeze(to_dense_adj(data[0].edge_index.cpu())).numpy()

    explainer = Explain(model=model, adj_matrix=adj_matrix, features=features)

    pgm_explanation_star = explainer.structure_learning(0)
    assert len(pgm_explanation_star.nodes) >= 0
    assert len(pgm_explanation_star.edges) >= 0
    print("explanations is", pgm_explanation_star.nodes, pgm_explanation_star.edges)

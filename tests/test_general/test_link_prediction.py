import pathlib

import torch

from stable_gnn.graph import Graph
from stable_gnn.pipelines.graph_classification_pipeline import TrainModelGC
from tests.data_generators import generate_gc_graphs

root = str(pathlib.Path(__file__).parent.resolve().joinpath("data_validation/")) + "/"
generate_gc_graphs(root, 30)

def test_linkpredict():
    name = "ba_gc"
    data = Graph(root=root + name + "/", name=name, adjust_flag=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_flag = False
    extrapolate_flag = True

    #######
    best_values = {"hidden_layer": 64, "size of network, number of convs": 2, "dropout": 0.0, "lr": 0.01, "coef": 10}
    model_training = TrainModelGC(data=data, device=device, ssl_flag=ssl_flag, extrapolate_flag=extrapolate_flag)

    model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma = model_training.run(best_values)
    torch.save(model, "model.pt")
    assert train_acc_mi >= test_acc_mi
    assert train_acc_ma >= test_acc_ma
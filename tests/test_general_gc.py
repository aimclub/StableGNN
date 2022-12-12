from datetime import datetime

import torch
import torch_geometric.transforms as T

from stable_gnn.graph import Graph
from tutorials.train_model_pipeline import TrainModelGC, TrainModelOptunaGC

dt = datetime.now()

name = "BACE"
conv = "GAT"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssl_flag = False
extrapolate_flag = False
root = "../data_validation/"
####

data = Graph(name, root=root + str(name), transform=T.NormalizeFeatures())

#######
train_flag = False
if train_flag:
    optuna_training = TrainModelOptunaGC(
        data=data,
        dataset_name=name,
        conv=conv,
        device=device,
        ssl_flag=ssl_flag,
        extrapolate_flag=extrapolate_flag,
    )
    best_values = optuna_training.run(number_of_trials=3)
    model_training = TrainModelGC(
        data=data,
        conv=conv,
        dataset_name=name,
        device=device,
        ssl_flag=ssl_flag,
        extrapolate_flag=extrapolate_flag,
    )

    model, test_acc_mi, test_acc_ma = model_training.run(best_values)
    torch.save(model, "model.pt")
model = torch.load("model.pt")
print(model)

import pathlib

import torch
import torch_geometric.transforms as T

from stable_gnn.graph import Graph
from stable_gnn.pipelines.graph_classification_pipeline import TrainModelGC, TrainModelOptunaGC

if __name__ == "__main__":
    name = "BACE"  # всего там 1000 файлов, но они маленькие, качается довольно долго
    conv = "GAT"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_flag = False
    extrapolate_flag = False
    root = str(pathlib.Path(__file__).parent.resolve().joinpath("data_validation/")) + "/"
    ####

    data = Graph(name, root=root + str(name), transform=T.NormalizeFeatures())
    print("i have read data")

    #######
    train_flag = True
    if train_flag:
        optuna_training = TrainModelOptunaGC(
            data=data,
            conv=conv,
            device=device,
            ssl_flag=ssl_flag,
            extrapolate_flag=extrapolate_flag,
        )

        best_values = optuna_training.run(number_of_trials=10)
        model_training = TrainModelGC(
            data=data,
            conv=conv,
            device=device,
            ssl_flag=ssl_flag,
            extrapolate_flag=extrapolate_flag,
        )

        model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma = model_training.run(
            best_values, plot_training_procces=True
        )
        print(test_acc_mi)

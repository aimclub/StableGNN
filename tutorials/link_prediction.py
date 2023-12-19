import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from stable_gnn.model_link_predict import ModelLinkPrediction
import numpy as np

if __name__ == "__main__":
    root = "../tmp/"
    name = "Citeseer"
    loss_name = 'APP'
    conv = 'SAGE'

    dataset = Planetoid(root=root + str(name), name=name, transform=T.NormalizeFeatures())
    k = []
    for _ in range(10):
        model = ModelLinkPrediction(number_of_trials=50, loss_name=loss_name, emb_conv_name=conv)

        train_edges, train_negative, test_edges, test_negative = model.train_test_edges(dataset)

        cl = model.train_cl(
            train_edges, train_negative
        )  # MLPClassifier()#GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=5, random_state=0)

        k.append(model.test(cl, test_edges, test_negative))
    print(np.mean(k),np.std(k))
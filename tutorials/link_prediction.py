import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from stable_gnn.model_link_predict import ModelLinkPrediction
from stable_gnn.graph import Graph
import pathlib


if __name__ == "__main__":
    
    data = Planetoid(root="/tmp/" + str("name"), name="Cora", transform=T.NormalizeFeatures())
    model = ModelLinkPrediction(data, number_of_trials=1)
    clf = model.train_cl()
    print("f1 measure", (model.test()))
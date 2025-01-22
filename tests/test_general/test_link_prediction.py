import pytest
import torch
import torch_geometric.transforms as T
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from torch_geometric.datasets import Planetoid

from stable_gnn.model_link_predict import ModelLinkPrediction

@pytest.mark.parametrize("conv", ["SAGE", "GAT", "GCN"])
@pytest.mark.parametrize("loss_name", ["APP", "LINE", "HOPE_AA", "VERSE_Adj"])
def test_linkpredict(loss_name: str, conv: str) -> None:
    root = "../tmp/"
    name = "Cora"
    dataset = Planetoid(root=root + str(name), name=name, transform=T.NormalizeFeatures())

    model_before = ModelLinkPrediction(number_of_trials=0, loss_name=loss_name, emb_conv_name=conv)
    model_after = ModelLinkPrediction(number_of_trials=10, loss_name=loss_name, emb_conv_name=conv)

    # Преобразование данных в тензоры
    train_edges_b, train_negative_b, test_edges_b, test_negative_b = [
        torch.tensor(edge, dtype=torch.long) for edge in model_before.train_test_edges(dataset)
    ]
    print(f"train_edges_b: {train_edges_b.size()}, test_edges_b: {test_edges_b.size()}")

    train_edges, train_negative, test_edges, test_negative = [
        torch.tensor(edge, dtype=torch.long) for edge in model_after.train_test_edges(dataset)
    ]
    print(f"train_edges: {train_edges.size()}, test_edges: {test_edges.size()}")

    # Обучение классификаторов
    cl_before = model_before.train_cl(train_edges_b, train_negative_b)
    cl_after = model_after.train_cl(train_edges, train_negative)

    # Тестирование модели
    acc_before = model_before.test(cl_before, test_edges_b, test_negative_b)
    acc_after = model_after.test(cl_after, test_edges, test_negative)

    assert acc_before < acc_after, f"Expected improvement, but got acc_before={acc_before} and acc_after={acc_after}"

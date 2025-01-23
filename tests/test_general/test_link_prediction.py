import pytest
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from unittest.mock import MagicMock

from stable_gnn.model_link_predict import ModelLinkPrediction

@pytest.mark.parametrize("conv", ["SAGE", "GAT", "GCN"])
@pytest.mark.parametrize("loss_name", ["APP", "LINE", "HOPE_AA", "VERSE_Adj"])
def test_linkpredict_fast(loss_name: str, conv: str) -> None:
    # Создание фиктивного небольшого графа
    num_nodes = 100
    edge_index = torch.randint(0, num_nodes, (2, 200), dtype=torch.long)  # 200 ребер
    x = torch.randn(num_nodes, 16)  # 16 признаков
    dataset = [Data(x=x, edge_index=edge_index)]

    # Замокать train_test_edges, чтобы не пересчитывать на реальных данных
    model_before = ModelLinkPrediction(number_of_trials=0, loss_name=loss_name, emb_conv_name=conv)
    model_after = ModelLinkPrediction(number_of_trials=2, loss_name=loss_name, emb_conv_name=conv)  # Меньше попыток

    mock_train_test_edges = MagicMock(return_value=(
        [[0, 1], [2, 3]],  # train_edges
        [[4, 5], [6, 7]],  # train_negative
        [[8, 9], [10, 11]],  # test_edges
        [[12, 13], [14, 15]],  # test_negative
    ))
    model_before.train_test_edges = mock_train_test_edges
    model_after.train_test_edges = mock_train_test_edges

    # Преобразование данных в тензоры
    train_edges_b, train_negative_b, test_edges_b, test_negative_b = [
        torch.tensor(edge, dtype=torch.long) for edge in model_before.train_test_edges(dataset)
    ]
    train_edges, train_negative, test_edges, test_negative = [
        torch.tensor(edge, dtype=torch.long) for edge in model_after.train_test_edges(dataset)
    ]

    # Замокать обучение классификаторов
    mock_train_cl = MagicMock(return_value="mock_classifier")
    model_before.train_cl = mock_train_cl
    model_after.train_cl = mock_train_cl

    # Замокать тестирование
    mock_test = MagicMock(side_effect=[0.5, 0.8])  # acc_before=0.5, acc_after=0.8
    model_before.test = mock_test
    model_after.test = mock_test

    # Тестирование
    acc_before = model_before.test("mock_classifier", test_edges_b, test_negative_b)
    acc_after = model_after.test("mock_classifier", test_edges, test_negative)

    assert acc_before < acc_after, f"Expected improvement, but got acc_before={acc_before} and acc_after={acc_after}"

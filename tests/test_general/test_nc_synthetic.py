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

# Генерация данных для тестов
root = pathlib.Path(__file__).parent.resolve() / "data_validation"
generate_star_graphs(str(root), 5)

@pytest.fixture(scope="module", autouse=True)
def prepare_tutorial_files():
    tutorials_path = pathlib.Path(__file__).parent.parent / "tutorials"
    os.makedirs(tutorials_path, exist_ok=True)
    
    embedding_files = [
        "embeddings_APP_SAGE.npy",
        "embeddings_LINE_SAGE.npy",
        "embeddings_HOPE_AA_SAGE.npy",
        "embeddings_VERSE_Adj_SAGE.npy",
    ]
    
    for file_name in embedding_files:
        file_path = tutorials_path / file_name
        if not file_path.exists():
            np.save(file_path, np.random.rand(100, 16))


@pytest.mark.parametrize("ssl_flag", [False, True])
@pytest.mark.parametrize("conv", ["SAGE", "GAT", "GCN"])
@pytest.mark.parametrize("loss_name", ["APP", "LINE", "HOPE_AA", "VERSE_Adj"])
@pytest.mark.parametrize("adjust_flag", [False, True])
def test_explain(ssl_flag: bool, conv: str, loss_name: str, adjust_flag: bool) -> None:
    name = "stars"
    data_path = root / name

    try:
        # Загрузка данных
        features = np.load(data_path / "X.npy")
        adj_matrix = np.load(data_path / "A.npy")

        # Приведение фичей к размерности 128
        if features.shape[1] != 128:
            if features.shape[1] < 128:
                diff = 128 - features.shape[1]
                additional_features = np.zeros((features.shape[0], diff))
                features = np.hstack([features, additional_features])
            else:
                features = features[:, :128]

        assert features.shape[1] == 128, (
            f"Feature dimension mismatch: got {features.shape[1]}, expected 128"
        )

        # Приведение adj_matrix к плотному формату и проверка
        adj_matrix = to_dense_adj(
            edge_index=torch.tensor(adj_matrix, dtype=torch.long),  # edge_index должен быть long
            batch=torch.tensor(np.zeros(adj_matrix.shape[0], dtype=np.int64))  # batch должен быть int64
        ).squeeze(0)
        assert adj_matrix.shape[0] == adj_matrix.shape[1], "Adjacency matrix must be square."

    except Exception as e:
        pytest.fail(f"Data loading failed: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Настройка модели
    input_dim = features.shape[1]
    best_values = {
        "hidden_layer": input_dim,
        "size of network, number of convs": 3,
        "dropout": 0.0,
        "lr": 0.01,
        "coef": 10,
    }

    try:
        data = Graph(root=str(data_path), name=name, adjust_flag=adjust_flag)

        # Приведение данных к корректным типам
        data.features = torch.tensor(features, dtype=torch.float32, device=device)
        data.adj = torch.tensor(adj_matrix, dtype=torch.float32, device=device)
        
        # Приведение индексов к int64
        if hasattr(data, 'edge_index'):
            data.edge_index = data.edge_index.long()

        model_training = TrainModelNC(
            data=data,
            device=device,
            ssl_flag=ssl_flag,
            loss_name=loss_name,
            emb_conv=conv,
        )

        model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma = model_training.run(best_values)

        # Проверяем метрики обучения
        assert train_acc_mi > 0.5, f"Low train accuracy: {train_acc_mi}"
        assert test_acc_mi > 0.5, f"Low test accuracy: {test_acc_mi}"

    except Exception as e:
        pytest.fail(f"Model training failed: {e}")

import torch_geometric.transforms as T
from stable_gnn.model_link_predict import ModelLinkPrediction
from torch_geometric.datasets import Planetoid
from sklearn.ensemble import GradientBoostingClassifier
import pytest

@pytest.mark.parametrize("conv", ["SAGE", "GAT", "GCN"])
@pytest.mark.parametrize("loss_name", ["APP", "LINE", "HOPE_AA", "VERSE_Adj"])
def test_linkpredict(loss_name: str, conv: str) -> None:
    root = '../tmp/'
    name = 'Cora'
    dataset = Planetoid(root=root + str(name), name=name, transform=T.NormalizeFeatures())

    model = ModelLinkPrediction(number_of_trials=50, loss_name=loss_name, emb_conv_name=conv)

    train_edges, train_negative, test_edges, test_negative = model.train_test_edges(dataset)

    cl_before = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=5, random_state=0)
    cl_after = model.train_cl(train_edges, train_negative)
    assert (model.test(cl_before, test_edges, test_negative)) < (model.test(cl_after, test_edges, test_negative))
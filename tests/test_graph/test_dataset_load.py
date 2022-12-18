import collections

import torch_geometric.transforms as T
from pytest import mark

from stable_gnn.graph import Graph


@mark.parametrize("name", ["wisconsin", "BACE"])
@mark.parametrize("adjust_flag", [False, True])
def test_autoload_graph(name, adjust_flag):
    map_y_dim = {"wisconsin": 5, "BACE": 1}
    map_x_dim = {"wisconsin": 1703, "BACE": 9}
    root = "../data_validation/"

    dataset = Graph(name, root=root + str(name), transform=T.NormalizeFeatures(), adjust_flag=adjust_flag)
    data = dataset[0]
    assert (max(float(data.edge_index[0].max()), float(data.edge_index[1].max())) + 1) == len(data.x) == data.num_nodes
    assert len(collections.Counter(data.y.tolist())) == map_y_dim[name]
    assert data.x.shape[1] == map_x_dim[name]

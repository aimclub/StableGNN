import collections

import torch_geometric.transforms as T

from stable_gnn.graph import Graph


def test_autoload_graph():
    names = ["wisconsin", "BACE"]
    map_y_dim = {"wisconsin": 5, "BACE": 1}
    map_x_dim = {"wisconsin": 1703, "BACE": 9}
    adjust_flag = False
    root = "../data_validation/"
    ####

    for name in names:
        print(name)
        dataset = Graph(name, root=root + str(name), transform=T.NormalizeFeatures(), adjust_flag=adjust_flag)
        data = dataset[0]
        assert (
            (max(float(data.edge_index[0].max()), float(data.edge_index[1].max())) + 1) == len(data.x) == data.num_nodes
        )
        assert len(collections.Counter(data.y.tolist())) == map_y_dim[name]
        assert data.x.shape[1] == map_x_dim[name]

import collections
import os

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj

from stable_gnn.explain import Explain
from stable_gnn.graph import Graph
from stable_gnn.train_model_pipeline import TrainModelNC, TrainModelOptunaNC
import networkx.generators as gen
import networkx as nx

def test_explain():
    root='../data_validation/'
    name = 'stars'

    if not os.path.exists(root+str(name)):

        size_of_star = 5
        num_of_stars = 20
        G = nx.DiGraph()
        prev_nodes = 0
        central_nodes = []
        for num in range(num_of_stars):
            g = gen.star_graph(size_of_star)
            mapping = {}
            for i in range(len(g.nodes())):
                mapping[i] = i + prev_nodes
            central_nodes.append(prev_nodes)
            prev_nodes += len(g.nodes())

            for edge in g.edges():
                G.add_edge(mapping[edge[0]], mapping[edge[1]])

        for node in central_nodes:
            for node2 in central_nodes:
                if node > node2:
                    G.add_edge(node, node2)

        path_to_dir = '../data_validation/stars/'
        if not os.path.exists('../data_validation/'):
            os.mkdir('../data_validation/')
        if not os.path.exists(path_to_dir):
            os.mkdir(path_to_dir)
        if not os.path.exists(path_to_dir + 'raw'):
            os.mkdir(path_to_dir + 'raw')

        with open(path_to_dir + 'raw/' + 'labels.txt', 'a') as f:
            for i in (G.nodes()):
                if i in central_nodes:
                    f.write(str(1) + '\n')
                else:
                    f.write(str(0) + '\n')

        with open(path_to_dir + 'raw/' + 'edges.txt', 'a') as f:
            for i in (G.edges()):
                f.write(str(i[0]) + ',' + str(i[1]) + '\n')

    adjust_flag = False
    data = Graph(root='../data_validation/' + str(name), name=name,adjust_flag=adjust_flag)[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_name = "APP"  # APP, LINE, HOPE_AA, VERSE_Adj

    ssl_flag = False
    root = "../data_validation/"

    #######
    best_values = {'hidden_layer': 64, 'size of network, number of convs': 3, 'dropout': 0.0, 'lr': 0.01}
    model_training = TrainModelNC(
        data=data,
        dataset_name=name,
        device=device,
        ssl_flag=ssl_flag,
        loss_name=loss_name
    )

    model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma = model_training.run(best_values)
    torch.save(model, "model.pt")
    print(train_acc_mi, test_acc_mi)

    features = np.load(root + name + "/X.npy")
    try:
        adj_matrix = np.load(root + name + "/A.npy")
    except:
        adj_matrix = torch.squeeze(to_dense_adj(data.edge_index.cpu())).numpy()

    explainer = Explain(model=model, adj_matrix=adj_matrix, features=features)

    pgm_explanation_star = explainer.structure_learning(5)
    assert len(pgm_explanation_star.nodes) >= 2
    assert len(pgm_explanation_star.edges) >= 1
    print("explanations is", pgm_explanation_star.nodes, pgm_explanation_star.edges)
    pgm_explanation = explainer.structure_learning(6)
    assert len(pgm_explanation_star.nodes) > len(pgm_explanation.nodes)
    assert len(pgm_explanation_star.edges) > len(pgm_explanation.edges)
    print("explanations is", pgm_explanation.nodes, pgm_explanation.edges)



test_explain()
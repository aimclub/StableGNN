import os

import networkx as nx
import networkx.generators as gen
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

from stable_gnn.explain import Explain
from stable_gnn.graph import Graph
from stable_gnn.pipelines.node_classification_pipeline import TrainModelNC


def test_explain():
    root = "../data_validation/"
    name = "stars"

    if not os.path.exists(root + str(name)):

        size_of_star = 5
        num_of_stars = 20
        graph = nx.DiGraph()
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
                graph.add_edge(mapping[edge[0]], mapping[edge[1]])

        for node in central_nodes:
            for node2 in central_nodes:
                if node > node2:
                    graph.add_edge(node, node2)

        path_to_dir = "../data_validation/stars/"
        if not os.path.exists("../data_validation/"):
            os.mkdir("../data_validation/")
        if not os.path.exists(path_to_dir):
            os.mkdir(path_to_dir)
        if not os.path.exists(path_to_dir + "raw"):
            os.mkdir(path_to_dir + "raw")

        with open(path_to_dir + "raw/" + "labels.txt", "a") as f:
            for i in graph.nodes():
                if i in central_nodes:
                    f.write(str(1) + "\n")
                else:
                    f.write(str(0) + "\n")

        with open(path_to_dir + "raw/" + "edges.txt", "a") as f:
            for i in graph.edges():
                f.write(str(i[0]) + "," + str(i[1]) + "\n")

    adjust_flag = False
    data = Graph(root="../data_validation/" + str(name), name=name, adjust_flag=adjust_flag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "../data_validation/"
    for ssl_flag in [False, True]:
        for conv in ["SAGE", "GAT", "GCN"]:
            for loss_name in ["APP", "LINE", "HOPE_AA", "VERSE_Adj"]:
                print(conv, loss_name, ssl_flag)
                #######
                best_values = {"hidden_layer": 64, "size of network, number of convs": 3, "dropout": 0.0, "lr": 0.01}
                model_training = TrainModelNC(
                    data=data, device=device, ssl_flag=ssl_flag, loss_name=loss_name, emb_conv=conv
                )

                model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma = model_training.run(best_values)
                torch.save(model, "../data_validation/" + str(name) + "/model.pt")
                print(train_acc_mi, test_acc_mi)

                features = np.load(root + name + "/X.npy")
                if os.path.exists(root + name + "/A.npy"):
                    adj_matrix = np.load(root + name + "/A.npy")
                else:
                    adj_matrix = torch.squeeze(to_dense_adj(data[0].edge_index.cpu())).numpy()

                explainer = Explain(model=model, adj_matrix=adj_matrix, features=features)

                pgm_explanation_star = explainer.structure_learning(0)
                assert len(pgm_explanation_star.nodes) >= 0
                assert len(pgm_explanation_star.edges) >= 0
                print("explanations is", pgm_explanation_star.nodes, pgm_explanation_star.edges)
                pgm_explanation = explainer.structure_learning(4)
                assert len(pgm_explanation.nodes) >= 0
                assert len(pgm_explanation.edges) >= 0

                print("explanations is", pgm_explanation.nodes, pgm_explanation.edges)

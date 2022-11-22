from torch_geometric.nn.conv import MessagePassing

import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv, ChebConv
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler, Data
from datetime import datetime
import collections
import numpy as np
from torch.nn import MSELoss
from torch.nn import Linear
from torch_geometric.data import ClusterData
from torch_geometric.utils import degree, to_dense_adj,dense_to_sparse
from pgmpy.estimators import K2Score
from torch_geometric.loader import NeighborSampler
from StableGNN.Graph import Graph
import pandas as pd
import bamt.Networks as Nets
from bamt.Preprocessors import Preprocessor
from sklearn import preprocessing
from torch_geometric.nn import global_mean_pool

class ModelName(torch.nn.Module):
    def __init__(
        self, dataset, device, conv="GAT", hidden_layer=64, dropout=0, num_layers=2,SSL=False
    ):
        super(ModelName, self).__init__()
        self.conv = conv
        self.num_layers = num_layers
        self.data = dataset
        self.num_features = dataset.x.shape[1]
        # print(dataset.num_features)
        self.convs = torch.nn.ModuleList()
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.device = device
        self.SSL=SSL
        if self.SSL: #TODO: разобраться с SSL сейчас он считает true deg на изначальной матрице смежности а не модифицированной
            self.deg = degree(self.data.edge_index[0], self.data.num_nodes)
        

        num_classes = len(collections.Counter(self.data.y.tolist()))
        if self.conv == "GAT":
            if self.num_layers == 1:
                self.convs.append(GATConv(self.num_features, hidden_layer))
            else:
                self.convs.append(GATConv(self.num_features, self.hidden_layer))
                for i in range(1, self.num_layers):
                    self.convs.append(GATConv(self.hidden_layer, self.hidden_layer))

        self.linear = Linear(self.hidden_layer, int(self.hidden_layer/2))

        self.linear_classifier = Linear(int(self.hidden_layer/2), num_classes)
        self.linear_degree_predictor = Linear(int(self.hidden_layer/2), 1)
        self.linear_cluster_distance_predictor = Linear(int(self.hidden_layer / 2), 1)

    def forward(self, x, edge_index, edge_weight, batch): #TODO: add batchnorm after self.linear_layer
        # 1. Obtain node embeddings
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        x = F.relu(x)
        
        deg_pred=0
        #cluster_pred=0 #TODO если убираем - то удалить
        if self.SSL:
            deg_pred = F.relu(self.linear_degree_predictor(x))
            #cluster_pred =  F.relu(self.linear_cluster_distance_predictor(x))

        x = self.linear_classifier(x)
        return x.log_softmax(dim=1), deg_pred

    def inference(self, data, dp=0):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=dp, training=self.training)
        x = self.linear(x)
        x = F.relu(x)
        deg_pred=0
        cluster_pred=0 #TODO если убираем - то удалить
        if self.SSL:
            deg_pred =  F.relu(self.linear_degree_predictor(x))
            #cluster_pred =  F.relu(self.linear_cluster_distance_predictor(x))
        x=self.linear_classifier(x)
        return x.log_softmax(dim=-1), deg_pred

    def loss_sup(self, pred, label):
        return F.nll_loss(pred, label)

    def message(self):
        pass

    def VirtualVertex(self):
        pass

    def Extrapolate(self, train_mask, val_mask, init_edges, remove_init_edges,white_list, score_func):
        self.init_edges = init_edges
        self.remove_init_edges = remove_init_edges
        self.white_list = white_list
        self.score_func = score_func
        if score_func == "K2":
            self.score = K2Score
        elif score_func == "MI":
            self.score = None
        else:
            raise Exception(
                "there is no ", self.score_func, "score function. Choose one of: MI, K2"
            )

        train_dataset, test_dataset, n_min = self.convert_dataset(self.data, train_mask,val_mask)
        self.n_min = n_min

        data_bamt = self.data_eigen_exctractor(train_dataset)
        bn = self.bn_build(data_bamt)
        lis = list(
            map(lambda x: self.func(x), bn.edges)
        )  # мы берем только те веришны, которые исходят из y или входят в у
        left_vertices = sorted(list(filter(lambda x: not np.isnan(x), lis)))
        left_edges = list(filter(lambda x: x[0] == "y" or x[1] == "y", bn.edges))
        left_edges = sorted(
            left_edges, key=lambda x: int(x[0][5:] if x[1] == "y" else int(x[1][5:]))
        )
        ll = list(map(lambda x: bn.weights[tuple(x)], left_edges))
        N = len(
            ll
        )  # TODO подумать: мб тут было бы логичнее взять N = число переменных из которых строилась bn
        weights_preprocessed = list(map(lambda x: x * N / sum(ll), ll))
       # print(weights_preprocessed, left_vertices)
        train_dataset = self.convolve(
            train_dataset, weights_preprocessed, left_vertices
        )
        test_dataset = self.convolve(
            test_dataset, weights_preprocessed, left_vertices
        )
        return train_dataset, test_dataset

    def func(self,x):
            if x[1] == "y" and len(x[0]) > 1:
                number = int(x[0][5:])
            elif x[0] == "y" and len(x[1]) > 1:
                number = int(x[1][5:])
            else:
                number = np.nan
            return number

    def SelfSupervisedLoss(self, deg_pred,cluster_pred,mask):
        num_parts = min(3, int(len(self.data.x)/300))
       # cluster = ClusterData(self.data, num_parts)

#TODO пока нет расстояния до центра кластера - много дополнительный вычислений
        return F.mse_loss(deg_pred,self.deg)#+MSELoss(cluster_pred,cluster)

#this function takes one graph, extract ego-networks for each node and construct new dataset--list of this ego-networks
    def convert_dataset(self, data, train_mask, val_mask):
        n_min = data.num_nodes
        loader = NeighborSampler(
            data.edge_index,
            batch_size=1,
            sizes=[-1] * 4, #вот тут можно сделать автоподбор такого числа слоев, чтоб число соседей было около 10?
        )

        train_dataset = []
        test_dataset = []

        for i,(batch_size, n_id, adjs) in enumerate(loader):

            edge_index = torch.concat([adjs[0].edge_index, adjs[1].edge_index, adjs[2].edge_index, adjs[3].edge_index],
                                      dim=1)
            x = data.x[n_id]
            y = data.y[n_id]
            if i in train_mask:
                train_dataset.append(Data(x=x,y=y[0],edge_index=edge_index))
            else:
                test_dataset.append(Data(x=x, y=y[0], edge_index=edge_index))

            if n_min > len(n_id):
                n_min = len(n_id)
        return train_dataset, test_dataset, n_min


    def data_eigen_exctractor(self, dataset):

        columns_list = list(map(lambda x: "eigen" + str(x), range(self.n_min)))
        data_bamt = pd.DataFrame(columns=columns_list + ["y"])
        for gr in dataset:
            A = to_dense_adj(gr.edge_index)
            eig = torch.eig(A.reshape(A.shape[1], A.shape[2]))[0].T[0].T
            ordered, indices = torch.sort(eig[: gr.num_nodes], descending=True)
            to_append = pd.Series(
                ordered[: self.n_min].tolist() + [gr.y.tolist()], index=data_bamt.columns
            )
            data_bamt = data_bamt.append(to_append, ignore_index=True)

        return data_bamt

    def bn_build(self, data_bamt):
        # поиск весов для bamt

        for col in data_bamt.columns[: len(data_bamt.columns)]:
            data_bamt[col] = data_bamt[col].astype(float)
        data_bamt["y"] = data_bamt["y"].astype(int)

        bn = Nets.HybridBN(has_logit=True)
        discretizer = preprocessing.KBinsDiscretizer(
            n_bins=10, encode="ordinal", strategy="quantile"
        )
        p = Preprocessor([("discretizer", discretizer)])
        discretized_data, est = p.apply(data_bamt)

        bn.add_nodes(p.info)

        params = dict()
        params["remove_init_edges"] = self.remove_init_edges
        if self.init_edges:

            params["init_edges"] = list(
                map(lambda x: ("eigen" + str(x), "y"), list(range(self.n_min)))
            ) + list(map(lambda x: ("y", "eigen" + str(x)), list(range(self.n_min))))

          #  print("init_edges", params["init_edges"])
        if self.white_list:

            params["white_list"] = list(
                map(lambda x: ("eigen" + str(x), "y"), list(range(self.n_min)))
            ) + list(map(lambda x: ("y", "eigen" + str(x)), list(range(self.n_min))))

           # print("white_list", params["white_list"])
        #   params = {'init_edges': [('eigen0', 'y'), ('eigen1', 'y'), ('eigen2', 'y'), ('eigen3', 'y'), ('eigen4', 'y'),
        #                           ('eigen5', 'y'), ('eigen6', 'y'), ('eigen7', 'y'), ('eigen8', 'y'), ('eigen9', 'y')],
        #           'remove_init_edges': False,
        #          'white_list': [('eigen0', 'y'), ('eigen1', 'y'), ('eigen2', 'y'), ('eigen3', 'y'), ('eigen4', 'y'),
        #                        ('eigen5', 'y'), ('eigen6', 'y'), ('eigen7', 'y'), ('eigen8', 'y'), ('eigen9', 'y')]}

        bn.add_edges(
            discretized_data,
            scoring_function=(self.score_func, self.score),
            params=params,
        )

        bn.calculate_weights(discretized_data)
        bn.plot("BN1.html")
        return bn

    def convolve(self, dataset, weights, left_vertices):
        new_Data = []
        for graph in dataset:
            A = to_dense_adj(graph.edge_index)
            eigs = torch.eig(A.reshape(A.shape[1], A.shape[2]), True)
            eigenvectors = eigs[1]
            eig = eigs[0].T[0].T
            ordered, indices = torch.sort(eig[: graph.num_nodes], descending=True)
            lef = indices[left_vertices]
            zeroed = torch.tensor(list(set(range(len(eig))) - set(lef.tolist())))
            # print(zeroed)
            if len(zeroed) > 0:
                eig[zeroed] = 0

            for e, d in enumerate(lef):
                eig[d] = eig[d] * weights[e]

            eigenvalues = torch.diag(eig)
            convolved = torch.matmul(
                torch.matmul(eigenvectors, eigenvalues), eigenvectors.T
            )
            new_A = convolved.type(torch.DoubleTensor)
            graph.edge_index, graph.edge_weight = dense_to_sparse(convolved)
            graph.edge_weight = graph.edge_weight  #.type(torch.FloatTensor)
            graph.edge_index = graph.edge_index.type(torch.LongTensor)
            # print((graph.edge_weight).dtype)
            new_Data.append(graph)
        return new_Data
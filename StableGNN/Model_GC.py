import bamt.Networks as Nets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bamt.Preprocessors import Preprocessor
from pgmpy.estimators import K2Score
from sklearn import preprocessing
from torch.nn import Linear
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj

from StableGNN.Graph import Graph


class ModelName(torch.nn.Module):
    def __init__(
        self,
        dataset,
        device,
        conv="GAT",
        hidden_layer=64,
        dropout=0,
        num_layers=2,
        SSL=False,
        heads=1,
    ):
        super(ModelName, self).__init__()
        self.conv = conv
        self.num_layers = num_layers
        self.data = dataset
        self.num_features = dataset[0].x.shape[1]
        # print(dataset.num_features)
        self.convs = torch.nn.ModuleList()
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.device = device
        self.SSL = SSL
        self.heads = heads

        labels = []
        for dat in self.data:
            if dat.y.tolist()[0] not in labels:
                labels.append(dat.y.tolist()[0])
        num_classes = len(labels)

        if self.conv == "GAT":
            if self.num_layers == 1:
                self.convs.append(GATConv(self.num_features, hidden_layer))
            else:
                self.convs.append(GATConv(self.num_features, self.hidden_layer))
                for i in range(1, self.num_layers):
                    self.convs.append(
                        GATConv(self.heads * self.hidden_layer, self.hidden_layer)
                    )

        self.linear = Linear(self.heads * self.hidden_layer, int(self.hidden_layer / 2))

        self.linear_classifier = Linear(int(self.hidden_layer / 2), num_classes)
        self.linear_degree_predictor = Linear(int(self.hidden_layer / 2), 1)

    def forward(
        self, x, edge_index, edge_weight, batch, graph_level=True
    ):  # TODO: add batchnorm after self.linear_layer
        # 1. Obtain node embeddings
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = x.relu()
        # 2. Readout layer
        if graph_level:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        x = F.relu(x)

        deg_pred = 0
        # cluster_pred=0 #TODO если убираем - то удалить
        if self.SSL:
            deg_pred = F.relu(self.linear_degree_predictor(x))
            # cluster_pred =  F.relu(self.linear_cluster_distance_predictor(x))

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
        deg_pred = 0
        cluster_pred = 0  # TODO если убираем - то удалить
        if self.SSL:
            deg_pred = F.relu(self.linear_degree_predictor(x))
            # cluster_pred =  F.relu(self.linear_cluster_distance_predictor(x))
        x = self.linear_classifier(x)
        return x.log_softmax(dim=-1), deg_pred

    def loss_sup(self, pred, label):
        return F.nll_loss(pred, label)

    def Extrapolate(
        self,
        train_indices,
        val_indices,
        init_edges,
        remove_init_edges,
        white_list,
        score_func,
    ):
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

        train_dataset, test_dataset, val_dataset, n_min = self.convert_dataset(
            self.data, train_indices, val_indices
        )
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
        val_dataset = self.convolve(val_dataset, weights_preprocessed, left_vertices)
        test_dataset = self.convolve(test_dataset, weights_preprocessed, left_vertices)

        return train_dataset, test_dataset, val_dataset

    def convert_dataset(self, data, train_indices, val_indices):
        train_dataset = []
        test_dataset = []
        val_dataset = []
        n_min = np.inf
        for i, dat in enumerate(data):
            if len(dat.x) < n_min:
                n_min = len(dat.x)

            if i in train_indices:
                train_dataset.append(dat)
            elif i in val_indices:
                val_dataset.append(dat)
            else:
                test_dataset.append(dat)

        return train_dataset, test_dataset, val_dataset, n_min

    def func(self, x):
        if x[1] == "y" and len(x[0]) > 1:
            number = int(x[0][5:])
        elif x[0] == "y" and len(x[1]) > 1:
            number = int(x[1][5:])
        else:
            number = np.nan
        return number

    def data_eigen_exctractor(self, dataset):

        columns_list = list(map(lambda x: "eigen" + str(x), range(self.n_min)))
        data_bamt = pd.DataFrame(columns=columns_list + ["y"])
        for gr in dataset:
            A = to_dense_adj(gr.edge_index)
            eig = torch.eig(A.reshape(A.shape[1], A.shape[2]))[0].T[0].T
            ordered, indices = torch.sort(eig[: gr.num_nodes], descending=True)
            to_append = pd.Series(
                ordered[: self.n_min].tolist() + gr.y.tolist(), index=data_bamt.columns
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
            graph.edge_weight = graph.edge_weight  # .type(torch.FloatTensor)
            graph.edge_index = graph.edge_index.type(torch.LongTensor)
            # print((graph.edge_weight).dtype)
            new_Data.append(graph)
        return new_Data

    def SelfSupervisedLoss(self, deg_pred, batch):
        deg_pred = deg_pred.reshape(deg_pred.shape[0])
        batch_ptr = (batch.ptr.type(torch.LongTensor)).cpu()
        indices = batch_ptr[: len(batch_ptr) - 1]
        ratio = np.mean(
            batch_ptr.numpy()[1:] - batch_ptr.numpy()[: len(batch_ptr) - 1]
        )  # после экстраполяции мы получили очень плотный граф, хотим степень снизить в N раз для более хороших предсказаний
        true = degree(batch.edge_index[0], batch.x.shape[0])[indices] / ratio
        return F.mse_loss(deg_pred, true)

from typing import List, Tuple

import bamt.Networks as Nets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bamt.Preprocessors import Preprocessor
from pgmpy.estimators import K2Score
from sklearn import preprocessing
from torch import device
from torch.nn import Linear
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.typing import Adj, Tensor
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj

from stable_gnn.graph import Graph


class ModelName(torch.nn.Module):
    """
    Model for Graph Classification task

    :param dataset: ([Graph]): List of input graphs
    :param device: (device): Device -- 'cuda' or 'cpu'
    :param conv: (str): Name of the convolution used for Neural Network
    :param hidden_layer: (int): The size of hidden layer (default: 64)
    :param dropout: (int): Dropout (default: 0)
    :param num_layers: (int): Number of layers in the model (default: 2)
    :param ssl_flag: (bool): If True, self supervised loss would be alsooptimized during the training, in addition to semi-supervised
    :param heads: (int): Number of heads in GAT layer
    """

    def __init__(
        self,
        dataset: List[Graph],
        device: device,
        conv: str = "GAT",
        hidden_layer: int = 64,
        dropout: int = 0,
        num_layers: int = 2,
        ssl_flag: bool = False,
        heads: int = 1,
    ) -> None:

        super(ModelName, self).__init__()
        self.conv = conv
        self.num_layers = num_layers
        self.data = dataset
        self.num_features = dataset[0].x.shape[1]
        self.convs = torch.nn.ModuleList()
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.device = device
        self.ssl_flag = ssl_flag
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
                    self.convs.append(GATConv(self.heads * self.hidden_layer, self.hidden_layer))

        self.linear = Linear(self.heads * self.hidden_layer, int(self.hidden_layer / 2))

        self.linear_classifier = Linear(int(self.hidden_layer / 2), num_classes)
        self.linear_degree_predictor = Linear(int(self.hidden_layer / 2), 1)

    def forward(self, x: Tensor, edge_index: Adj, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Count the representation of node on the next layer of the model

        :param x: (Tensor) Input features
        :param edge_index: (Adj) Edge index of a batch
        :param batch: Batch of data
        :return: (Tensor, Tensor): Predicted probabilities of labels and predicted degrees of graphs
        """
        # 1. Obtain node embeddings
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = x.relu()
        # 2. Readout layer

        x = global_mean_pool(x, batch)
        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        x = F.relu(x)

        deg_pred = 0
        if self.ssl_flag:
            deg_pred = F.relu(self.linear_degree_predictor(x))

        x = self.linear_classifier(x)
        return x.log_softmax(dim=1), deg_pred

    def loss_sup(self, pred: Tensor, label: Tensor) -> Tensor:
        """Negative log likelihood loss

        :param pred: (Tensor): Predicted labels
        :param label: (Tensor): Genuine labels
        :return: (Tensor): Loss
        """
        return F.nll_loss(pred, label)

    def Extrapolate(
        self,
        train_indices: List[int],
        val_indices: List[int],
        init_edges: bool = False,
        remove_init_edges: bool = False,
        white_list: bool = False,
        score_func: str = "MI",
    ) -> Tuple[List[Graph], List[Graph], List[Graph]]:
        """
        Adjust dataset so that to increase extrapolation ability

        :param train_indices: ([int]): List of train indices
        :param val_indices: ([int]): List of validation indices
        :param init_edges: (bool): If True, there would be a list of init edges as start for Learning structure of Bayesian Net(default:'False')
        :param remove_init_edges: If True, it is possible that edges from init_list would be removed during the structure learning of Bayesian Net(default:'False')
        :param white_list: If True, edges inBayesian Net would be only from this white list (default:'False')
        :param score_func: (str): Name of score function to optimize, either 'MI' or 'K2' (default:'MI')
        :return: ([Graph], [Graph], [Graph]): Lists of train, test and validation graphs
        """
        self.init_edges = init_edges
        self.remove_init_edges = remove_init_edges
        self.white_list = white_list
        self.score_func = score_func
        if score_func == "K2":
            self.score = K2Score
        elif score_func == "MI":
            self.score = None
        else:
            raise Exception("there is no ", self.score_func, "score function. Choose one of: MI, K2")

        train_dataset, test_dataset, val_dataset, n_min = self.convert_dataset(self.data, train_indices, val_indices)
        self.n_min = n_min

        data_bamt = self._data_eigen_exctractor(train_dataset)
        bn = self._bayesian_network_build(data_bamt)
        lis = list(
            map(lambda x: self._func(x), bn.edges)
        )  # мы берем только те веришны, которые исходят из y или входят в у
        left_vertices = sorted(list(filter(lambda x: not np.isnan(x), lis)))
        left_edges = list(filter(lambda x: x[0] == "y" or x[1] == "y", bn.edges))
        left_edges = sorted(left_edges, key=lambda x: int(x[0][5:] if x[1] == "y" else int(x[1][5:])))
        ll = list(map(lambda x: bn.weights[tuple(x)], left_edges))
        len_of_remaining_nodes = len(
            ll
        )  # TODO подумать: мб тут было бы логичнее взять N = число переменных из которых строилась bn
        weights_preprocessed = list(map(lambda x: x * len_of_remaining_nodes / sum(ll), ll))
        train_dataset = self._convolve(train_dataset, weights_preprocessed, left_vertices)
        val_dataset = self._convolve(val_dataset, weights_preprocessed, left_vertices)
        test_dataset = self._convolve(test_dataset, weights_preprocessed, left_vertices)

        return train_dataset, test_dataset, val_dataset

    @staticmethod
    def convert_dataset(
        data: List[Graph], train_indices: List[int], val_indices: List[int]
    ) -> Tuple[List[Graph], List[Graph], List[Graph], int]:
        """
        Convert input dataset to train,test, val according to provided indices

        :param data: ([Graph]): List of graphs as input dataset
        :param train_indices: ([int]): List of indices for train dataset
        :param val_indices: ([int]): List of indices for validation dataset
        :return: ([Graph],[Graph],[Graph], int): Lists of train and validation graphs and the minimum size among all graphs
        """
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

    def _func(self, x: List[str]) -> int:
        if x[1] == "y" and len(x[0]) > 1:
            number = int(x[0][5:])
        elif x[0] == "y" and len(x[1]) > 1:
            number = int(x[1][5:])
        else:
            number = np.nan
        return number

    def _data_eigen_exctractor(self, dataset: List[Graph]) -> pd.DataFrame:

        columns_list = list(map(lambda x: "eigen" + str(x), range(self.n_min)))
        data_bamt = pd.DataFrame(columns=columns_list + ["y"])
        for gr in dataset:
            adj_matrix = to_dense_adj(gr.edge_index)
            eig = torch.eig(adj_matrix.reshape(adj_matrix.shape[1], adj_matrix.shape[2]))[0].T[0].T
            ordered, indices = torch.sort(eig[: gr.num_nodes], descending=True)
            to_append = pd.Series(ordered[: self.n_min].tolist() + gr.y.tolist(), index=data_bamt.columns)
            data_bamt = data_bamt.append(to_append, ignore_index=True)

        return data_bamt

    def _bayesian_network_build(self, data_bamt: pd.DataFrame) -> Nets.HybridBN:
        # поиск весов для bamt
        for col in data_bamt.columns[: len(data_bamt.columns)]:

            data_bamt[col] = data_bamt[col].astype(float)
        data_bamt["y"] = data_bamt["y"].astype(int)

        bn = Nets.HybridBN(has_logit=True)
        discretizer = preprocessing.KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
        p = Preprocessor([("discretizer", discretizer)])
        discretized_data, est = p.apply(data_bamt)

        bn.add_nodes(p.info)

        params = dict()
        params["remove_init_edges"] = self.remove_init_edges
        if self.init_edges:

            params["init_edges"] = list(map(lambda x: ("eigen" + str(x), "y"), list(range(self.n_min)))) + list(
                map(lambda x: ("y", "eigen" + str(x)), list(range(self.n_min)))
            )

        if self.white_list:

            params["white_list"] = list(map(lambda x: ("eigen" + str(x), "y"), list(range(self.n_min)))) + list(
                map(lambda x: ("y", "eigen" + str(x)), list(range(self.n_min)))
            )

        bn.add_edges(
            discretized_data,
            scoring_function=(self.score_func, self.score),
            params=params,
        )

        bn.calculate_weights(discretized_data)
        bn.plot("BN1.html")
        return bn

    @staticmethod
    def _convolve(dataset: List[Graph], weights: List[float], left_vertices: List[int]) -> List[Graph]:
        new_data = []
        for graph in dataset:
            A = to_dense_adj(graph.edge_index)
            eigs = torch.eig(A.reshape(A.shape[1], A.shape[2]), True)
            eigenvectors = eigs[1]
            eig = eigs[0].T[0].T
            ordered, indices = torch.sort(eig[: graph.num_nodes], descending=True)
            lef = indices[left_vertices]
            zeroed = torch.tensor(list(set(range(len(eig))) - set(lef.tolist())))
            if len(zeroed) > 0:
                eig[zeroed] = 0

            for e, d in enumerate(lef):
                eig[d] = eig[d] * weights[e]

            eigenvalues = torch.diag(eig)
            convolved = torch.matmul(torch.matmul(eigenvectors, eigenvalues), eigenvectors.T)

            graph.edge_index, graph.edge_weight = dense_to_sparse(convolved)
            graph.edge_weight = graph.edge_weight
            graph.edge_index = graph.edge_index.type(torch.LongTensor)
            new_data.append(graph)
        return new_data

    @staticmethod
    def self_supervised_loss(deg_pred: Tensor, batch: Batch) -> Tensor:
        """
        Self Supervised Loss for Graph Classsification task, MSE between predicted average degree of each graph and genuine ones

        :param deg_pred: (Tensor): Tensor of predicted degrees of graphs in dataset
        :param batch: (): Batch of train data
        :return: (Tensor): Loss
        """
        deg_pred = deg_pred.reshape(deg_pred.shape[0])
        batch_ptr = (batch.ptr.type(torch.LongTensor)).cpu()
        indices = batch_ptr[: len(batch_ptr) - 1]
        ratio = np.mean(
            batch_ptr.numpy()[1:] - batch_ptr.numpy()[: len(batch_ptr) - 1]
        )  # после экстраполяции мы получили очень плотный граф, хотим степень снизить в N раз для более хороших предсказаний
        true = degree(batch.edge_index[0], batch.x.shape[0])[indices] / ratio
        return F.mse_loss(deg_pred, true)

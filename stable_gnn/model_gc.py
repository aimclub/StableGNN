from typing import List, Optional, Tuple

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


class ModelGraphClassification(torch.nn.Module):
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
        super(ModelGraphClassification, self).__init__()
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

    def forward(self, x: Tensor, edge_index: Adj, batch: Tensor) -> Tuple[Tensor, Tensor]:
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

        deg_pred = Tensor(0)
        if self.ssl_flag:
            deg_pred = F.relu(self.linear_degree_predictor(x))

        x = self.linear_classifier(x)
        return x.log_softmax(dim=1), deg_pred

    @staticmethod
    def loss_sup(pred: Tensor, label: Tensor) -> Tensor:
        """Negative log likelihood loss

        :param pred: (Tensor): Predicted labels
        :param label: (Tensor): Genuine labels
        :return: (Tensor): Loss
        """
        return F.nll_loss(pred, label)



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

        return train_dataset, test_dataset, val_dataset, int(n_min)



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

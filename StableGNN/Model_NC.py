from torch_geometric.nn.conv import MessagePassing

import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv, ChebConv
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from datetime import datetime
import collections
import numpy as np
from torch.nn import Linear
from torch_geometric.utils import degree, to_dense_adj,dense_to_sparse

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
        num_classes = len(collections.Counter(self.data.y.tolist()))
        self.SSL = SSL
        if self.SSL:  # TODO: разобраться с SSL сейчас он считает true deg на изначальной матрице смежности а не модифицированной
            self.deg = degree(self.data.edge_index[0], self.data.num_nodes)

        if self.conv == "GAT":
            if self.num_layers == 1:
                self.convs.append(GATConv(self.num_features, hidden_layer))
            else:
                self.convs.append(GATConv(self.num_features, self.hidden_layer))
                for i in range(1, self.num_layers):
                    self.convs.append(GATConv(self.hidden_layer, self.hidden_layer))

        self.linear = Linear(self.hidden_layer, num_classes)

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):

            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        deg_pred=0
        if self.SSL:
            deg_pred = F.relu(self.linear_degree_predictor(x))

        return x.log_softmax(dim=1), deg_pred

    def inference(self, data, dp=0):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=dp, training=self.training)
        x = self.linear(x)
        return x.log_softmax(dim=-1)

    def loss_sup(self, pred, label):
        return F.nll_loss(pred, label)

    def message(self):
        pass

    def VirtualVertex(self):
        pass

    def SelfSupervisedLoss(self, deg_pred):
            deg_pred = deg_pred.reshape(deg_pred.shape[0])
            true = degree()
            return F.mse_loss(deg_pred, true)
import collections

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv, SGConv
from torch_geometric.utils import degree

from StableGNN.GeomGCN import GeomGCN


class ModelName(torch.nn.Module):
    def __init__(
        self,
        dataset,
        data_name,
        device,
        hidden_layer=64,
        dropout=0,
        num_layers=2,
        SSL=False,
        heads=1,
        **kwargs
    ):
        super(ModelName, self).__init__()
        self.num_layers = num_layers
        self.data = dataset
        self.data_name = data_name
        print("BEGINNING", len(self.data.x))
        self.num_features = dataset.x.shape[1]
        # print(dataset.num_features)
        self.convs = torch.nn.ModuleList()

        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.device = device
        self.heads = heads
        self.num_classes = len(collections.Counter(self.data.y.tolist()))
        self.SSL = SSL
        if (
            self.SSL
        ):  # TODO: разобраться с SSL сейчас он считает true deg на изначальной матрице смежности а не модифицированной
            self.deg = degree(self.data.edge_index[0], self.data.num_nodes)

        if self.num_layers == 1:
            self.convs.append(
                GeomGCN(
                    self.num_features,
                    self.hidden_layer,
                    LAST_LAYER=True,
                    data_name=data_name,
                )
            )
        else:
            self.convs.append(
                GeomGCN(self.num_features * 8, self.hidden_layer, data_name=data_name)
            )
            for i in range(1, self.num_layers - 1):
                self.convs.append(
                    GeomGCN(
                        self.hidden_layer * 8, self.hidden_layer, data_name=data_name
                    )
                )
            self.convs.append(
                GeomGCN(
                    self.hidden_layer,
                    self.hidden_layer,
                    LAST_LAYER=True,
                    data_name=data_name,
                )
            )

        self.linear = Linear(
            self.hidden_layer, int(self.hidden_layer / 2)
        )  # просто чтоб снизить размерность
        self.linear_classifier = Linear(int(self.hidden_layer / 2), self.num_classes)
        self.linear_degree_predictor = Linear(int(self.hidden_layer / 2), 1)

    def forward(self, x, adjs, weights=None, batch=None):
        for i, (edge_index, _, size) in enumerate(adjs):
            print(size)
            x_target = x[: size[1]]  # Target nodes are always placed first.

            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        deg_pred = 0
        if self.SSL:
            deg_pred = F.relu(self.linear_degree_predictor(x))
        return x.log_softmax(dim=1), deg_pred

    def inference(self, data, dp=0):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
        x = self.linear(x)
        deg_pred = 0
        if self.SSL:
            deg_pred = F.relu(self.linear_degree_predictor(x))
        return x.log_softmax(dim=-1), deg_pred

    def loss_sup(self, pred, label):
        return F.nll_loss(pred, label)

    def SelfSupervisedLoss(self, deg_pred, dat=None):
        # s num_parts = min(3, int(len(self.data.x) / 300))
        # cluster = ClusterData(self.data, num_parts)
        # TODO пока нет расстояния до центра кластера - много дополнительный вычислений
        true_deg = degree(self.data.edge_index[0], self.data.num_nodes).to(self.device)[
            dat
        ]

        # print(deg_pred.squeeze(-1),true_deg)
        return F.mse_loss(
            deg_pred.squeeze(-1), true_deg
        )  # +MSELoss(cluster_pred,cluster)

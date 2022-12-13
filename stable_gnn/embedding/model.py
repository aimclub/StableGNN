from typing import Dict

import torch
import torch.nn.functional as F
from torch import device
from torch_geometric.loader.neighbor_sampler import EdgeIndex
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv, SGConv
from torch_geometric.typing import Tensor

from stable_gnn.graph import Graph


class Net(torch.nn.Module):
    """
    The model for learning latent embeddings in unsupervised manner for Geom-GCN layer

    :param dataset: (Graph):
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_function: (dict): Dict of parameters of unsupervised loss function
    :param conv: (str): Either 'GCN', 'GAT' or 'SAGE' convolution (default:'GCN')
    :param hidden_layer: (int): The size of hidden layer (default:64)
    :param out_layer: (int): The size of output layer (default:128)
    :param dropout: (float): Dropout (default:0.0)
    :param num_layers: (int): Number of layers in the model (default:2)
    :param heads: (int): Number of heads in GAT conv (default:1)
    """

    def __init__(
        self,
        dataset: Graph,
        device: device,
        loss_function: Dict,
        conv: str = "GCN",
        hidden_layer: int = 64,
        out_layer: int = 128,
        dropout: float = 0,
        num_layers: int = 2,
        heads: int = 1,
    ) -> None:

        super(Net, self).__init__()
        self.conv = conv
        self.num_layers = num_layers
        self.data = dataset
        self.num_features = dataset.x.shape[1]
        self.loss_function = loss_function
        self.convs = torch.nn.ModuleList()
        self.hidden_layer = hidden_layer
        self.out_layer = out_layer
        self.dropout = dropout
        self.device = device
        out_channels = self.out_layer
        self.heads = heads

        if loss_function["loss var"] == "Random Walks":
            self.loss = self._loss_random_walks
        elif loss_function["loss var"] == "Context Matrix":
            self.loss = self._loss_context_matrix
        elif loss_function["loss var"] == "Factorization":
            self.loss = self._loss_factorization
        elif loss_function["loss var"] == "Laplacian EigenMaps":
            self.loss = self._loss_laplacian_eigen_maps
        elif loss_function["loss var"] == "Force2Vec":
            self.loss = self._loss_t_distribution

        if self.conv == "GCN":
            if self.num_layers == 1:
                self.convs.append(GCNConv(self.num_features, out_channels))
            else:
                self.convs.append(GCNConv(self.num_features, self.hidden_layer))
                for i in range(1, self.num_layers - 1):
                    self.convs.append(GCNConv(self.hidden_layer, self.hidden_layer))
                self.convs.append(GCNConv(self.hidden_layer, out_channels))
        elif self.conv == "SAGE":

            if self.num_layers == 1:
                self.convs.append(SAGEConv(self.num_features, out_channels))
            else:
                self.convs.append(SAGEConv(self.num_features, self.hidden_layer))
                for i in range(1, self.num_layers - 1):
                    self.convs.append(SAGEConv(self.hidden_layer, self.hidden_layer))
                self.convs.append(SAGEConv(self.hidden_layer, out_channels))
        elif self.conv == "GAT":
            if self.num_layers == 1:
                self.convs.append(GATConv(self.num_features, out_channels, heads=self.heads))
            else:
                self.convs.append(GATConv(self.num_features, self.hidden_layer, heads=self.heads))
                for i in range(1, self.num_layers - 1):
                    self.convs.append(
                        GATConv(
                            self.heads * self.hidden_layer,
                            self.hidden_layer,
                            heads=self.heads,
                        )
                    )
                self.convs.append(GATConv(self.heads * self.hidden_layer, out_channels, heads=self.heads))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters"""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, adjs: EdgeIndex) -> Tensor:
        """
        Find representations of the node

        :param x: (Tensor): Features of nodes
        :param adjs: (EdgeIndex): Edge indices of computational graph for each layer
        :return: (Tensor): Representations of nodes
        """
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def inference(self, data: Graph, dp: float = 0) -> Tensor:
        """
        Count representations of the node

        :param data: (Graph): Input data
        :param dp: (float): Dropout (default:0.0)
        :return: (Tensor): Representations of nodes
        """
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):

            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=dp, training=self.training)
        return x

    def _loss_random_walks(self, out, pos_neg_samples):
        (pos_rw, neg_rw) = pos_neg_samples
        pos_rw, neg_rw = pos_rw.type(torch.LongTensor).to(self.device), neg_rw.type(torch.LongTensor).to(self.device)
        # Positive loss.
        pos_loss = 0
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        h_start = out[start].view(pos_rw.size(0), 1, self.out_layer)
        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1, self.out_layer)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)

        pos_loss = -(torch.nn.LogSigmoid()(dot)).mean()

        # Negative loss
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        h_start = out[start].view(neg_rw.size(0), 1, self.out_layer)
        h_rest = out[rest.view(-1)].view(neg_rw.size(0), -1, self.out_layer)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -(torch.nn.LogSigmoid()((-1) * dot)).mean()

        return pos_loss + neg_loss

    def _loss_context_matrix(self, out, pos_neg_samples):
        (pos_rw, neg_rw) = pos_neg_samples
        pos_rw = pos_rw.to(self.device)
        neg_rw = neg_rw.to(self.device)

        start, rest = (
            neg_rw[:, 0].type(torch.LongTensor),
            neg_rw[:, 1:].type(torch.LongTensor).contiguous(),
        )
        indices = start != rest.view(-1)
        start = start[indices]
        h_start = out[start].view(start.shape[0], 1, self.out_layer)
        rest = rest.view(-1)
        rest = rest[indices]
        h_rest = out[rest].view(rest.shape[0], -1, self.out_layer)

        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -(torch.nn.LogSigmoid()((-1) * dot)).mean()
        # Positive loss.
        start, rest = (
            pos_rw[:, 0].type(torch.LongTensor),
            pos_rw[:, 1].type(torch.LongTensor).contiguous(),
        )
        weight = pos_rw[:, 2]
        h_start = out[start].view(pos_rw.size(0), 1, self.out_layer)

        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1, self.out_layer)
        dot = ((h_start * h_rest).sum(dim=-1)).view(-1)
        if self.loss_function["Name"] == "LINE":
            pos_loss = -2 * (weight * torch.nn.LogSigmoid()((-1) * dot)).mean()

        elif self.loss_function["Name"].split("_")[0] == "VERSE" or self.loss_function["Name"] == "APP":
            pos_loss = -(weight * torch.nn.LogSigmoid()((-1) * dot)).mean()

        return pos_loss + neg_loss

    def _loss_factorization(self, out, context_matrix):
        context_matrix = context_matrix.to(self.device)
        lmbda = self.loss_function["lmbda"]
        loss = 0.5 * sum(
            sum((context_matrix - torch.matmul(out, out.t())) * (context_matrix - torch.matmul(out, out.t())))
        ) + 0.5 * lmbda * sum(sum(out * out))
        return loss

    def _loss_laplacian_eigen_maps(self, out, adj_matrix):
        dd = torch.device("cuda", 0)
        # dd=torch.device('cpu')
        laplacian = (torch.diag(sum(adj_matrix)) - adj_matrix).type(torch.FloatTensor).to(dd)
        out_tr = out.t().to(dd)
        loss = torch.trace(torch.matmul(torch.matmul(out_tr, laplacian), out))
        yDy = torch.matmul(
            torch.matmul(out_tr, torch.diag(sum(adj_matrix.t())).type(torch.FloatTensor).to(dd)),
            out,
        ) - torch.diag(torch.ones(out.shape[1])).type(torch.FloatTensor).to(dd)
        loss_2 = torch.sqrt(sum(sum(yDy * yDy)))
        return loss + loss_2

    def _loss_t_distribution(self, out, pos_neg_samples):
        eps = 10e-6
        (pos_rw, neg_rw) = pos_neg_samples
        pos_rw = pos_rw.to(self.device)
        neg_rw = neg_rw.to(self.device)

        start, rest = (
            neg_rw[:, 0].type(torch.LongTensor),
            neg_rw[:, 1:].type(torch.LongTensor).contiguous(),
        )
        indices = start != rest.view(-1)
        start = start[indices]
        h_start = out[start].view(start.shape[0], 1, self.out_layer)
        rest = rest.view(-1)
        rest = rest[indices]
        h_rest = out[rest].view(rest.shape[0], -1, self.out_layer)
        t_squared = ((h_start - h_rest) * (h_start - h_rest)).mean(dim=-1).view(-1)
        neg_loss = (-torch.log((t_squared / (1 + t_squared)) + eps)).mean()

        # Positive loss.
        start, rest = (
            pos_rw[:, 0].type(torch.LongTensor),
            pos_rw[:, 1].type(torch.LongTensor).contiguous(),
        )

        h_start = out[start].view(pos_rw.size(0), 1, self.out_layer)
        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1, self.out_layer)
        t_squared = ((h_start - h_rest) * (h_start - h_rest)).sum(dim=-1).view(-1)
        pos_loss = -(torch.log(1 / (1 + t_squared))).mean()
        return pos_loss + neg_loss

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import device
from torch_geometric.loader.neighbor_sampler import EdgeIndex
from torch_geometric.typing import Tensor

from stable_gnn.graph import Graph


class BaseNet(torch.nn.Module, ABC):
    """The model for learning latent embeddings in unsupervised manner for Geom-GCN layer

    :param device: (device): Either 'cuda' or 'cpu'
    :param hidden_layer: (int): The size of hidden layer (default:64)
    :param out_layer: (int): The size of output layer (default:128)
    :param dropout: (float): Dropout (default:0.0)
    :param num_layers: (int): Number of layers in the model (default:2)
    :param heads: (int): Number of heads in GAT conv (default:1)
    """

    def __init__(
        self,
        device: device,
        num_featurs: int,
        hidden_layer: int = 64,
        out_layer: int = 128,
        num_layers: int = 2,
        heads: int = 1,
        dropout: float = 0,
    ) -> None:

        super(BaseNet, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_featurs
        self.out_layer = out_layer
        self.hidden_layer = hidden_layer
        self.heads = heads
        self.dropout = dropout
        self.device = device
        self.convs = torch.nn.ModuleList()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters"""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, adjs: EdgeIndex) -> Tensor:
        """Find representations of the node

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
        """Count representations of the node

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

    @abstractmethod
    def loss(self, out: Tensor, pos_neg_samples: Tensor) -> Tensor:
        """Calculate loss

        :param out: Tensor
        :param pos_neg_samples: Tensor
        :returns: (Tensor) Loss
        """
        raise NotImplementedError("Define loss in %s." % (self.__class__.__name__))

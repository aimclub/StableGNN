from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import device
import torch_geometric
from torch_geometric.loader.neighbor_sampler import EdgeIndex
from torch_geometric.typing import Tensor

from stable_gnn.graph import Graph


class BaseNet(torch.nn.Module, ABC):
    """The model for learning latent embeddings in an unsupervised manner for Geom-GCN layer

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

        # New additions for node and edge types
        self.node_type_embeddings = None  # Placeholder for node type embeddings
        self.edge_type_weights = None  # Placeholder for edge type weights

    def reset_parameters(self) -> None:
        """Reset parameters"""
        for conv in self.convs:
            conv.reset_parameters()

    def set_node_types(self, node_types: Tensor, embedding_dim: int) -> None:
        """Set embeddings for node types

        :param node_types: (Tensor): Tensor of node types
        :param embedding_dim: (int): Dimensionality of embeddings for node types
        """
        num_node_types = int(node_types.max().item() + 1)
        self.node_type_embeddings = torch.nn.Embedding(num_node_types, embedding_dim).to(self.device)

    def set_edge_types(self, edge_types: Tensor) -> None:
        """Set weights for edge types

        :param edge_types: (Tensor): Tensor of edge types
        """
        num_edge_types = int(edge_types.max().item() + 1)
        self.edge_type_weights = torch.nn.Parameter(torch.ones(num_edge_types, device=self.device))

    def forward(self, x: Tensor, adjs: EdgeIndex, node_types: Tensor = None, edge_types: Tensor = None) -> Tensor:
        for i, (edge_index, _, size) in enumerate(adjs):
            # Проверка на пустые данные
            if x.size(0) == 0 or edge_index.size(1) == 0:
                print(f"Skipping layer {i} due to empty input.")
                continue

            x_target = x[: size[1]]  # Целевые узлы

            # Применение эмбеддингов типов узлов
            if node_types is not None and self.node_type_embeddings is not None:
                node_type_emb = self.node_type_embeddings(node_types[: size[1]])
                x_target = x_target + node_type_emb

            # Применение слоя
            edge_weight = self.edge_type_weights[edge_types] if edge_types is not None and self.edge_type_weights is not None else None
            if isinstance(self.convs[i], torch_geometric.nn.GATConv):
                x = self.convs[i]((x, x_target), edge_index)  # GATConv игнорирует edge_weight
            elif isinstance(self.convs[i], torch_geometric.nn.SAGEConv):
                x = self.convs[i]((x, x_target), edge_index)  # SAGEConv также игнорирует edge_weight
            else:
                x = self.convs[i]((x, x_target), edge_index, edge_weight=edge_weight)

            # Активация и dropout
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


    def inference(self, data: Graph, dp: float = 0, node_types: Tensor = None, edge_types: Tensor = None) -> Tensor:
        """Count representations of the node

        :param data: (Graph): Input data
        :param dp: (float): Dropout (default:0.0)
        :param node_types: (Tensor): Types of nodes (optional)
        :param edge_types: (Tensor): Types of edges (optional)
        :return: (Tensor): Representations of nodes
        """
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            # Apply node type embeddings if available
            if node_types is not None and self.node_type_embeddings is not None:
                node_type_emb = self.node_type_embeddings(node_types)
                x = x + node_type_emb

            # Apply edge type weights if available
            edge_weight = None
            if edge_types is not None and self.edge_type_weights is not None:
                edge_weight = self.edge_type_weights[edge_types]

            # Check layer type for edge_weight support
            if isinstance(conv, (torch_geometric.nn.SAGEConv, torch_geometric.nn.GATConv)):
                x = conv(x, edge_index)  # Skip edge_weight for unsupported layers
            else:
                x = conv(x, edge_index, edge_weight=edge_weight)

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

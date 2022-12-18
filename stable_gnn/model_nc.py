import collections
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import device
from torch.nn import Linear
from torch_geometric.typing import Tensor
from torch_geometric.utils import degree

from stable_gnn.embedding import EmbeddingFactory
from stable_gnn.geom_gcn import GeomGCN
from stable_gnn.graph import Graph


class ModelNodeClassification(torch.nn.Module):
    """
    Model for Node Classification task with Layer, considering grph characteristics

    :param dataset: (Graph): Input Graph
    :param device: (device): Device 'cuda' or 'cpu'
    :param hidden_layer: (int): The size of hidden layer (default: 64)
    :param dropout: (float): Dropout (defualt: 0.0)
    :param num_layers: (int): Number of layers in the model (default:2)
    :param ssl_flag: (bool): If True, self supervised loss will be optimized additionally to semi-supervised (default: False)
    :param loss_name: (str): Name of loss function for embedding learning in GeomGCN layer
    """

    def __init__(
        self,
        dataset: Graph,
        device: device,
        hidden_layer: int = 64,
        dropout: float = 0.0,
        num_layers: int = 2,
        ssl_flag: bool = False,
        loss_name: str = "APP",
        emb_conv_name: str = "SAGE",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.data = dataset
        self.num_features = dataset[0].x.shape[1]
        self.convs = torch.nn.ModuleList()
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.device = device
        self.num_classes = len(collections.Counter(self.data[0].y.tolist()))
        self.ssl_flag = ssl_flag
        # TODO: разобраться с SSL сейчас он считает true deg на изначальной матрице смежности а не модифицированной
        if self.ssl_flag:
            self.deg = degree(self.data[0].edge_index[0], self.data[0].num_nodes)

        embeddings = EmbeddingFactory().build_embeddings(
            loss_name=loss_name, conv=emb_conv_name, data=dataset, device=device
        )

        if self.num_layers == 1:
            self.convs.append(
                GeomGCN(
                    self.num_features,
                    self.hidden_layer,
                    last_layer=True,
                    data=self.data,
                    device=self.device,
                    embeddings=embeddings,
                )
            )
        else:
            self.convs.append(
                GeomGCN(
                    self.num_features * 8, self.hidden_layer, device=self.device, data=self.data, embeddings=embeddings
                )
            )
            for _ in range(1, self.num_layers - 1):
                self.convs.append(
                    GeomGCN(
                        self.hidden_layer * 8,
                        self.hidden_layer,
                        device=self.device,
                        data=self.data,
                        embeddings=embeddings,
                    )
                )
            self.convs.append(
                GeomGCN(
                    self.hidden_layer,
                    self.hidden_layer,
                    last_layer=True,
                    data=self.data,
                    device=self.device,
                    embeddings=embeddings,
                )
            )

        self.linear = Linear(self.hidden_layer, int(self.hidden_layer / 2))  # просто чтоб снизить размерность
        self.linear_classifier = Linear(int(self.hidden_layer / 2), self.num_classes)
        self.linear_degree_predictor = Linear(int(self.hidden_layer / 2), 1)

    def inference(self, data: Graph) -> Tuple[Tensor, Tensor]:
        """Count the representation of the node on the next layer of the model

        :param data: (Graph): Input Graph
        :return: (Tensor, Tensor):  Predicted probabilities of labels and predicted degrees of nodes
        """
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
        x = self.linear(x)
        deg_pred = 0
        if self.ssl_flag:
            deg_pred = F.relu(self.linear_degree_predictor(x))
        return x.log_softmax(dim=-1), deg_pred

    @staticmethod
    def loss_sup(pred: Tensor, label: Tensor) -> Tensor:
        """
        Count negative log likelihood loss function

        :param pred: (Tensor): Predicted labels
        :param label: (Tensor): Genuine labels
        :return: (Tensor): Loss
        """
        return F.nll_loss(pred, label)

    def self_supervised_loss(self, deg_pred: Tensor, dat: Tensor) -> Tensor:
        """
        Self supervised loss, predicting degrees of nodes

        :param deg_pred: (Tensor): Predicted degrees
        :param dat: (Tensor): Train mask
        :return: (Tensor): Loss
        """
        # TODO пока нет расстояния до центра кластера - много дополнительный вычислений
        true_deg = degree(self.data[0].edge_index[0], self.data[0].num_nodes).to(self.device)[dat]

        return F.mse_loss(deg_pred.squeeze(-1), true_deg)

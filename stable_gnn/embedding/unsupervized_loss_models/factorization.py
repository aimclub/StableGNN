import torch
from torch import Tensor, device

from stable_gnn.embedding.models.abstract_model import BaseNet


class FactorizationModel(BaseNet):
    """The FactorizationModel for learning latent embeddings in unsupervised manner for Geom-GCN layer

    :param device: (device): Either 'cuda' or 'cpu'
    :param hidden_layer: (int): The size of hidden layer (default:64)
    :param out_layer: (int): The size of output layer (default:128)
    :param dropout: (float): Dropout (default:0.0)
    :param num_layers: (int): Number of layers in the model (default:2)
    :param heads: (int): Number of heads in GAT conv (default:1)
    """

    def __init__(
        self,
        lmbda: int,
        device: device,
        num_featurs: int,
        hidden_layer: int = 64,
        out_layer: int = 128,
        num_layers: int = 2,
        heads: int = 1,
        dropout: float = 0,
    ):
        super().__init__(device, num_featurs, hidden_layer, out_layer, num_layers, heads, dropout)
        self._lmbda = lmbda

    def loss(self, out: Tensor, context_matrix: Tensor) -> Tensor:
        """Calculate loss

        :param out: Tensor
        :param pos_neg_samples: Tensor
        :returns: (Tensor) Loss
        """
        context_matrix = context_matrix.to(self.device)
        loss = 0.5 * sum(
            sum((context_matrix - torch.matmul(out, out.t())) * (context_matrix - torch.matmul(out, out.t())))
        ) + 0.5 * self._lmbda * sum(sum(out * out))
        return loss

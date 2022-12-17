import torch
from torch import Tensor, device

from stable_gnn.embedding.models.abstract_model import BaseNet


class Force2VecModel(BaseNet):
    """The Force2VecModel for learning latent embeddings in unsupervised manner for Geom-GCN layer

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
    ):
        super().__init__(device, num_featurs, hidden_layer, out_layer, num_layers, heads, dropout)

    def loss(self, out: Tensor, pos_neg_samples: Tensor) -> Tensor:
        """Calculate loss

        :param out: Tensor
        :param pos_neg_samples: Tensor
        :returns: (Tensor) Loss
        """
        eps = 10e-6
        pos_rw, neg_rw = pos_neg_samples
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

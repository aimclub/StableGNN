import torch
from torch import Tensor, device

from stable_gnn.embedding.models.abstract_model import BaseNet


class RandomWalkBasedModel(BaseNet):
    """The RandomWalkBasedModel for learning latent embeddings in unsupervised manner for Geom-GCN layer

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

    def _calc_random_walk(self, out: Tensor, samples: Tensor) -> Tensor:
        start, rest = samples[:, 0], samples[:, 1:].contiguous()
        h_start = out[start].view(samples.size(0), 1, self.out_layer)
        h_rest = out[rest.view(-1)].view(samples.size(0), -1, self.out_layer)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        return dot

    def loss(self, out: Tensor, pos_neg_samples: Tensor) -> Tensor:
        """Calculate loss

        :param out: Tensor
        :param pos_neg_samples: Tensor
        :returns: (Tensor) Loss
        """
        pos_rw, neg_rw = pos_neg_samples

        # Positive loss.
        pos_rw = pos_rw.type(torch.LongTensor).to(self.device)
        pos_dot = self._calc_random_walk(out=out, samples=pos_rw)
        pos_loss = -(torch.nn.LogSigmoid()(pos_dot)).mean()

        # Negative loss
        neg_rw = neg_rw.type(torch.LongTensor).to(self.device)
        neg_dot = self._calc_random_walk(out=out, samples=neg_rw)
        neg_loss = -(torch.nn.LogSigmoid()((-1) * neg_dot)).mean()
        return pos_loss + neg_loss

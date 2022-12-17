import torch
from torch import Tensor, device

from stable_gnn.embedding.models.abstract_model import BaseNet


class LalplacianEigenMapsModel(BaseNet):
    """The LalplacianEigenMapsModel for learning latent embeddings in unsupervised manner for Geom-GCN layer

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

    def loss(self, out: Tensor, adj_matrix: Tensor) -> Tensor:
        """Calculate loss

        :param out: Tensor
        :param pos_neg_samples: Tensor
        :returns: (Tensor) Loss
        """
        laplacian = (torch.diag(sum(adj_matrix)) - adj_matrix).type(torch.FloatTensor).to(self.device)
        out_tr = out.t().to(self.device)

        loss = torch.trace(torch.matmul(torch.matmul(out_tr, laplacian), out))
        y_dy = torch.matmul(
            torch.matmul(out_tr, torch.diag(sum(adj_matrix.t())).type(torch.FloatTensor).to(self.device)), out
        )

        y_dy = y_dy - torch.diag(torch.ones(out.shape[1])).type(torch.FloatTensor).to(self.device)

        loss_2 = torch.sqrt(sum(sum(y_dy * y_dy)))
        return loss + loss_2

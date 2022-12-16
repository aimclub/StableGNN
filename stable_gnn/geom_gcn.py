import os
from typing import Any, Union, List

import numpy as np
import torch
from sklearn.neighbors import BallTree
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor
from torch_geometric.utils import degree

from stable_gnn.embedding.model_train_embeddings import ModelTrainEmbeddings, OptunaTrainEmbeddings
from stable_gnn.embedding.sampling import SamplerAPP, SamplerContextMatrix, SamplerFactorization
from stable_gnn.graph import Graph


class GeomGCN(MessagePassing):
    r"""The graph convolutional operator from the `"GEOM-GCN: GEOMETRIC GRAPH CONVOLUTIONAL NETWORKS" <https://arxiv.org/pdf/2002.05287>`_ paper

    .. math::
        \textbf{e}_{(i,r)}^{v,l+1} = \sum_{u \in N_{i}(v)} \delta(\tau(z_v,z_u),r)(deg(v)deg(u))^{\frac{1}{2}} \textbf{h}_u^l, \forall i \in {g,s}, \forall r \in R

    .. math::
        \textbf{h}_v^{l+1}=\sigma(W_l \cdot \mathbin\Vert_{i\in \{g,s\}} \mathbin\Vert_{r \in R} \textbf{e}_{(i,r)}^{v,l+1})


    where :math:`\textbf{e}_{(i,r)}^{v,l+1}` is a virtual vertex, recieved by summing up representations :math:`\textbf{h}_u^l` of nodes on layer l in structural neighbourhoods :math:`i=s` and graph neighbourhood :math:`i=g` separately for each neighbors with relation :math:`r` from the set of relations :math:`R`.
    :math:`z_v` is an embedding of nodes in latent space, :math:`deg(v)` is a degree of node :math:`v`

    :param in_channels: (int): Size of each input sample.
    :param out_channels: (int): Size of each output sample.
    :param data: (Graph): Input dataset
    :param last_layer: (bool): When true, the virtual vertices are summed, otherwise -- concatenated.
    :param loss_name: (str): Name of the loss function fo unsupervised representation learning
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data: Graph,
        last_layer: bool = False,
        loss_name: str = "APP",
    ) -> None:
        super().__init__(aggr="add")

        self.lin = Linear(in_channels, out_channels, bias=False)
        self.data_name = data.name
        self.data = data[0]
        self.last_layer = last_layer
        self.loss_name = loss_name
        torch.manual_seed(0)
        self.reset_parameters()

        # TODO проверить можем ли мы пробрасывать девайс снаружи
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset_parameters(self) -> None:
        """Reset parameters"""
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Modify representations, convolutional layer

        :param x: (Tensor): Representations of nodes
        :param edge_index: (Tensor): Edges of input graph
        :return: Hodden representation of nodes on the next layer
        """
        out = self._virtual_vertex(edge_index=edge_index, x=x, loss_name=self.loss_name)
        out = self.lin(out)
        return out

    @staticmethod
    def _normalization_term(edge_index: Tensor, x: Any) -> float:
        row, col = edge_index
        deg = degree(col, 251, dtype=x[0].dtype)
        deg_sqrt = deg.pow(0.5)
        norm = deg_sqrt[row] * deg_sqrt[col]
        return norm

    def _virtual_vertex(self, edge_index: Tensor, x: Union[Tensor, OptPairTensor], loss_name: str) -> np.array:
        if isinstance(x, Tensor):
            x = (x, x)
        graph_size = (
            max(
                edge_index[0].max(),
                edge_index[1].max(),
            )
            + 1
        )
        deg = degree(edge_index[0], graph_size)
        emb = self._embedding(loss_name)
        (
            edge_index_s_ur,
            edge_index_s_ul,
            edge_index_s_lr,
            edge_index_s_ll,
            edge_index_g_ur,
            edge_index_g_ul,
            edge_index_g_lr,
            edge_index_g_ll,
        ) = self._edge_indices_divider(emb, deg, edge_index)

        e_g_ur = self.propagate(edge_index_g_ur, x=x, norm=self._normalization_term(edge_index_g_ur, x))
        e_g_ul = self.propagate(edge_index_g_ul, x=x, norm=self._normalization_term(edge_index_g_ul, x))
        e_g_lr = self.propagate(edge_index_g_lr, x=x, norm=self._normalization_term(edge_index_g_lr, x))
        e_g_ll = self.propagate(edge_index_g_ll, x=x, norm=self._normalization_term(edge_index_g_ll, x))

        ei = edge_index_s_ur.to(self.device)
        e_s_ur = self.propagate(ei, x=x, norm=self._normalization_term(ei, x))

        ei = edge_index_s_ul.to(self.device)
        e_s_ul = self.propagate(ei, x=x, norm=self._normalization_term(ei, x))

        ei = edge_index_s_lr.to(self.device)
        e_s_lr = self.propagate(ei, x=x, norm=self._normalization_term(ei, x))

        ei = edge_index_s_ll.to(self.device)
        e_s_ll = self.propagate(ei, x=x, norm=self._normalization_term(ei, x))

        # здесь конкат для всех кроме последнего слоя, для последнего должно быть mean
        if self.last_layer:

            x = (e_s_ur + e_s_ul + e_s_lr + e_s_ll + e_g_ur + e_g_ul + e_g_lr + e_g_ll) / 8

        else:

            x = torch.concat([e_s_ur, e_s_ul, e_s_lr, e_s_ll, e_g_ur, e_g_ul, e_g_lr, e_g_ll], axis=1)

        # не уверена можно ли использовать propogate на разные соседства, нет ли там какого-то пересечения, сохранения инфы в ходе дела?
        return x

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        """
        Count message from the neighbour

        :param x_j (Tensor): Representation of the node neighbour
        :param norm (Tensor): Normalization term
        :return: (Tensor): Message from the neighbor
        """
        return norm.view(-1, 1) * x_j

    def _embedding(self, loss_name: str) -> np.array:
        if loss_name == "APP":
            loss = {
                "Name": "APP",
                "C": "PPR",
                "num_negative_samples": [1, 6, 11, 16, 21],
                "loss var": "Context Matrix",
                "flag_tosave": True,
                "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "Sampler": SamplerAPP,
            }  # APP
        elif loss_name == "LINE":
            loss = {
                "Name": "LINE",
                "C": "Adj",
                "num_negative_samples": [1, 6, 11, 16, 21],
                "loss var": "Context Matrix",
                "flag_tosave": False,
                "Sampler": SamplerContextMatrix,
                "lmbda": [0.0, 1.0],
            }
        elif loss_name == "HOPE_AA":
            loss = {
                "Name": "HOPE_AA",
                "C": "AA",
                "loss var": "Factorization",
                "flag_tosave": True,
                "Sampler": SamplerFactorization,
                "lmbda": [0.0, 1.0],
            }
        elif loss_name == "VERSE_Adj":
            loss = {
                "Name": "VERSE_Adj",
                "C": "Adj",
                "num_negative_samples": [1, 6, 11, 16, 21],
                "loss var": "Context Matrix",
                "flag_tosave": False,
                "Sampler": SamplerContextMatrix,
                "lmbda": [0.0, 1.0],
            }
        else:
            raise NameError

        embeddings_name = (
            "../data_validation/"
            + self.data_name
            + "/processed/"
            + "embeddings_"
            + self.data_name
            + "_"
            + self.loss_name
            + ".npy"
        )
        if os.path.exists(embeddings_name):
            emb = np.load(embeddings_name)
            return emb
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            optuna_training = OptunaTrainEmbeddings(
                name=self.data_name, data=self.data, conv="SAGE", device=device, loss_function=loss
            )
            best_values = optuna_training.run(number_of_trials=10)

            loss_trgt = dict()
            for par in loss:
                loss_trgt[par] = loss[par]

            if "alpha" in loss_trgt:
                loss_trgt["alpha"] = best_values["alpha"]
            if "num_negative_samples" in loss_trgt:
                loss_trgt["num_negative_samples"] = best_values["num_negative_samples"]
            if "lmbda" in loss_trgt:
                loss_trgt["lmbda"] = best_values["num_negative_samples"]

            model_training = ModelTrainEmbeddings(
                name=self.data_name, data=self.data, conv="SAGE", device=device, loss_function=loss_trgt
            )
            out = model_training.run(best_values)
            torch.cuda.empty_cache()
            np.save(
                embeddings_name,
                out.detach().cpu().numpy(),
            )
            return out.detach().cpu().numpy()

    def _relation(self, emb1: np.array, emb2: np.array) -> int:
        if (emb1[0] > emb2[0]) and (emb1[1] <= emb2[1]):
            return 1
        elif (emb1[0] <= emb2[0]) and (emb1[1] > emb2[1]):
            return 2
        elif (emb1[0] <= emb2[0]) and (emb1[1] <= emb2[1]):
            return 0
        else:
            return 3

    def _edge_indices_divider(self, emb: np.array, deg: Tensor, edge_index: Tensor) -> List[Tensor]:
        edge_index_s = self._structural_neighbourhood(emb, deg)

        edge_index_s_ur = torch.stack(
            [
                edge_index_s[0][torch.where(edge_index_s[2] == 0)],
                edge_index_s[1][torch.where(edge_index_s[2] == 0)],
            ]
        )
        edge_index_s_ul = torch.stack(
            [
                edge_index_s[0][torch.where(edge_index_s[2] == 1)],
                edge_index_s[1][torch.where(edge_index_s[2] == 1)],
            ]
        )
        edge_index_s_lr = torch.stack(
            [
                edge_index_s[0][torch.where(edge_index_s[2] == 2)],
                edge_index_s[1][torch.where(edge_index_s[2] == 2)],
            ]
        )
        edge_index_s_ll = torch.stack(
            [
                edge_index_s[0][torch.where(edge_index_s[2] == 3)],
                edge_index_s[1][torch.where(edge_index_s[2] == 3)],
            ]
        )

        edge_index = self._edge_index_conversion(edge_index.cpu(), emb)
        ei = edge_index.to("cuda" if torch.cuda.is_available() else "cpu")
        edge_index_g_ur = torch.stack([ei[0][torch.where(ei[2] == 0)], ei[1][torch.where(ei[2] == 0)]])
        edge_index_g_ul = torch.stack([ei[0][torch.where(ei[2] == 1)], ei[1][torch.where(ei[2] == 1)]])
        edge_index_g_lr = torch.stack([ei[0][torch.where(ei[2] == 2)], ei[1][torch.where(ei[2] == 2)]])
        edge_index_g_ll = torch.stack([ei[0][torch.where(ei[2] == 3)], ei[1][torch.where(ei[2] == 3)]])

        return [
            edge_index_s_ur,
            edge_index_s_ul,
            edge_index_s_lr,
            edge_index_s_ll,
            edge_index_g_ur,
            edge_index_g_ul,
            edge_index_g_lr,
            edge_index_g_ll,
        ]

    def _structural_neighbourhood(
        self, emb: np.array, deg: Tensor
    ) -> Tensor:  # для каждой связи добавляем третий инедекс вес который означает именно _Relation
        if os.path.exists(
            "../data_validation/"
            + self.data_name
            + "/processed"
            + "/structural_neighbourhood_"
            + self.data_name
            + ".npy"
        ):
            new_edge_index = np.load(
                "../data_validation/"
                + self.data_name
                + "/processed"
                + "/structural_neighbourhood_"
                + self.data_name
                + ".npy"
            )
        else:
            deg = deg.tolist()
            new_edge_index = []
            tree = BallTree(emb, leaf_size=2)
            for i in range(len(emb)):

                dist, ind = tree.query(emb[i : i + 1], k=int(deg[i]))
                for nei in ind[0]:  # indices of 3 closest neighbors
                    _Relation = self._relation(emb[i], emb[nei])
                    new_edge_index.append([i, nei, _Relation])

            np.save(
                "../data_validation/"
                + self.data_name
                + "/processed"
                + "/structural_neighbourhood_"
                + self.data_name
                + ".npy",
                np.array(new_edge_index),
            )

        new_edge_index = torch.tensor(
            new_edge_index
        ).T  # надо проверить мб из листа делать тензор будет проще чем из нп эррей?

        return new_edge_index

    def _edge_index_conversion(self, edge_index: Tensor, emb: np.array) -> Tensor:

        list_of__Relations = []
        iterating = edge_index.T.tolist()
        for e in iterating:
            list_of__Relations.append(self._relation(emb[e[0]], emb[e[1]]))

        return torch.concat(
            [
                edge_index,
                (torch.tensor(list_of__Relations)).reshape(1, len(list_of__Relations)),
            ]
        )

import collections
import os

import numpy as np
import torch
from sklearn.neighbors import BallTree
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor
from torch_geometric.utils import add_self_loops, degree

from stable_gnn.embedding.model_train_embeddings import _ModelTrainEmbeddings, _OptunaTrainEmbeddings
from stable_gnn.embedding.sampling import _SamplerAPP


class GeomGCN(MessagePassing):
    r"""The graph convolutional operator from the `"GEOM-GCN: GEOMETRIC GRAPH CONVOLUTIONAL NETWORKS" <https://arxiv.org/pdf/2002.05287>`_ paper

    .. math::
        \textbf{e}_{(i,r)}^{v,l+1} = \sum_{u \in N_{i}(v)} \delta(\tau(z_v,z_u),r)(deg(v)deg(u))^{\frac{1}{2}} \textbf{h}_u^l, \forall i \in {g,s}, \forall r \in R

    ..math::
        \textbf{h}_v^{l+1}=\sigma(W_l \cdot \mathbin\Vert_{i\in \{g,s\}} \mathbin\Vert_{r \in R} \textbf{e}_{(i,r)}^{v,l+1})

    where :math: '\textbf{e}_{(i,r)}^{v,l+1}' is a virtual vertex, recieved by summing up representations :math: '\textbf{h}_u^l' of nodes on layer l in structural neighbourhoods :math:'i=s' and graph neighbourhood :math: 'i=g' separately for each neighbors with relation :math:'r' from the set of relations :math:'R'.
    :math:'z_v' is an embedding of nodes in latent space, :math:'deg(v)' is a degree of node :math:'v'

    :param in_channels: (int): Size of each input sample.
    :param out_channels: (int): Size of each output sample.
    :param data_name: (str): Name of your dataset. this is needed for saving embedding.
    :param last_layer: (bool): When true, the virtual vertices are summed, otherwise -- concatenated.
    """

    def __init__(self, in_channels: int, out_channels: int, data_name: str, last_layer: bool = False) -> None:
        super().__init__(aggr="add")
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()
        self.data_name = data_name
        self.last_layer = last_layer
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        out = self._virtual_vertex(edge_index=edge_index, x=x)
        out = self.lin(out)
        return out

    def _normalization_term(self, edge_index, x):
        row, col = edge_index
        deg = degree(col, 251, dtype=x[0].dtype)
        deg_sqrt = deg.pow(0.5)
        norm = deg_sqrt[row] * deg_sqrt[col]
        return norm

    def _virtual_vertex(self, edge_index, x):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        graph_size = max(
            len(collections.Counter(edge_index[0].tolist())),
            len(collections.Counter(edge_index[1].tolist())),
        )
        deg = degree(edge_index[0], graph_size)
        emb = self._embedding()
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

        ei = edge_index_s_ur.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        e_s_ur = self.propagate(ei, x=x, norm=self._normalization_term(ei, x))
        ei = edge_index_s_ul.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        e_s_ul = self.propagate(ei, x=x, norm=self._normalization_term(ei, x))
        ei = edge_index_s_lr.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        e_s_lr = self.propagate(ei, x=x, norm=self._normalization_term(ei, x))
        ei = edge_index_s_ll.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        e_s_ll = self.propagate(ei, x=x, norm=self._normalization_term(ei, x))

        # здесь конкат для всех кроме последнего слоя, для последнего должно быть mean
        if self.last_layer:

            x = (e_s_ur + e_s_ul + e_s_lr + e_s_ll + e_g_ur + e_g_ul + e_g_lr + e_g_ll) / 8

        else:

            x = torch.concat([e_s_ur, e_s_ul, e_s_lr, e_s_ll, e_g_ur, e_g_ul, e_g_lr, e_g_ll], axis=1)

        # не уверена можно ли использовать propogate на разные соседства, нет ли там какого-то пересечения, сохранения инфы в ходе дела?
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def _embedding(self):
        if os.path.exists(
            "./data_validation/" + self.data_name + "/processed" + "/embeddings_" + self.data_name + ".npy"
        ):
            emb = np.load(
                "./data_validation/" + self.data_name + "/processed" + "/embeddings_" + self.data_name + ".npy"
            )
            return emb
        else:
            loss = {
                "Name": "APP",
                "C": "PPR",
                "num_negative_samples": [1, 6, 11, 16, 21],
                "loss var": "Context Matrix",
                "flag_tosave": True,
                "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "Sampler": _SamplerAPP,
            }  # APP
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MO = _OptunaTrainEmbeddings(name=self.data_name, conv="SAGE", device=device, loss_function=loss)
            best_values = MO.run(number_of_trials=500)

            loss_trgt = dict()
            for par in loss:
                loss_trgt[par] = loss[par]

            loss_trgt["alpha"] = best_values["alpha"]
            loss_trgt["num_negative_samples"] = best_values["num_negative_samples"]

            M = _ModelTrainEmbeddings(name=self.data_name, conv="SAGE", device=device, loss_function=loss_trgt)
            out = M.run(best_values)
            torch.cuda.empty_cache()
            np.save(
                "./data_validation/" + self.data_name + "/processed" + "/embeddings_" + self.data_name + ".npy",
                out.detach().cpu().numpy(),
            )
            return out

    def _Relation(self, emb1, emb2):
        if (emb1[0] > emb2[0]) and (emb1[1] <= emb2[1]):
            return 1
        elif (emb1[0] <= emb2[0]) and (emb1[1] > emb2[1]):
            return 2
        elif (emb1[0] <= emb2[0]) and (emb1[1] <= emb2[1]):
            return 0
        else:
            return 3

    def _edge_indices_divider(self, emb, deg, edge_index):
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
        self, emb, deg
    ):  # для каждой связи добавляем третий инедекс вес который означает именно _Relation
        if os.path.exists(
            "./data_validation/"
            + self.data_name
            + "/processed"
            + "/structural_neighbourhood_"
            + self.data_name
            + ".npy"
        ):
            new_edge_index = np.load(
                "./data_validation/"
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
                    _Relation = self._Relation(emb[i], emb[nei])
                    new_edge_index.append([i, nei, _Relation])

            np.save(
                "./data_validation/"
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

    def _edge_index_conversion(self, edge_index, emb):

        list_of__Relations = []
        iterating = edge_index.T.tolist()
        for e in iterating:
            list_of__Relations.append(self._Relation(emb[e[0]], emb[e[1]]))

        return torch.concat(
            [
                edge_index,
                (torch.tensor(list_of__Relations)).reshape(1, len(list_of__Relations)),
            ]
        )

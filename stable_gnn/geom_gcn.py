from typing import Any, List, Union

import torch
from numpy.typing import NDArray
from sklearn.neighbors import BallTree
from torch import Tensor, device
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptPairTensor
from torch_geometric.utils import degree

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
    :param embeddings: (NDArray): array of node unsupervised embeddings
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data: Graph,
        device: device,
        embeddings: NDArray,
        last_layer: bool = False,
    ) -> None:
        super().__init__(aggr="add")

        self.lin = Linear(in_channels, out_channels, bias=False)
        self.data_name = data.name
        self.data = data[0]
        self.last_layer = last_layer
        torch.manual_seed(0)
        self.reset_parameters()

        self.device = device

        self.emb = embeddings

    def reset_parameters(self) -> None:
        """Reset parameters"""
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> NDArray:
        """
        Modify representations, convolutional layer

        :param x: (Tensor): Representations of nodes
        :param edge_index: (Tensor): Edges of input graph
        :return: Hidden representation of nodes on the next layer
        """
        out = self._virtual_vertex(edge_index=edge_index, x=x)
        out = self.lin(out)
        return out

    @staticmethod
    def _normalization_term(edge_index: Tensor, x: Any) -> float:
        row, col = edge_index
        deg = degree(col, 251, dtype=x[0].dtype)
        deg_sqrt = deg.pow(0.5)
        norm = deg_sqrt[row] * deg_sqrt[col]
        return norm

    def _virtual_vertex(self, edge_index: Tensor, x: Union[Tensor, OptPairTensor]) -> NDArray:
        if isinstance(x, Tensor):
            x = (x, x)
        graph_size = max(edge_index[0].max(), edge_index[1].max()) + 1
        deg = degree(edge_index[0], graph_size)

        (
            edge_index_s_ur,
            edge_index_s_ul,
            edge_index_s_lr,
            edge_index_s_ll,
            edge_index_g_ur,
            edge_index_g_ul,
            edge_index_g_lr,
            edge_index_g_ll,
        ) = self._edge_indices_divider(self.emb, deg, edge_index)

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

    @staticmethod
    def _relation(emb1: NDArray, emb2: NDArray) -> int:
        if (emb1[0] > emb2[0]) and (emb1[1] <= emb2[1]):
            return 1
        elif (emb1[0] <= emb2[0]) and (emb1[1] > emb2[1]):
            return 2
        elif (emb1[0] <= emb2[0]) and (emb1[1] <= emb2[1]):
            return 0
        else:
            return 3

    def _edge_indices_divider(self, emb: NDArray, deg: Tensor, edge_index: Tensor) -> List[Tensor]:
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
        ei = edge_index.to(self.device)
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
        self, emb: NDArray, deg: Tensor
    ) -> Tensor:  # для каждой связи добавляем третий инедекс вес который означает именно _Relation

        deg_list = deg.tolist()
        new_edge_index = []
        tree = BallTree(emb, leaf_size=2)
        for i in range(len(emb)):

            dist, ind = tree.query(emb[i : i + 1], k=int(deg_list[i]))
            for nei in ind[0]:  # indices of 3 closest neighbors
                _Relation = self._relation(emb[i], emb[nei])
                new_edge_index.append([i, nei, _Relation])

        new_edge_index = torch.tensor(
            new_edge_index
        ).T  # надо проверить мб из листа делать тензор будет проще чем из нп эррей?

        return new_edge_index

    def _edge_index_conversion(self, edge_index: Tensor, emb: NDArray) -> Tensor:

        list_of_relations = []
        iterating = edge_index.T.tolist()
        for e in iterating:
            list_of_relations.append(self._relation(emb[e[0]], emb[e[1]]))

        return torch.concat(
            [
                edge_index,
                (torch.tensor(list_of_relations)).reshape(1, len(list_of_relations)),
            ]
        )

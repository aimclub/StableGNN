import collections
import math
from typing import Dict, Tuple

import torch
from torch import device
from torch_geometric.data import Batch
from torch_geometric.typing import Tensor
from torch_sparse import SparseTensor

from stable_gnn.graph import Graph

try:
    import torch_cluster  # noqa

    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

from torch_geometric.utils import subgraph

from stable_gnn.embedding.sampling.abstract_samplers import BaseSampler, BaseSamplerWithNegative

class NegativeSampler(BaseSamplerWithNegative):
    """
    Sampler for positive and negative edges using random walk based methods

    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    """

    def __init__(self, data: Graph, device: device, loss_info: Dict) -> None:
        super().__init__(data, device, loss_info)

    def _neg_sample(self, batch: Tensor) -> Tensor:
        a, _ = subgraph(batch.tolist(), self.data.edge_index)
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        neg_batch = self._sample_negative(batch, num_negative_samples=self.num_negative_samples)
        return neg_batch

    def negative_sample(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Sample positive and negative edges for batch nodes

        :param batch: (Batch): Nodes for positive and negative sampling from them
        :return: (Tensor, Tensor): positive and negative samples
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return self._neg_sample(batch)
    def _pos_sample(self, batch: Tensor) -> Tensor:
        pass

class SamplerRandomWalk(BaseSamplerWithNegative):
    """
    Sampler for positive and negative edges using random walk based methods

    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    """

    def __init__(self, data: Graph, device: device, loss_info: Dict) -> None:
        super().__init__(data, device, loss_info)
        self.p = self.loss["p"]
        self.q = self.loss["q"]
        self.walk_length = self.loss["walk_length"]
        self.walks_per_node = self.loss["walks_per_node"]
        self.context_size = (
            self.loss["context_size"] if self.walk_length >= self.loss["context_size"] else self.walk_length
        )

    def _neg_sample(self, batch: Tensor) -> Tensor:
        a, _ = subgraph(batch.tolist(), self.data.edge_index)
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        neg_batch = self._sample_negative(batch, num_negative_samples=self.num_negative_samples)
        return neg_batch

    def _pos_sample(self, batch: Tensor) -> Tensor:
        len_batch = len(batch)
        a, _ = subgraph(batch, self.data.edge_index)
        row, col = a
        row = row
        col = col
        adj = SparseTensor(row=row, col=col, sparse_sizes=(len_batch, len_batch))

        rowptr, col, _ = adj.csr()
        start = batch.repeat(self.walks_per_node).to(self.device)
        rw = random_walk(rowptr, col, start, self.walk_length, self.p, self.q)

        if not isinstance(rw, torch.Tensor):
            rw = rw[0]
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])  # теперь у нас внутри walks лежат 12 матриц размерам 10*1
        pos_samples = torch.cat(walks, dim=0)  # .to(self.device)
        return pos_samples


class SamplerContextMatrix(BaseSamplerWithNegative):
    """
    Sample positive and negative edges for context matrix based unsupervised loss function

    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    """

    def __init__(self, data: Graph, device: device, loss_info: Dict) -> None:
        super().__init__(data, device, loss_info)
        if self.loss["C"] == "PPR":
            self.alpha = round(self.loss["alpha"], 1)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch
        pos_batch = []
        if self.loss["C"] == "Adj" and (self.loss["Name"] == "LINE" or self.loss["Name"] == "Force2Vec"):
            adj = self._edge_index_to_adj_train(batch)
            pos_batch = self._convert_to_samples(batch, adj)

        elif self.loss["C"] == "Adj" and self.loss["Name"] == "VERSE_Adj":
            adj = self._edge_index_to_adj_train(batch).type(torch.FloatTensor)

            adj = (adj / sum(adj)).t()
            adj[torch.isinf(adj)] = 0
            adj[torch.isnan(adj)] = 0
            pos_batch = self._convert_to_samples(batch, adj)

        elif self.loss["C"] == "SR":
            adj, _ = subgraph(batch, self.data.edge_index)
            row, col = adj
            row = row.to(self.device)
            col = col.to(self.device)
            adj_sparse = SparseTensor(row=row, col=col, sparse_sizes=(len(batch), len(batch)))
            r = 200
            length = list(map(lambda x: x * int(r / 100), [22, 17, 14, 10, 8, 6, 5, 4, 3, 11]))  # O(t)
            masks = []
            for i, l in enumerate(length):  # max(l) = (1-sqrt(c)); O((1-sqrt(c))*t^2)
                mask1 = torch.zeros([l, 10])  # O( (1-sqrt(c)) *t)
                mask1.t()[: (i + 1)] = 1  # O( 3*(1-sqrt(c)) *t)=O((1-sqrt(c)) *t)
                masks.append(mask1)
            mask = torch.cat(masks)  # O((1-sqrt(c)) *t^2)
            mask_new = 1 - mask
            adj = self._find_sim_rank_for_batch_torch(batch, adj_sparse, self.device, mask, mask_new, r)
            pos_batch = self._convert_to_samples(batch, adj)

        elif self.loss["C"] == "PPR":
            alpha = self.alpha
            adj = self._edge_index_to_adj_train(batch).type(torch.FloatTensor)
            inv_d = torch.diag(1 / sum(adj.t()))
            inv_d[torch.isinf(inv_d)] = 0
            adj_matrix = (1 - alpha) * torch.inverse(
                torch.diag(torch.ones(len(adj))) - alpha * torch.matmul(inv_d, adj)
            )
            pos_batch = self._convert_to_samples(batch, adj_matrix)
        return pos_batch

    @staticmethod
    def _convert_to_samples(batch: Tensor, adj: Tensor) -> Tensor:
        pos_batch = []
        batch_l = batch.tolist()
        for x in batch_l:
            for j in batch_l:
                if adj[x][j] != torch.tensor(0):
                    pos_batch.append([int(x), int(j), adj[x][j]])

        return torch.tensor(pos_batch)

    def _find_sim_rank_for_batch_torch(
        self, batch: Tensor, adj: SparseTensor, device: device, mask: Tensor, mask_new: Tensor, r: int
    ) -> Tensor:
        t = 10
        # approx with SARW
        batch = batch.to(device)
        adj = adj.to(device)
        sim_rank = torch.zeros(len(batch), len(batch)).to(device)  # O(n^2)
        for u in batch:  # O(n^3*(tr))
            print("{}/{}".format(u, len(batch)))
            for nei in batch:
                pi_u = adj.random_walk(u.repeat(r).flatten(), walk_length=t)  # O(tnr)
                pi_v = adj.random_walk(nei.repeat(r).flatten(), walk_length=t)  # O(tnr)
                pi_u = pi_u[:, 1:]
                pi_v = pi_v[:, 1:]
                mask = mask.to(self.device)
                mask_new = mask_new.to(self.device)
                pi_u = pi_u * mask - mask_new  # O(r*t)
                pi_v = pi_v * mask - mask_new  # O(r*t)

                a1 = pi_u == pi_v  # O(r*t)
                a2 = pi_u != -1  # O(r*t)
                a3 = pi_v != -1  # O(r*t)
                a_to_compare = a1 * a2 * a3  # O(r*t)
                sr = len(torch.unique(a_to_compare.nonzero(as_tuple=True)[0]))  # O()
                sim_rank[u][nei] = sr / r  #

        return sim_rank


class SamplerFactorization(BaseSampler):
    """
    Sample positive and negative edges for context matrix based unsupervised loss function

    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    """

    def _pos_sample(self, batch: Tensor) -> Tensor:
        pass

    def __init__(self, data: Graph, device: device, loss_info: Dict) -> None:
        super().__init__(data, device, loss_info)

    def sample(self, batch: Tensor) -> Tensor:
        """
        Sample of positive and negative edges for Graph Factorization-based unsupervosed loss functions

        :param batch: (Batch): Nodes for which sampling should be conducted
        :return: (Tensor): Positive and negative edges
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)

        adj_matrix = self._edge_index_to_adj_train(batch)
        if self.loss["loss var"] == "Factorization":
            if self.loss["C"] == "Adj":
                context_matrix = adj_matrix

            elif self.loss["C"] == "CN":
                adj_matrix = adj_matrix.type(torch.FloatTensor).to(self.device)
                context_matrix = torch.matmul(adj_matrix, adj_matrix)

            elif self.loss["C"] == "AA":
                deg_matrix = torch.diag(1 / (sum(adj_matrix) + sum(adj_matrix.t())))
                adj_matrix = adj_matrix.type(torch.FloatTensor)
                deg_matrix[torch.isinf(deg_matrix)] = 0
                deg_matrix[torch.isnan(deg_matrix)] = 0
                context_matrix = torch.matmul(torch.matmul(adj_matrix, deg_matrix), adj_matrix)

            elif self.loss["C"] == "Katz":
                adj_matrix = adj_matrix.type(torch.FloatTensor)
                adj_matrix[torch.isinf(adj_matrix)] = 0
                adj_matrix[torch.isnan(adj_matrix)] = 0
                betta = self.loss["betta"]
                ones_matrix = torch.diag(torch.ones(len(adj_matrix)))
                inv = torch.cholesky_inverse(ones_matrix - betta * adj_matrix)
                context_matrix = betta * torch.matmul(inv, adj_matrix)

            elif self.loss["C"] == "RPR":
                alpha = self.loss["alpha"]
                adj_matrix = adj_matrix.type(torch.FloatTensor)
                inv_d = torch.diag(1 / sum(adj_matrix.t()))
                inv_d[torch.isinf(inv_d)] = 0
                context_matrix = (1 - alpha) * torch.inverse(
                    torch.diag(torch.ones(len(adj_matrix))) - alpha * torch.matmul(inv_d, adj_matrix)
                )
            else:
                raise ValueError(f"Not supported C: {self.loss['C']}")
            return context_matrix
        else:
            return adj_matrix


class SamplerAPP(BaseSamplerWithNegative):
    """
    Sample positive and negative edges for APP unsupervised loss function

    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    """

    def __init__(self, data: Graph, device: device, loss_info: Dict) -> None:
        super().__init__(data, device, loss_info)
        self.alpha = self.loss["alpha"]
        self.r = 200
        self.num_negative_samples *= 10

        new_edge_index_list = []
        li = data.edge_index.t().tolist()
        for edge in li:
            if edge not in new_edge_index_list:
                new_edge_index_list.append(edge)
            if [edge[1], edge[0]] not in new_edge_index_list:
                new_edge_index_list.append([edge[1], edge[0]])
        new_edge_index = torch.tensor(new_edge_index_list).t()
        self.data.edge_index = new_edge_index

    def sample(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Sample positive and negative edges for batch of nodes

        :param batch: (Batch): Batch for which sampling should be conducted
        :return: (Tensor,Tensor): positive and negative edges
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return self._pos_sample(batch), self._neg_sample(batch)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.to(self.device)
        len_batch = len(batch)
        mask = torch.tensor([False] * len(self.data.x))
        mask[batch.tolist()] = True
        a, _ = subgraph(batch, self.data.edge_index.to(self.device))
        row, col = a
        row = row.to(self.device)
        col = col.to(self.device)

        sparse_matrix = SparseTensor(row=row, col=col, sparse_sizes=(len_batch, len_batch))
        pos_dict = self._find_ppr_approx(batch, sparse_matrix, self.r, self.alpha, row)
        pos_batch = []
        for pos_pair in pos_dict:
            pos_row = list(pos_pair)
            pos_row.append(pos_dict[pos_pair])
            pos_batch.append(pos_row)
        return torch.tensor(pos_batch)

    def _find_ppr_approx(
        self, batch: Batch, sparse_matrix: SparseTensor, r: float, alpha: float, row: int
    ) -> Dict[Tuple[int, int], int]:
        n = math.ceil(math.log(1 / (r * alpha), (1 - alpha)))
        length = list(map(lambda x: int((1 - alpha) ** x * alpha * r), list(range(n - 1))))
        r = sum(length)
        dict_data: Dict[Tuple[int, int], int] = dict()
        for u in batch:
            if len(torch.where(row == u)[0]) > 0:
                pi_u = sparse_matrix.random_walk(u.to(self.device).repeat(r).flatten(), walk_length=n)
                split = torch.split(pi_u, length)

                for i, seg in enumerate(split):
                    pos_samples = collections.Counter(seg[:, (i + 1)].tolist())
                    for pos_sample in pos_samples:
                        if (int(seg[0][0]), int(pos_sample)) in dict_data:
                            dict_data[(int(seg[0][0]), int(pos_sample))] += pos_samples[pos_sample]
                        elif (int(pos_sample), int(seg[0][0])) in dict_data:
                            dict_data[(int(pos_sample), int(seg[0][0]))] += pos_samples[pos_sample]
                        else:
                            dict_data[(int(seg[0][0]), int(pos_sample))] = 1
        return dict_data

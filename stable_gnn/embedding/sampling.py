import collections
import math
import os
import pickle

import numpy as np
import torch
from torch_sparse import SparseTensor

from stable_gnn.embedding.negative_sampling import NegativeSampler
from stable_gnn.graph import Graph
from torch import device
from typing import Dict
from torch_geometric.data import Batch
from torch_geometric.typing import Tensor
from typing import Tuple
from abc import ABC, abstractmethod

try:
    import torch_cluster  # noqa

    RW = torch.ops.torch_cluster.random_walk
except ImportError:
    RW = None

from torch_geometric.utils import subgraph


class Sampler(ABC):
    """
    Base class for sampling of positive and negative edges for unsupervised loss function

    :param dataset_name: (str): Name of the inout dataset
    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    """

    def _edge_index_to_adj_train(self, batch):
        x_new = torch.sort(batch).values

        x_new = x_new.tolist()

        adj_matrix = torch.zeros((len(x_new), len(x_new)), dtype=torch.long)
        edge_index_0 = self.data.edge_index[0].tolist()
        edge_index_1 = self.data.edge_index[1].tolist()
        for j, i in enumerate(edge_index_0):
            if i in x_new:
                if edge_index_1[j] in x_new:
                    adj_matrix[i][edge_index_1[j]] = 1

        return adj_matrix

    def _edge_index_to_adj_train_old(self, mask):
        x_new = torch.tensor(np.where(mask == 1)[0], dtype=torch.int32)
        adj_matrix = torch.zeros((len(x_new), len(x_new)), dtype=torch.long)

        edge_index_0 = self.data.edge_index[0].to("cpu")
        edge_index_1 = self.data.edge_index[1].to("cpu")

        for j, i in enumerate(edge_index_0):
            if i in x_new:
                if edge_index_1[j] in x_new:
                    adj_matrix[i][edge_index_1[j]] = 1
        return adj_matrix

    @abstractmethod
    def sample(self, batch: Batch):
        """
        Sample positive edges. Must be implemented

        :param batch: (Batch): Nodes for sampling positive edges for them
        """
        raise NotImplementedError


class SamplerWithNegSamples(Sampler):
    """
    Sampler for positive and negative edges using random walk based methods

    :param dataset_name: (str): Name of the inout dataset
    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    """

    def sample(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Sample positive and negative edges for batch nodes

        :param batch: (Batch): Nodes for positive and negative sampling from them
        :return: (Tensor, Tensor): positive and negative samples
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return (self._pos_sample(batch), self._neg_sample(batch))

    @abstractmethod
    def _pos_sample(self, batch):
        pass

    def _neg_sample(self, batch):
        a, _ = subgraph(batch.tolist(), self.data.edge_index)
        neg_batch = self.negative_sampler.negative_sampling(batch, num_negative_samples=self.num_negative_samples)
        return neg_batch


class SamplerRandomWalk(SamplerWithNegSamples):
    """
    Sampler for positive and negative edges using random walk based methods

    :param dataset_name: (str): Name of the inout dataset
    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    """

    def __init__(self, dataset_name: str, data: Graph, device: device, loss_info: Dict) -> None:
        self.device = device
        self.dataset_name = dataset_name
        self.data = data.to(self.device)
        self.negative_sampler = NegativeSampler(self.data, self.device)
        self.loss = loss_info
        self.num_negative_samples = self.loss["num_negative_samples"]

        self.p = self.loss["p"]
        self.q = self.loss["q"]
        self.walk_length = self.loss["walk_length"]
        self.walks_per_node = self.loss["walks_per_node"]
        self.context_size = (
            self.loss["context_size"] if self.walk_length >= self.loss["context_size"] else self.walk_length
        )
        super().__init__()

    def _neg_sample(self, batch):
        a, _ = subgraph(batch.tolist(), self.data.edge_index)
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        neg_batch = self.negative_sampler.negative_sampling(batch, num_negative_samples=self.num_negative_samples)
        return neg_batch

    def _pos_sample(self, batch):
        name_of_samples = (
            self.dataset_name
            + "_"
            + str(self.walk_length)
            + "_"
            + str(self.walks_per_node)
            + "_"
            + str(self.context_size)
            + "_"
            + str(self.p)
            + "_"
            + str(self.q)
            + ".pickle"
        )
        if os.path.exists(name_of_samples):
            with open(name_of_samples, "rb") as f:
                pos_samples = pickle.load(f)
        else:
            len_batch = len(batch)
            a, _ = subgraph(batch, self.data.edge_index)
            row, col = a
            row = row
            col = col
            adj = SparseTensor(row=row, col=col, sparse_sizes=(len_batch, len_batch))

            rowptr, col, _ = adj.csr()
            start = batch.repeat(self.walks_per_node).to(self.device)
            rw = RW(rowptr, col, start, self.walk_length, self.p, self.q)

            if not isinstance(rw, torch.Tensor):
                rw = rw[0]
            walks = []
            num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
            for j in range(num_walks_per_rw):
                walks.append(
                    rw[:, j : j + self.context_size]
                )  # теперь у нас внутри walks лежат 12 матриц размерам 10*1
            pos_samples = torch.cat(walks, dim=0)  # .to(self.device)
        return pos_samples


class SamplerContextMatrix(SamplerWithNegSamples):
    """
    Sample positive and negative edges for context matrix based unsupervised loss function

    :param dataset_name: (str): Name of the inout dataset
    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    :param help_dir: (str): Name of directory for helping data
    """

    def __init__(self, dataset_name: str, data: Graph, device: device, loss_info: Dict, help_dir: str) -> None:
        self.device = device
        self.dataset_name = dataset_name
        self.data = data.to(self.device)
        self.negative_sampler = NegativeSampler(self.data, self.device)
        self.loss = loss_info
        self.num_negative_samples = self.loss["num_negative_samples"]
        self.help_dir = help_dir
        if self.loss["C"] == "PPR":
            self.alpha = round(self.loss["alpha"], 1)
        super().__init__()

    def _pos_sample(self, batch):
        batch = batch
        pos_batch = []
        if self.loss["C"] == "Adj" and (self.loss["Name"] == "LINE" or self.loss["Name"] == "Force2Vec"):
            name = f"{self.help_dir}/pos_samples_LINE_" + self.dataset_name + ".pickle"
            if os.path.exists(name):
                with open(name, "rb") as f:
                    pos_batch = pickle.load(f)
            else:
                A = self._edge_index_to_adj_train(batch)
                pos_batch = self._convert_to_samples(batch, A)
                with open(name, "wb") as f:
                    pickle.dump(pos_batch, f)
        elif self.loss["C"] == "Adj" and self.loss["Name"] == "VERSE_Adj":
            name = f"{self.help_dir}/pos_samples_VERSEAdj_" + self.dataset_name + ".pickle"
            if os.path.exists(name):
                with open(name, "rb") as f:
                    pos_batch = pickle.load(f)
            else:

                adj = self._edge_index_to_adj_train(batch).type(torch.FloatTensor)

                A = (adj / sum(adj)).t()
                A[torch.isinf(A)] = 0
                A[torch.isnan(A)] = 0
                pos_batch = self._convert_to_samples(batch, A)
                with open(name, "wb") as f:
                    pickle.dump(pos_batch, f)

        elif self.loss["C"] == "SR":
            sim_rank_name = "SimRank" + self.dataset_name + ".pickle"
            if os.path.exists(sim_rank_name):
                with open(sim_rank_name, "rb") as f:
                    A = pickle.load(f)
            else:
                adj, _ = subgraph(batch, self.data.edge_index)
                row, col = adj
                row = row.to(self.device)
                col = col.to(self.device)
                ASparse = SparseTensor(row=row, col=col, sparse_sizes=(len(batch), len(batch)))
                r = 200
                length = list(map(lambda x: x * int(r / 100), [22, 17, 14, 10, 8, 6, 5, 4, 3, 11]))  # O(t)
                mask = []
                for i, l in enumerate(length):  # max(l) = (1-sqrt(c)); O((1-sqrt(c))*t^2)
                    mask1 = torch.zeros([l, 10])  # O( (1-sqrt(c)) *t)
                    mask1.t()[: (i + 1)] = 1  # O( 3*(1-sqrt(c)) *t)=O((1-sqrt(c)) *t)
                    mask.append(mask1)
                mask = torch.cat(mask)  # O((1-sqrt(c)) *t^2)
                mask_new = 1 - mask
                A = self._find_sim_rank_for_batch_torch(batch, ASparse, self.device, mask, mask_new, r)
                with open(sim_rank_name, "wb") as f:
                    pickle.dump(A, f)
            samples_name = f"{self.help_dir}/samples_simrank_" + self.dataset_name + ".pickle"
            if os.path.exists(samples_name):

                with open(samples_name, "rb") as f:
                    pos_batch = pickle.load(f)

            else:
                pos_batch = self._convert_to_samples(batch, A)
                with open(samples_name, "wb") as f:
                    pickle.dump(pos_batch, f)

        elif self.loss["C"] == "PPR":
            alpha = self.alpha
            name_of_file = f"{self.help_dir}/pos_samples_VERSEPPR_" + str(alpha) + "_" + self.dataset_name + ".pickle"
            if os.path.exists(name_of_file):
                with open(name_of_file, "rb") as f:
                    pos_batch = pickle.load(f)
            else:
                adj = self._edge_index_to_adj_train(batch).type(torch.FloatTensor)
                print("1")
                invD = torch.diag(1 / sum(adj.t()))
                invD[torch.isinf(invD)] = 0
                print("2")
                adj_matrix = (1 - alpha) * torch.inverse(torch.diag(torch.ones(len(adj))) - alpha * torch.matmul(invD, adj))
                print("3")
                pos_batch = self._convert_to_samples(batch, adj_matrix)
                print("4")
                with open(name_of_file, "wb") as f:
                    pickle.dump(pos_batch, f)

        return pos_batch

    @staticmethod
    def _convert_to_samples(batch, A):
        pos_batch = []
        batch_l = batch.tolist()
        for x in batch_l:
            for j in batch_l:
                if A[x][j] != torch.tensor(0):
                    pos_batch.append([int(x), int(j), A[x][j]])

        return torch.tensor(pos_batch)

    def _find_sim_rank_for_batch_torch(self, batch, adj, device, mask, mask_new, r):
        t = 10
        # approx with SARW
        batch = batch.to(device)
        adj = adj.to(device)
        SimRank = torch.zeros(len(batch), len(batch)).to(device)  # O(n^2)
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
                SR = len(torch.unique(a_to_compare.nonzero(as_tuple=True)[0]))  # O()
                SimRank[u][nei] = SR / r  #

        return SimRank


class SamplerFactorization(Sampler):
    """
    Sample positive and negative edges for context matrix based unsupervised loss function

    :param dataset_name: (str): Name of the inout dataset
    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    :param help_dir: (str): Name of directory for helping data
    """

    def __init__(self, dataset_name: str, data: Graph, device: device, loss_info: Dict) -> None:
        self.device = device
        self.dataset_name = dataset_name
        self.data = data.to(self.device)
        self.negative_sampler = NegativeSampler(self.data, self.device)
        self.loss = loss_info
        super().__init__()

    def sample(self, batch: Batch) -> Tensor:
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
                if True:
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
                if True:
                    adj_matrix = adj_matrix.type(torch.FloatTensor)
                    invD = torch.diag(1 / sum(adj_matrix.t()))
                    invD[torch.isinf(invD)] = 0
                    context_matrix = (1 - alpha) * torch.inverse(torch.diag(torch.ones(len(adj_matrix))) - alpha * torch.matmul(invD, adj_matrix))

            return context_matrix
        else:
            return adj_matrix


class SamplerAPP(SamplerWithNegSamples):
    """
    Sample positive and negative edges for APP unsupervised loss function

    :param dataset_name: (str): Name of the inout dataset
    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    """

    def __init__(self, dataset_name: str, data: Graph, device: device, loss_info: Dict) -> None:

        self.device = device
        self.dataset_name = dataset_name
        self.data = data.to(self.device)
        self.negative_sampler = NegativeSampler(self.data, self.device)
        self.loss = loss_info
        self.num_negative_samples = self.loss["num_negative_samples"]
        self.alpha = self.loss["alpha"]
        self.r = 200
        self.num_negative_samples *= 10

        new_edge_index = []
        li = data.edge_index.t().tolist()
        for edge in li:
            if edge not in new_edge_index:
                new_edge_index.append(edge)
            if [edge[1], edge[0]] not in new_edge_index:
                new_edge_index.append([edge[1], edge[0]])
        new_edge_index = torch.tensor(new_edge_index).t()
        self.data.edge_index = new_edge_index
        super().__init__()

    def sample(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Sample positive and negative edges for batch of nodes

        :param batch: (Batch): Batch for which sampling should be conducted
        :return: (Tensor,Tensor): positive and negative edges
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return self._pos_sample(batch), self._neg_sample(batch)

    def _pos_sample(self, batch):
        batch = batch.to(self.device)
        len_batch = len(batch)
        mask = torch.tensor([False] * len(self.data.x))
        mask[batch.tolist()] = True

        a, _ = subgraph(batch, self.data.edge_index)
        row, col = a
        row = row.to(self.device)
        col = col.to(self.device)

        ASparse = SparseTensor(row=row, col=col, sparse_sizes=(len_batch, len_batch))
        pos_dict = self._find_PPR_approx(batch, ASparse, self.device, self.r, self.alpha, row)
        pos_batch = []
        for pos_pair in pos_dict:
            pos_row = list(pos_pair)
            pos_row.append(pos_dict[pos_pair])
            pos_batch.append(pos_row)
        return torch.tensor(pos_batch)

    def _find_PPR_approx(self, batch, Adj, device, r, alpha, row):
        N = math.ceil(math.log(1 / (r * alpha), (1 - alpha)))
        length = list(map(lambda x: int((1 - alpha) ** x * alpha * r), list(range(N - 1))))
        r = sum(length)
        dict_data = dict()
        for u in batch:
            if len(torch.where(row == u)[0]) > 0:
                pi_u = Adj.random_walk(u.to(device).repeat(r).flatten(), walk_length=N)
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

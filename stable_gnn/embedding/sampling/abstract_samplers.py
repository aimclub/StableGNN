import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from torch import device
from torch_geometric.data import Batch
from torch_geometric.typing import Tensor
from torch_geometric.utils import subgraph

from stable_gnn.graph import Graph


class BaseSampler(ABC):
    """
    Base class for sampling of positive and negative edges for unsupervised loss function

    :param data: (Graph): Input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param loss_info: (dict): Dict of parameters of unsupervised loss function
    """

    def __init__(self, data: Graph, device: device, loss_info: Dict) -> None:
        self.device = device
        self.data = data.to(self.device)
        self.loss = loss_info
        super(BaseSampler, self).__init__()

    def _edge_index_to_adj_train(self, batch: Tensor) -> Tensor:
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

    @abstractmethod
    def _pos_sample(self, batch: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch: Tensor) -> Tensor:
        """Sample edges. Must be implemented

        :param batch: (Batch): Nodes for sampling positive edges for them
        """
        raise NotImplementedError("Define sample function")


class BaseSamplerWithNegative(BaseSampler):
    """Sampler for negative edges

    :param data: (Graph): Input Graph data
    :param device: (device): Either 'cuda' or 'cpu'
    """

    def __init__(self, data: Graph, device: device, loss_info: Dict) -> None:
        super().__init__(data, device, loss_info)
        self.num_negative_samples = self.loss["num_negative_samples"]

    @staticmethod
    def _not_less_than(num_negative_samples: int, all_negative_samples: List[int]) -> List[int]:  # type: ignore
        if len(all_negative_samples) <= num_negative_samples:
            return all_negative_samples
        if len(all_negative_samples) > num_negative_samples:
            return random.choices(all_negative_samples, k=num_negative_samples)

    @staticmethod
    def _adj_list(edge_index: Tensor) -> Dict[int, List[int]]:  # считаем список рёбер из edge_index
        adj_list: Dict[int, List[int]] = dict()
        for x in list(zip(edge_index[0].tolist(), edge_index[1].tolist())):
            if x[0] in adj_list:
                adj_list[x[0]].append(x[1])
            else:
                adj_list[x[0]] = [x[1]]
        return adj_list

    @staticmethod
    def _torch_list(adj_list: Dict[int, List[int]]) -> Tensor:
        line = list()
        other_line = list()
        for node, neighbors in adj_list.items():
            line += [node] * len(neighbors)
            other_line += neighbors
        return torch.transpose((torch.tensor([line, other_line])), 0, 1)

    def _sample_negative(self, batch: Tensor, num_negative_samples: int) -> Tensor:
        """
        Sample negative edges for batch of nodes

        :param batch: (Batch): Nodes for negative sampling
        :param num_negative_samples: (int): Number of negative samples for each edge
        :return: (Tensor): Negative samples
        """
        a, _ = subgraph(batch, self.data.edge_index)
        adj = self._adj_list(a)
        g = dict()
        batch = batch.tolist()
        for node in batch:
            g[node] = batch

        for node, neighbors in adj.items():
            g[node] = list(
                set(batch) - set(neighbors) - {node}
            )  # тут все элементы которые не являются соседними, но при этом входят в батч
        for node, neg_elem in g.items():
            g[node] = self._not_less_than(
                num_negative_samples, g[node]
            )  # если просят конкретное число негативных примеров, надо либо обрезать либо дублировать
        return self._torch_list(g)

    def sample(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Sample positive and negative edges for batch nodes

        :param batch: (Batch): Nodes for positive and negative sampling from them
        :return: (Tensor, Tensor): positive and negative samples
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return self._pos_sample(batch), self._neg_sample(batch)

    def _neg_sample(self, batch: Tensor) -> Tensor:
        a, _ = subgraph(batch.tolist(), self.data.edge_index)
        neg_batch = self._sample_negative(batch, num_negative_samples=self.num_negative_samples)
        return neg_batch

    @abstractmethod
    def _pos_sample(self, batch: Tensor) -> Tensor:
        raise NotImplementedError

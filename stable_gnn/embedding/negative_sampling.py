import random
from typing import Dict, List

import torch
from torch import device
from torch_geometric.data import Batch
from torch_geometric.typing import Tensor
from torch_geometric.utils import subgraph

from stable_gnn.graph import Graph


class NegativeSampler:
    """
    Sampler for negative edges

    :param data: (Graph): Input Graph data
    :param device: (device): Either 'cuda' or 'cpu'
    """

    def __init__(self, data: Graph, device: device = "cpu") -> None:
        self.data = data
        self.device = device
        super(NegativeSampler, self).__init__()

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

    def negative_sampling(self, batch: Batch, num_negative_samples: int) -> Tensor:
        """
        Sample negative edges for batch of nodes

        :param batch: (Batch): Nodes for negative sampling
        :param num_negative_samples: (int): number of negative samples for each edge
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

from typing import List, Optional, Tuple

import bamt.Networks as Nets
import pandas as pd
import torch
from bamt.Preprocessors import Preprocessor
from pgmpy.estimators import K2Score
from sklearn import preprocessing
from torch.nn import Module
from torch_geometric.typing import Adj, Tensor
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj

from stable_gnn.graph import Graph


class Extrapolate:
    """
    An Extrapolate class for both node and graph classification.

    How to build extrapolation:

    ::

        exptrapolation = Extrapolate(dataset=dataset, model=model)
        (train_dataset, test_dataset, val_dataset,) = Extrapolation(train_indices,val_indices,init_edges,remove_init_edges,white_list,score_func)

    :param dataset: ([Graph]): Dataset of class Graph.
    :param model: (Module): The model to explain.
    """

    def __init__(self, dataset: List[Graph], model: Module) -> None:
        self.data = dataset
        self.model = model
        super(Extrapolate, self).__init__()

    def __call__(
        self,
        train_indices: Tensor,
        val_indices: Tensor,
        init_edges: bool = False,
        remove_init_edges: bool = False,
        white_list: bool = False,
        score_func: str = "MI",
    ) -> Tuple[List[Graph], List[Graph], List[Graph]]:
        """
        Adjust dataset so that to increase extrapolation ability

        :param train_indices: ([int]): List of train indices
        :param val_indices: ([int]): List of validation indices
        :param init_edges: (bool): If True, there would be a list of init edges as start for Learning structure of Bayesian Net(default:'False')
        :param remove_init_edges: If True, it is possible that edges from init_list would be removed during the structure learning of Bayesian Net(default:'False')
        :param white_list: If True, edges inBayesian Net would be only from this white list (default:'False')
        :param score_func: (str): Name of score function to optimize, either 'MI' or 'K2' (default:'MI')
        :return: ([Graph], [Graph], [Graph]): Lists of train, test and validation graphs
        """
        self.init_edges = init_edges
        self.remove_init_edges = remove_init_edges
        self.white_list = white_list
        self.score_func = score_func
        if score_func == "K2":
            self.score = K2Score
        elif score_func == "MI":
            self.score = None
        else:
            raise Exception("there is no ", self.score_func, "score function. Choose one of: MI, K2")

        train_dataset, test_dataset, val_dataset, n_min = self.model.convert_dataset(
            self.data, train_indices, val_indices
        )
        self.n_min = n_min

        data_bamt = self._data_eigen_exctractor(train_dataset)
        bn = self._bayesian_network_build(data_bamt)
        lis = list(
            map(lambda x: self._func(x), bn.edges)
        )  # мы берем только те веришны, которые исходят из y или входят в у
        left_vertices = sorted([x for x in lis if x is not None])
        left_edges = list(filter(lambda x: x[0] == "y" or x[1] == "y", bn.edges))
        left_edges = sorted(left_edges, key=lambda x: int(x[0][5:] if x[1] == "y" else int(x[1][5:])))
        ll = list(map(lambda x: bn.weights[tuple(x)], left_edges))
        len_of_remaining_nodes = len(
            ll
        )  # TODO подумать: мб тут было бы логичнее взять N = число переменных из которых строилась bn
        weights_preprocessed = list(map(lambda x: x * len_of_remaining_nodes / sum(ll), ll))
        train_dataset = self._convolve(train_dataset, weights_preprocessed, left_vertices)
        val_dataset = self._convolve(val_dataset, weights_preprocessed, left_vertices)
        test_dataset = self._convolve(test_dataset, weights_preprocessed, left_vertices)

        return train_dataset, test_dataset, val_dataset

    @staticmethod
    def _func(x: List[str]) -> Optional[int]:
        if x[1] == "y" and len(x[0]) > 1:
            return int(x[0][5:])
        elif x[0] == "y" and len(x[1]) > 1:
            return int(x[1][5:])
        return None

    def _data_eigen_exctractor(self, dataset: List[Graph]) -> pd.DataFrame:
        columns_list = list(map(lambda x: "eigen" + str(x), range(self.n_min)))
        data_bamt = pd.DataFrame(columns=columns_list + ["y"])
        for gr in dataset:
            adj_matrix = to_dense_adj(gr.edge_index)
            eig = torch.real(
                torch.linalg.eig(adj_matrix.reshape(adj_matrix.shape[1], adj_matrix.shape[2]))[0]
            )  # .T[0].T
            ordered, indices = torch.sort(eig[: gr.num_nodes], descending=True)
            to_append = pd.Series(ordered[: self.n_min].tolist() + gr.y.tolist(), index=data_bamt.columns)
            data_bamt = data_bamt.append(to_append, ignore_index=True)

        return data_bamt

    def _bayesian_network_build(self, data_bamt: pd.DataFrame) -> Nets.HybridBN:
        # поиск весов для bamt
        for col in data_bamt.columns[: len(data_bamt.columns)]:
            data_bamt[col] = data_bamt[col].astype(float)
        data_bamt["y"] = data_bamt["y"].astype(int)

        bn = Nets.HybridBN(has_logit=True)
        discretizer = preprocessing.KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
        p = Preprocessor([("discretizer", discretizer)])
        discretized_data, est = p.apply(data_bamt)

        bn.add_nodes(p.info)

        params = dict()
        params["remove_init_edges"] = self.remove_init_edges

        if self.init_edges:
            params["init_edges"] = list(map(lambda x: ("eigen" + str(x), "y"), list(range(self.n_min)))) + list(  # type: ignore
                map(lambda x: ("y", "eigen" + str(x)), list(range(self.n_min)))
            )

        if self.white_list:
            params["white_list"] = list(map(lambda x: ("eigen" + str(x), "y"), list(range(self.n_min)))) + list(  # type: ignore
                map(lambda x: ("y", "eigen" + str(x)), list(range(self.n_min)))
            )

        bn.add_edges(
            discretized_data,
            scoring_function=(self.score_func, self.score),
            params=params,
        )

        bn.calculate_weights(discretized_data)
        bn.plot("BN1.html")
        return bn

    @staticmethod
    def _convolve(dataset: List[Graph], weights: List[float], left_vertices: List[int]) -> List[Graph]:
        new_data = []
        for graph in dataset:
            adj = to_dense_adj(graph.edge_index)
            eigs = torch.linalg.eig(adj.reshape(adj.shape[1], adj.shape[2]))
            eigenvectors = torch.real(eigs[1])

            eig = torch.real(eigs[0])  # .T[0].T
            ordered, indices = torch.sort(eig[: graph.num_nodes], descending=True)
            lef = indices[left_vertices]
            zeroed = torch.tensor(list(set(range(len(eig))) - set(lef.tolist())))
            if len(zeroed) > 0:
                eig[zeroed] = 0

            for e, d in enumerate(lef):
                eig[d] = eig[d] * weights[e]

            eigenvalues = torch.diag(eig)
            convolved = torch.matmul(torch.matmul(eigenvectors, eigenvalues), eigenvectors.T)

            graph.edge_index, graph.edge_weight = dense_to_sparse(convolved)
            graph.edge_weight = graph.edge_weight
            graph.edge_index = graph.edge_index.type(torch.LongTensor)
            new_data.append(graph)
        return new_data

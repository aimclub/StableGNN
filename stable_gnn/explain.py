from typing import List, Optional, Tuple

import bamt.Networks as Nets
import numpy as np

import pandas as pd
import torch
from pgmpy.estimators import BicScore, HillClimbSearch, K2Score
from pgmpy.estimators.CITests import chi_square
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from scipy.special import softmax
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from torch_geometric.typing import Tensor


class Explain:
    """
    An explainer class for instance-level explanations of Graph Neural Networks. Explanation is Probabilistic Graphical Model in a form of Bayesian network. This Bayesian network estimates the probability that node has the predicted role given a realization of other nodes.

    How to build explanation:

    ::

        explainer = Explain(model=model, A=A, X=X)
        pgm_explanation = explainer.structure_learning(28)

    :param model: (Module): The model to explain.
    :param A: (Tensor): Adjacency matrix of input data.
    :param X: (Tensor): Feature matrix of input data.
    """

    def __init__(self, model: Module, adj_matrix: Tensor, features: Tensor) -> None:
        self.model = model
        self.model.eval()
        self.adj_matrix = adj_matrix
        self.features = features
        self.n_hops = len(model.convs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Explainer settings")
        print("\\ A dim: ", self.adj_matrix.shape)
        print("\\ X dim: ", self.features.shape)

    def _n_hops_A(self, n_hops):
        # Compute the n-hops adjacency matrix
        adj = torch.tensor(self.adj_matrix, dtype=torch.float)
        hop_adj = power_adj = adj
        for i in range(n_hops - 1):
            power_adj = power_adj @ adj
            hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
        return hop_adj.numpy().astype(int)

    def _extract_n_hops_neighbors(self, nA, target):
        # Return the n-hops neighbors of a node
        node_nA_row = nA[target]
        print("node_nA_row", sum(node_nA_row))
        neighbors = np.nonzero(node_nA_row)[0]
        target_new = sum(node_nA_row[:target])
        sub_A = self.adj_matrix[neighbors][:, neighbors]
        sub_X = self.features[neighbors]
        return target_new, sub_A, sub_X, neighbors

    def _perturb_features_on_node(self, feature_matrix, target, random=0, mode=0):
        X_perturb = feature_matrix
        if random == 0:
            perturb_array = X_perturb[target]
        else:
            if mode == 0:
                perturb_array = np.random.randint(2, size=X_perturb[target].shape[0])
            elif mode == 1:
                perturb_array = np.multiply(
                    X_perturb[target],
                    np.random.uniform(low=0.0, high=2.0, size=X_perturb[target].shape[0]),
                )
        X_perturb[target] = perturb_array
        return X_perturb

    def _data_generation(
        self, target: int, num_samples: int = 100, pred_threshold: float = 0.1
    ) -> Tuple[pd.DataFrame,]:
        print("Explaining node: " + str(target))
        nA = self._n_hops_A(self.n_hops)
        target_new, sub_A, sub_X, neighbors = self._extract_n_hops_neighbors(nA, target)

        if target not in neighbors:
            neighbors = np.append(neighbors, target)
        print(self.features)
        X_torch = torch.tensor([self.features], dtype=torch.float).squeeze()
        A_torch = torch.tensor([self.adj_matrix], dtype=torch.float).squeeze()

        data = Data(x=X_torch, edge_index=A_torch.nonzero().t().contiguous())

        loader = NeighborSampler(
            data.edge_index,
            node_idx=torch.tensor([True] * len(X_torch)),
            batch_size=int(len(X_torch)),
            sizes=[-1] * 1,
        )
        for batch_size, n_id, adjs in loader:
            if len(loader.sizes) == 1:
                adjs = [adjs]
            adjs = [adj.to(self.device) for adj in adjs]
            edge_index = adjs[0].edge_index
            edge_weight = None
            batch = None
            pred_torch = self.model.forward(
                data.x[n_id.to(self.device)].to(self.device), edge_index, edge_weight, batch, graph_level=False
            )

        soft_pred = np.asarray(
            [softmax(np.asarray(pred_torch[0].cpu()[node_].data)) for node_ in range(self.features.shape[0])]
        )  # TODO кажется это двойная работа по софтмаксу и ниже еще такая строчка есть

        samples = []
        pred_samples = []

        for iteration in range(num_samples):
            X_perturb = self.features.copy()
            sample = []
            for node in neighbors:
                seed = np.random.randint(2)
                if seed == 1:
                    latent = 1
                    X_perturb = self._perturb_features_on_node(X_perturb, node, random=seed)
                else:
                    latent = 0
                sample.append(latent)

            X_perturb_torch = torch.tensor([X_perturb], dtype=torch.float).squeeze()
            A_torch = torch.tensor([self.adj_matrix], dtype=torch.float).squeeze()
            data_perturb = Data(x=X_perturb_torch, edge_index=A_torch.nonzero().t().contiguous())
            loader_perturb = NeighborSampler(
                data_perturb.edge_index,
                node_idx=torch.tensor([True] * len(X_perturb_torch)),
                batch_size=int(len(X_perturb_torch)),
                sizes=[-1] * 1,
            )

            for batch_size, n_id, adjs in loader_perturb:
                if len(loader_perturb.sizes) == 1:
                    adjs = [adjs]
                adjs = [adj.to(self.device) for adj in adjs]
                edge_index = adjs[0].edge_index
                edge_weight = None
                batch = None
                pred_perturb_torch = self.model.forward(
                    data_perturb.x[n_id.to(self.device)].to(self.device),
                    edge_index,
                    edge_weight,
                    batch,
                    graph_level=False,
                )

            soft_pred_perturb = np.asarray(
                [
                    softmax(np.asarray(pred_perturb_torch[0].cpu()[node_].data))
                    for node_ in range(self.features.shape[0])
                ]
            )

            sample_bool = []
            for node in neighbors:
                if (soft_pred_perturb[node, np.argmax(soft_pred[node])] + pred_threshold) < np.max(soft_pred[node]):
                    sample_bool.append(1)
                else:
                    sample_bool.append(0)

            samples.append(sample)
            pred_samples.append(sample_bool)

        samples = np.asarray(samples)
        pred_samples = np.asarray(pred_samples)
        Combine_Samples = samples - samples
        print("combine samples", Combine_Samples.shape)
        for s in range(samples.shape[0]):
            Combine_Samples[s] = np.asarray(
                [samples[s, i] * 10 + pred_samples[s, i] + 1 for i in range(samples.shape[1])]
            )

        data = pd.DataFrame(Combine_Samples)
        return data, neighbors

    def _variable_selection(self, target, top_node=None, num_samples=20, pred_threshold=0.1):
        data, neighbors = self._data_generation(target=target, num_samples=num_samples, pred_threshold=pred_threshold)
        ind_sub_to_ori = dict(
            zip(list(data.columns), neighbors)
        )  # mapping из перечисления 1,...n_neighhbours в индексы самих соседей
        data = data.rename(columns={0: "A", 1: "B"})  # Trick to use chi_square test on first two data columns
        ind_ori_to_sub = dict(zip(neighbors, list(data.columns)))  # mapping индексов соседей в простое перечисление

        p_values = []
        dependent_neighbors = []
        dependent_neighbors_p_values = []

        for node in neighbors:
            chi2, p, _ = chi_square(ind_ori_to_sub[node], ind_ori_to_sub[target], [], data, boolean=False)
            p_values.append(p)
            if p < 0.05:
                dependent_neighbors.append(node)
                dependent_neighbors_p_values.append(p)

        pgm_stats = dict(zip(neighbors, p_values))

        if top_node is None:
            pgm_nodes = dependent_neighbors
        else:
            top_p = np.min((top_node, len(neighbors) - 1))
            ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
            pgm_nodes = [ind_sub_to_ori[node] for node in ind_top_p]

        data = data.rename(columns={"A": 0, "B": 1})
        data = data.rename(columns=ind_sub_to_ori)

        return pgm_nodes, data, pgm_stats

    def structure_learning(
        self,
        target: int,
        top_node: Optional[int] = None,
        num_samples: int = 20,
        pred_threshold: float = 0.1,
        child: Optional[bool] = None,
    ) -> BayesianNetwork:
        """
        Learn structure of Bayesian Net, which represents pgm explanation of target node.

        :param target: (int): Index of the node to be explained
        :param top_node: (int, optional): The number of top the most probable nodes for Bayesian Net. If None, all nodes would be used (default: 'None')
        :param num_samples: (int): The number of samples for data generation, which is used for structure learning, more number of samples -- better learning.
        :param pred_threshold: (float): Probability that the features in each node is perturbed (default: 0.1)
        :param child: (bool, Optional): If False or None, no-child constraint is applied (default: None)
        :return: (BayesianNetwork): Pgm explanation in Bayesian Net form
        """
        subnodes, data, pgm_stats = self._variable_selection(target, top_node, num_samples, pred_threshold)
        print("after var selection", subnodes)
        subnodes = [str(int(node)) for node in subnodes]
        target = str(int(target))
        subnodes_no_target = [node for node in subnodes if node != target]
        data.columns = data.columns.astype(str)

        MK_blanket = self._search_MK(data, target, subnodes_no_target.copy())

        if child is None:
            print(subnodes_no_target)
            est = HillClimbSearch(data[subnodes_no_target])
            pgm_no_target = est.estimate(scoring_method=BicScore(data))
            print("estimation", pgm_no_target.nodes(), pgm_no_target.edges())
            for node in MK_blanket:
                if node != target:
                    pgm_no_target.add_edge(node, target)

            #   Create the pgm

            pgm_explanation = BayesianNetwork()
            for node in pgm_no_target.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_no_target.edges():
                pgm_explanation.add_edge(edge[0], edge[1])

            #   Fit the pgm

            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self._generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self._generalize_others)
            pgm_explanation.fit(data_ex)

        else:
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self._generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self._generalize_others)

            est = HillClimbSearch(data_ex)
            pgm_w_target_explanation = est.estimate(scoring_method=BicScore(data_ex))

            print("estimation", pgm_w_target_explanation.nodes(), pgm_w_target_explanation.edges())

            #   Create the pgm
            pgm_explanation = BayesianNetwork()
            for node in pgm_w_target_explanation.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_w_target_explanation.edges():
                pgm_explanation.add_edge(edge[0], edge[1])

            #   Fit the pgm

            pgm_explanation.fit(data_ex)
            print("we added edges")
        return pgm_explanation

    def structure_learning_bamt(
        self,
        target: int,
        top_node: Optional[int] = None,
        num_samples: int = 20,
        pred_threshold: float = 0.1,
        child: Optional[bool] = None,
    ) -> BayesianNetwork:
        """Learn structure of Bayesian Net, but using bamt framework"""
        subnodes, data, pgm_stats = self._variable_selection(target, top_node, num_samples, pred_threshold)
        subnodes = [str(int(node)) for node in subnodes]
        target = str(int(target))
        subnodes_no_target = [node for node in subnodes if node != target]
        data.columns = data.columns.astype(str)

        if child is None:

            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self._generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self._generalize_others)

            for col in data_ex.columns[: len(data_ex.columns)]:
                data_ex[col] = data_ex[col].astype(int)
            data_ex[target] = data_ex[target].astype(int)

            pgm_explanation = Nets.DiscreteBN()
            p_info = dict()

            p_info["types"] = dict(list(zip(list(data_ex.columns), ["disc_num"] * len(data_ex.columns))))

            print(p_info)
            pgm_explanation.add_nodes(p_info)

            pgm_explanation.add_edges(
                data_ex,
                scoring_function=("BIC",),
                # params=params,
            )

            pgm_explanation.plot("BN1.html")

        else:
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self._generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self._generalize_others)

            est = HillClimbSearch(data_ex)
            pgm_w_target_explanation = est.estimate(scoring_method=BicScore(data_ex))

            #   Create the pgm
            pgm_explanation = BayesianNetwork()
            for node in pgm_w_target_explanation.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_w_target_explanation.edges():
                pgm_explanation.add_edge(edge[0], edge[1])

            #   Fit the pgm
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self._generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self._generalize_others)
            pgm_explanation.fit(data_ex)
            print("we added edges")
        return pgm_explanation

    def pgm_conditional_prob(self, target: int, pgm_explanation: BayesianNetwork, evidence_list: List[str]) -> float:
        """
        Probability of target node, conditioned on the set of neighbours

        :param target (int): Index of node explained
        :param pgm_explanation (Bayesian Net): The Bayesian Net explaining the target
        :param evidence_list ([str]]): List of neighbours to condition on
        :return: (float): The probability of the target node conditioned on 'evidence_list'
        """
        pgm_infer = VariableElimination(pgm_explanation)
        for node in evidence_list:
            if node not in list(pgm_infer.variables):
                print("Not valid evidence list.")
                return None
        evidences = self.generate_evidence(evidence_list)
        elimination_order = [node for node in list(pgm_infer.variables) if node not in evidence_list]
        elimination_order = [node for node in elimination_order if node != target]

        q = pgm_infer.query([target], evidence=evidences, elimination_order=elimination_order, show_progress=False)

        return q.values[0]

    def _generalize_target(self, x):
        if x > 10:
            return x - 10
        else:
            return x

    def _generalize_others(self, x):
        if x == 2:
            return 1
        elif x == 12:
            return 11
        else:
            return x

    def _search_MK(self, data, target, nodes):
        target = str(int(target))
        data.columns = data.columns.astype(str)
        nodes = [str(int(node)) for node in nodes]

        MB = nodes
        while True:
            count = 0
            for node in nodes:
                evidences = MB.copy()
                evidences.remove(node)
                _, p, _ = chi_square(target, node, evidences, data[nodes + [target]], boolean=False)
                if p > 0.05:
                    MB.remove(node)
                    count = 0
                else:
                    count = count + 1
                    if count == len(MB):
                        return MB

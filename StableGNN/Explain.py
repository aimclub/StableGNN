import torch
import numpy as np
from scipy.special import softmax
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

import bamt.Networks as Nets
from bamt.Preprocessors import Preprocessor
from sklearn import preprocessing
from pgmpy.estimators import K2Score

class Explain:
    """
    explanations of GCN model predictions in bayesian net form

    Args:
        model -- trained GCN model whose results we want to explain
        A -- adjacency matrix of input dataset
        X -- attribute matrix of input dataset
        ori_pred --
        num_layers -- number f layers in GCN model
        mode --
        print_result -- 1 if needed to be print, 0 otherwise
    """

    def __init__(self, model, A, X, num_layers, mode=0, print_result=1):
        self.model = model
        self.model.eval()
        self.A = A
        self.X = X
        self.num_layers = num_layers
        self.mode = mode
        self.print_result = print_result
        self.device='cuda'
        print("Explainer settings")
        print("\\ A dim: ", self.A.shape)
        print("\\ X dim: ", self.X.shape)
        print("\\ Number of layers: ", self.num_layers)
        print("\\ Perturbation mode: ", self.mode)
        print("\\ Print result: ", self.print_result)

    def n_hops_A(self, n_hops):
        # Compute the n-hops adjacency matrix
        adj = torch.tensor(self.A, dtype=torch.float)
        hop_adj = power_adj = adj
        for i in range(n_hops - 1):
            power_adj = power_adj @ adj
            hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
        return hop_adj.numpy().astype(int)

    def extract_n_hops_neighbors(self, nA, node_idx):
        # Return the n-hops neighbors of a node
        node_nA_row = nA[node_idx]
        print('node_nA_row', sum(node_nA_row))
        neighbors = np.nonzero(node_nA_row)[0]
        node_idx_new = sum(node_nA_row[:node_idx])
        sub_A = self.A[neighbors][:, neighbors]
        sub_X = self.X[neighbors]
        return node_idx_new, sub_A, sub_X, neighbors

    def perturb_features_on_node(self, feature_matrix, node_idx, random=0, mode=0):
        # return a random perturbed feature matrix
        # random = 0 for nothing, 1 for random.
        # mode = 0 for random 0-1, 1 for scaling with original feature

        X_perturb = feature_matrix
        if random == 0:
            perturb_array = X_perturb[node_idx]
        else:
            if mode == 0:
                perturb_array = np.random.randint(2, size=X_perturb[node_idx].shape[0])
            elif mode == 1:
                perturb_array = np.multiply(X_perturb[node_idx],np.random.uniform(low=0.0, high=2.0, size=X_perturb[node_idx].shape[0]),)
        X_perturb[node_idx] = perturb_array
        return X_perturb

    def DataGeneration(self, node_idx, num_samples=100, pred_threshold=0.1):
        print("Explaining node: " + str(node_idx))
        nA = self.n_hops_A(self.num_layers)
        node_idx_new, sub_A, sub_X, neighbors = self.extract_n_hops_neighbors(nA, node_idx)
        if (node_idx not in neighbors):
            neighbors = np.append(neighbors, node_idx)
        print(self.X)
        X_torch = torch.tensor([self.X], dtype=torch.float).squeeze()
        A_torch = torch.tensor([self.A], dtype=torch.float).squeeze()

        data = Data(x=X_torch, edge_index = A_torch.nonzero().t().contiguous())

        loader = NeighborSampler(
            data.edge_index,
            node_idx=torch.tensor([True]*len(X_torch)),
            batch_size=int(len(X_torch)),
            sizes=[-1] * 1,
        )
        for batch_size, n_id, adjs in loader:
            if len(loader.sizes) == 1:
                adjs = [adjs]
            adjs = [adj.to(self.device) for adj in adjs]
            pred_torch = self.model.forward(data.x[n_id.to(self.device)].to(self.device), adjs)


      # pred_torch, _ = self.model.forward(data, )
        soft_pred = np.asarray([softmax(np.asarray(pred_torch.cpu()[node_].data)) for node_ in range(self.X.shape[0])]) #TODO кажется это двойная работа по софтмаксу и ниже еще такая строчка есть

    #    pred_node = np.asarray(pred_torch[0][node_idx].data)
     #   label_node = np.argmax(pred_node)
      #  soft_pred_node = softmax(pred_node)

        Samples = []
        Pred_Samples = []

        for iteration in range(num_samples):
            X_perturb = self.X.copy()
            sample = []
            for node in neighbors:
                seed = np.random.randint(2)
                if seed == 1:
                    latent = 1
                    X_perturb = self.perturb_features_on_node(X_perturb, node, random=seed)
                else:
                    latent = 0
                sample.append(latent)

            X_perturb_torch = torch.tensor([X_perturb], dtype=torch.float).squeeze()
            A_torch = torch.tensor([self.A], dtype=torch.float).squeeze()
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
                pred_perturb_torch = self.model.forward(data_perturb.x[n_id.to(self.device)].to(self.device), adjs)

            soft_pred_perturb = np.asarray(
                [softmax(np.asarray(pred_perturb_torch.cpu()[node_].data)) for node_ in range(self.X.shape[0])])

            sample_bool = []
            for node in neighbors:
                if (soft_pred_perturb[node, np.argmax(soft_pred[node])] + pred_threshold) < np.max(soft_pred[node]):
                    sample_bool.append(1)
                else:
                    sample_bool.append(0)

            Samples.append(sample)
            Pred_Samples.append(sample_bool)

        Samples = np.asarray(Samples)
        Pred_Samples = np.asarray(Pred_Samples)
        Combine_Samples = Samples - Samples
        print('combine samples',Combine_Samples.shape)
        for s in range(Samples.shape[0]):
            Combine_Samples[s] = np.asarray(
                [Samples[s, i] * 10 + Pred_Samples[s, i] + 1 for i in range(Samples.shape[1])])

        data = pd.DataFrame(Combine_Samples)
        return data, neighbors

    def VariableSelection(self, data, neighbors, node_idx, top_node=None, p_threshold=0.05):

        ind_sub_to_ori = dict(zip(list(data.columns), neighbors)) #mapping из перечисления 1,...n_neighhbours в индексы самих соседей
        data = data.rename(columns={0: "A", 1: "B"})  # Trick to use chi_square test on first two data columns
        ind_ori_to_sub = dict(zip(neighbors, list(data.columns)))#mapping индексов соседей в простое перечисление

        p_values = []
        dependent_neighbors = []
        dependent_neighbors_p_values = []

        for node in neighbors:
            chi2, p, _ = chi_square(ind_ori_to_sub[node], ind_ori_to_sub[node_idx], [], data, boolean=False)
            p_values.append(p)
            if p < p_threshold:
                dependent_neighbors.append(node)
                dependent_neighbors_p_values.append(p)

        pgm_stats = dict(zip(neighbors, p_values))

        if top_node == None:
            pgm_nodes = dependent_neighbors
        else:
            top_p = np.min((top_node, len(neighbors) - 1))
            ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
            pgm_nodes = [ind_sub_to_ori[node] for node in ind_top_p]

        data = data.rename(columns={"A": 0, "B": 1})
        data = data.rename(columns=ind_sub_to_ori)

        return pgm_nodes, data, pgm_stats

    def StructureLearning(self, target, data, subnodes, child=None):
        print('after var selection', subnodes)
        subnodes = [str(int(node)) for node in subnodes]
        target = str(int(target))
        subnodes_no_target = [node for node in subnodes if node != target]
        data.columns = data.columns.astype(str)

        MK_blanket = self.search_MK(data, target, subnodes_no_target.copy())

        if child == None:
            print(subnodes_no_target)
            est = HillClimbSearch(data[subnodes_no_target])
            pgm_no_target = est.estimate(scoring_method=BicScore(data))
            print('estimation',pgm_no_target.nodes(),pgm_no_target.edges())
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
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)
            pgm_explanation.fit(data_ex)

        else:
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)

            est = HillClimbSearch(data_ex)
            pgm_w_target_explanation = est.estimate(scoring_method=BicScore(data_ex))
            print('estimation', pgm_w_target_explanation.nodes(), pgm_w_target_explanation.edges())
            #   Create the pgm
            pgm_explanation = BayesianNetwork()
            for node in pgm_w_target_explanation.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_w_target_explanation.edges():
                pgm_explanation.add_edge(edge[0], edge[1])

            #   Fit the pgm

            pgm_explanation.fit(data_ex)
            print('we added edges')
        return pgm_explanation
    def StructureLearning_bamt(self, target, data, subnodes, child=None):

        subnodes = [str(int(node)) for node in subnodes]
        target = str(int(target))
        subnodes_no_target = [node for node in subnodes if node != target]
        data.columns = data.columns.astype(str)

        MK_blanket = self.search_MK(data, target, subnodes_no_target.copy())

        if child == None:
            print(subnodes_no_target)
           # est = HillClimbSearch(data[subnodes_no_target])
           # pgm_no_target = est.estimate(scoring_method=BicScore(data))

            #for node in MK_blanket:
             #   if node != target:
              #      pgm_no_target.add_edge(node, target)

            #   Create the pgm

            #pgm_explanation = BayesianNetwork()
            #for node in pgm_no_target.nodes():
             #   pgm_explanation.add_node(node)
            #for edge in pgm_no_target.edges():
             #   pgm_explanation.add_edge(edge[0], edge[1])

            #   Fit the pgm

            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)

            for col in data_ex.columns[: len(data_ex.columns)]:
                data_ex[col] = data_ex[col].astype(int)
            data_ex[target] = data_ex[target].astype(int)

            pgm_explanation = Nets.DiscreteBN()
            p_info=dict()
            p_info['types']=dict(list(zip(list(data_ex.columns), ['disc_num']*len(data_ex.columns))))
            print(p_info)
            pgm_explanation.add_nodes(p_info)

            pgm_explanation.add_edges(
                data_ex,
                scoring_function=('BIC',),
                #params=params,
            )

            #pgm_explanation.calculate_weights(discretized_data)
            pgm_explanation.plot("BN1.html")

        else:
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)

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
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)
            pgm_explanation.fit(data_ex)
            print('we added edges')
        return pgm_explanation

    #    return (f'{self.__class__.__name__}({self.in_channels}, '
    #           f'{self.out_channels}, heads={self.heads}, '
    #          f'type={self.attention_type})')

    def pgm_conditional_prob(self, target, pgm_explanation, evidence_list):
        pgm_infer = VariableElimination(pgm_explanation)
        for node in evidence_list:
            if node not in list(pgm_infer.variables):
                print("Not valid evidence list.")
                return None
        evidences = self.generate_evidence(evidence_list)
        elimination_order = [node for node in list(pgm_infer.variables) if node not in evidence_list]
        elimination_order = [node for node in elimination_order if node != target]
        q = pgm_infer.query([target], evidence=evidences,
                            elimination_order=elimination_order, show_progress=False)
        return q.values[0]

    def search_MK(self, data, target, nodes):
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
    def generalize_target(self, x):
        if x > 10:
            return x - 10
        else:
            return x

    def generalize_others(self, x):
        if x == 2:
            return 1
        elif x == 12:
            return 11
        else:
            return x
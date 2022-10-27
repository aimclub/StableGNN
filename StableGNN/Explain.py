import torch
import numpy as np
from scipy.special import softmax
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from pgmpy.estimators.CITests import chi_square

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

    def __init__(self, model, A, X, ori_pred, num_layers, mode=0, print_result=1):
        self.model = model
        self.model.eval()
        self.A = A
        self.X = X
        self.ori_pred = ori_pred
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

        X_torch = torch.tensor([np.array(self.X)], dtype=torch.float).squeeze()
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
        print(pred_torch.cpu())
        soft_pred = np.asarray([softmax(np.asarray(pred_torch.cpu()[node_].data)) for node_ in range(self.X.shape[0])])

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
            print('X torch perturb', iteration, X_perturb_torch.shape)

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
        for s in range(Samples.shape[0]):
            Combine_Samples[s] = np.asarray(
                [Samples[s, i] * 10 + Pred_Samples[s, i] + 1 for i in range(Samples.shape[1])])

        data = pd.DataFrame(Combine_Samples)
        ind_sub_to_ori = dict(zip(list(data.columns), neighbors))
        data = data.rename(columns={0: "A", 1: "B"})  # Trick to use chi_square test on first two data columns
        return data, neighbors

    def VariableSelection(self,data,neighbors,node_idx,top_node=None, p_threshold=0.05):
        ind_sub_to_ori = dict(zip(list(data.columns), neighbors))
        data = data.rename(columns={0: "A", 1: "B"})  # Trick to use chi_square test on first two data columns
        ind_ori_to_sub = dict(zip(neighbors, list(data.columns)))

        p_values = []
        dependent_neighbors = []
        dependent_neighbors_p_values = []
        for node in neighbors:

            chi2, p = chi_square(ind_ori_to_sub[node], ind_ori_to_sub[node_idx], [], data)
            p_values.append(p)
            if p < p_threshold:
                dependent_neighbors.append(node)
                dependent_neighbors_p_values.append(p)

        pgm_stats = dict(zip(neighbors, p_values))

        pgm_nodes = []
        if top_node == None:
            pgm_nodes = dependent_neighbors
        else:
            top_p = np.min((top_node, len(neighbors) - 1))
            ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
            pgm_nodes = [ind_sub_to_ori[node] for node in ind_top_p]

        data = data.rename(columns={"A": 0, "B": 1})
        data = data.rename(columns=ind_sub_to_ori)

        return pgm_nodes, pgm_stats

    def StructureLearning(self):
        pass

    #    return (f'{self.__class__.__name__}({self.in_channels}, '
    #           f'{self.out_channels}, heads={self.heads}, '
    #          f'type={self.attention_type})')
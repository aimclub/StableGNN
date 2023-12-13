import random
from typing import List

import torch
import torch_geometric.transforms as T
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator
from torch_geometric.datasets import Planetoid

from stable_gnn.embedding import EmbeddingFactory
from stable_gnn.embedding.sampling.samplers import NegativeSampler
from stable_gnn.graph import Graph


class ModelLinkPrediction():
    """
    Model for Link Prediction task with unsupervised embeddings

    :param dataset: (Graph): Input Graph
    :param number_of_trials (int): Number of trials for optuna tuning embeddings
    :param device: (device): Device 'cuda' or 'cpu'
    :param emb_conv_name: (str): Name of convolution for embedding learning
    :param loss_name: (str): Name of loss function for embedding learning 
    """

    def __init__(
        self,
        number_of_trials: int,
        device: torch.device = "cuda",
        loss_name: str = "APP",
        emb_conv_name: str = "SAGE",
    ) -> None:
        super().__init__()


        self.number_of_trials=number_of_trials
        self.loss_name=loss_name
        self.emb_conv_name=emb_conv_name
        self.device = device



    def train_test_edges(self, dataset: Graph) -> (List[List[int]], List[List[int]],List[List[int]],List[List[int]]):
        '''
        Split dataset to train and test and calculate negative samples

        :param dataset: (Graph): Data to split on train, test and negatives
        :return: (Tuple): Tuple of four lists of train edges, negativу train samples, test and negative test samples edges
        '''
        self.data = dataset[0]
        self.data.edge_index = self.data.edge_index.type(torch.LongTensor)

        all_edges = self.data.edge_index.T.tolist()
        train_edges = []
        test_edges = []
        indices_train_edges = random.sample(range(len(all_edges)), int(len(all_edges) * 0.8))
        for i, edge in enumerate(all_edges):
            if i in indices_train_edges:
                train_edges.append(edge)
            else:
                test_edges.append(edge)

        neg_samples_train =self._neg_samples(train_edges, self.data)
        neg_samples_test = self._neg_samples(test_edges, self.data)
        self.data.edge_index = torch.LongTensor(train_edges).T
        return train_edges, neg_samples_train, test_edges, neg_samples_test

    def _neg_samples(self, positive_edges: List[int], data: Graph) -> List[int]:
        num_neg_samples_test = int(len(positive_edges) / len(self.data.x))
        num_neg_samples_test = num_neg_samples_test if num_neg_samples_test > 0 else 1
        ns = NegativeSampler(data=data, device=self.device, loss_info={"num_negative_samples": num_neg_samples_test})
        neg_edges = ns._sample_negative(
            torch.LongTensor(list(range(len(self.data.x)))).to(self.device), num_negative_samples=num_neg_samples_test
        )
        return neg_edges

    def train_cl(self, train_edges: List[List[int]], neg_samples_train: List[List[int]]) -> BaseEstimator:
        '''
        Train classifier for link prediction

        :param train_edges: (List): List of existing edges
        :param neg_samples_train: (List): List of negative samples to train
        :return: (BaseEstimator): Classifier which support fit predict notation
        '''
        self.embeddings = EmbeddingFactory().build_embeddings(
            loss_name=self.loss_name, conv=self.emb_conv_name, data=[self.data], device=self.device, number_of_trials=self.number_of_trials,
            tune_out=True
        )

        emb_norm = torch.nn.functional.normalize(torch.tensor(self.embeddings))
        self.clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=5, random_state=0)
        x_pred = []
        for edge in train_edges:
            x_pred.append(torch.concat((emb_norm[edge[0]], emb_norm[edge[1]])).tolist())
        for edge in neg_samples_train:
            x_pred.append(torch.concat((emb_norm[edge[0]], emb_norm[edge[1]])).tolist())

        true_train = [1] * len(train_edges) + [0] * len(neg_samples_train)
        self.clf.fit(x_pred, true_train)
        return self.clf

    def test(self, clf: BaseEstimator, test_edges: List[List[int]], neg_samples_test: List[List[int]] ) -> float:
        '''
        Calculate f1 measure for test edges
        
        :param: cl (BaseEstimator)
        :param test_edges: (List): List of existing edges to test on
        :param neg_samples_test: (List): List of negative samples to test on
        :return: (float): Value of f1 measure
        '''
        emb_norm = torch.nn.functional.normalize(torch.tensor(self.embeddings))
        pred_test = []
        for edge in test_edges:
            pred_test.append(torch.concat((emb_norm[edge[0]], emb_norm[edge[1]])).tolist())

        for edge in neg_samples_test:
            pred_test.append(torch.concat((emb_norm[edge[0]], emb_norm[edge[1]])).tolist())
        y_pred = clf.predict(pred_test)
        y_true = [1] * len(test_edges) + [0] * len(neg_samples_test)
        return f1_score(y_true, y_pred)


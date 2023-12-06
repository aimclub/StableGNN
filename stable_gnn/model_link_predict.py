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
        dataset: Graph,
        number_of_trials: int,
        device: torch.device = "cuda",
        loss_name: str = "APP",
        emb_conv_name: str = "SAGE",
    ) -> None:
        super().__init__()

        self.data = dataset[0]
        self.data.edge_index = self.data.edge_index.type(torch.LongTensor)
        # это для того чтоб тестовые негативные примеры не включали

        train_edges, test_edges = self._train_test_edges(self.data)
        self.data.edge_index = torch.LongTensor(train_edges).T
        self.positive_edges = test_edges
        self.device = device
        self.neg_samples_test = self._neg_samples(self.positive_edges, self.data)
        self.neg_samples_train = self._neg_samples(train_edges, self.data)

        self.embeddings = EmbeddingFactory().build_embeddings(
            loss_name=loss_name, conv=emb_conv_name, data=dataset, device=device, number_of_trials=number_of_trials
        )

    def _train_test_edges(self, data: Graph) -> (List[int], List[int]):
        all_edges = data.edge_index.T.tolist()
        train_edges = []
        test_edges = []
        indices_train_edges = random.sample(range(len(all_edges)), int(len(all_edges) * 0.8))
        for i, edge in enumerate(all_edges):
            if i in indices_train_edges:
                train_edges.append(edge)
            else:
                test_edges.append(edge)
        return train_edges, test_edges

    def _neg_samples(self, positive_edges: List[int], data: Graph) -> List[int]:
        num_neg_samples_test = int(len(positive_edges) / len(self.data.x))
        num_neg_samples_test = num_neg_samples_test if num_neg_samples_test > 0 else 1
        ns = NegativeSampler(data=data, device=self.device, loss_info={"num_negative_samples": num_neg_samples_test})
        neg_edges = ns._sample_negative(
            torch.LongTensor(list(range(len(self.data.x)))).to(self.device), num_negative_samples=num_neg_samples_test
        )
        return neg_edges

    def train_cl(self) -> BaseEstimator:
        '''
        Train classifier for link prediction

        :return: (BaseEstimator): Classifier which support fit predict notation
        '''
        emb_norm = torch.nn.functional.normalize(torch.tensor(self.embeddings))
        self.clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=5, random_state=0)
        x_pred = []
        for edge in self.data.edge_index.T:
            x_pred.append(torch.concat((emb_norm[edge[0]], emb_norm[edge[1]])).tolist())
        for edge in self.neg_samples_train:
            x_pred.append(torch.concat((emb_norm[edge[0]], emb_norm[edge[1]])).tolist())

        true_train = [1] * len(self.data.edge_index.T) + [0] * len(self.neg_samples_train)
        self.clf.fit(x_pred, true_train)
        return self.clf

    def test(self) -> float:
        '''
        Calculate f1 measure for test edges

        :return: (float): Value of f1 measure
        '''
        emb_norm = torch.nn.functional.normalize(torch.tensor(self.embeddings))
        pred_test = []
        for edge in self.positive_edges:
            pred_test.append(torch.concat((emb_norm[edge[0]], emb_norm[edge[1]])).tolist())

        for edge in self.neg_samples_test:
            pred_test.append(torch.concat((emb_norm[edge[0]], emb_norm[edge[1]])).tolist())
        y_pred = self.clf.predict(pred_test)
        y_true = [1] * len(self.positive_edges) + [0] * len(self.neg_samples_test)
        return f1_score(y_true, y_pred)


if __name__ == "__main__":
    data = Planetoid(root="/tmp/" + str("name"), name="Citeseer", transform=T.NormalizeFeatures())
    model = ModelLinkPrediction(data)
    clf = model.train_cl()
    print("f1", (model.test()))

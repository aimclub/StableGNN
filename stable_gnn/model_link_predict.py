import torch
from stable_gnn.graph import Graph
import random
from stable_gnn.embedding import EmbeddingFactory
from stable_gnn.embedding.sampling.samplers import NegativeSampler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np

class ModelLinkPrediction(torch.nn.Module):
    """
    Model for Link Prediction task with unsupervised embeddings

    :param dataset: (Graph): Input Graph
    :param device: (device): Device 'cuda' or 'cpu'
    :param hidden_layer: (int): The size of hidden layer (default: 64)
    :param dropout: (float): Dropout (defualt: 0.0)
    :param num_layers: (int): Number of layers in the model (default:2)
    :param ssl_flag: (bool): If True, self supervised loss will be optimized additionally to semi-supervised (default: False)
    :param loss_name: (str): Name of loss function for embedding learning in GeomGCN layer
    """

    def __init__(
        self,
        dataset: Graph,
        device: torch.device='cuda',
        hidden_layer: int = 64,
        out_layer: int = 32,
        dropout: float = 0.0,
        num_layers: int = 2,
        loss_name: str = "APP",
        emb_conv_name: str = "SAGE",
    ) -> None:
        super().__init__()

        self.hidden_layer = hidden_layer
        self.out_layer = out_layer
        self.dropout = dropout
        self.size = num_layers
        self.data = dataset[0]
        self.data.edge_index = self.data.edge_index.type(torch.LongTensor)
        # это для того чтоб тестовые негативные примеры не включали

        train_edges, test_edges = self.train_test_edges(self.data)
        self.data.edge_index = torch.LongTensor(train_edges).T
        self.positive_edges = test_edges
        self.device = device
        self.neg_samples = self.neg_samples(self.positive_edges, self.data)
        ###вот отюсда доделывать
        #модифицированный dataset с positive links обрезанными

        self.embeddings = EmbeddingFactory().build_embeddings(
            loss_name=loss_name, conv=emb_conv_name, data=dataset, device=device
        )

    def train_test_edges(self, data):
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

    def neg_samples(self, positive_edges, data):
        num_neg_samples_test = int(len(positive_edges) / len(self.data.x))
        num_neg_samples_test = num_neg_samples_test if num_neg_samples_test > 0 else 1
        ns = NegativeSampler(data=data, device=self.device, loss_info={'num_negative_samples': num_neg_samples_test})
        neg_edges = ns._sample_negative(
            torch.LongTensor(list(range(len(self.data.x)))).to(self.device), num_negative_samples=num_neg_samples_test
        )
        return neg_edges

    def predict(self):
        emb_norm = torch.nn.functional.normalize(torch.tensor(self.embeddings))

        pred_test = []
        for edge in self.positive_edges:
            pred_test.append((torch.dot(emb_norm[edge[0]], emb_norm[edge[1]])))
        # print(torch.sigmoid(torch.dot(emb_norm[edge[0]],emb_norm[edge[1]])))

        for edge in self.neg_samples:
            pred_test.append((torch.dot(emb_norm[edge[0]], emb_norm[edge[1]])))
        return pred_test

    def roc_auc(self):
        true_test = [1] * len(self.positive_edges) + [0] * len(self.neg_samples)

        pred_test = self.predict()
        return roc_auc_score(true_test, pred_test)
    def treshold_selection(self):
        # предсказания модели
        pred_test = self.predict()
        # вычисление ROC
        true_test = [1] * len(self.positive_edges) + [0] * len(self.neg_samples)
        fpr, tpr, thresholds = roc_curve(true_test, pred_test)

        # поиск оптимального порога (например, максимальное значение TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        return optimal_threshold

    def f1_measure(self):
        # предсказания модели
        pred_test = self.predict()
        # вычисление ROC
        true_test = [1] * len(self.positive_edges) + [0] * len(self.neg_samples)
        optimal_threshold = self.treshold_selection()

        # преобразование предсказаний в бинарные классы
        y_pred_binary = (pred_test > optimal_threshold).astype(int)

        # расчет F1-меры
        f1 = f1_score(true_test, y_pred_binary)
        return f1

if __name__ == "__main__":
    data = Planetoid(root='/tmp/' + str('name'), name='Citeseer', transform=T.NormalizeFeatures())
    model = ModelLinkPrediction(data)
    print(model.treshold_selection())
    print(model.f1_measure())
    print(model.roc_auc())
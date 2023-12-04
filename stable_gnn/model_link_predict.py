import torch
from stable_gnn.graph import Graph
import random
from stable_gnn.embedding import EmbeddingFactory
from stable_gnn.embedding.sampling.abstract_samplers import BaseSamplerWithNegative
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

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
        self.data = dataset
        self.data.edge_index = self.data.edge_index.type(torch.LongTensor)
        # это для того чтоб тестовые негативные примеры не включали

        train_edges, test_edges = self.train_test_edges(self.data)
        self.data.edge_index = torch.LongTensor(train_edges).T
        self.positive_edges = test_edges

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
        print("first num neg samples test", (num_neg_samples_test))
        num_neg_samples_test = num_neg_samples_test if num_neg_samples_test > 0 else 1
        ns = BaseSamplerWithNegative(data=data, device=device, loss_info={'num_negaive_samples': num_neg_samples_test})
        neg_edges = ns._sample_negative(
            torch.LongTensor(list(range(len(self.data.x)))), num_negative_samples=num_neg_samples_test
        )
        return neg_edges

    def predict(self):
        emb_norm = torch.nn.functional.normalize(torch.tensor(self.embeddings.detach().cpu()))

        pred_test = []
        for edge in self.positive_edges:
            pred_test.append((torch.dot(emb_norm[edge[0]], emb_norm[edge[1]])))
        # print(torch.sigmoid(torch.dot(emb_norm[edge[0]],emb_norm[edge[1]])))
        neg_samples = self.neg_samples(self.positive_edges, self.data)
        for edge in neg_samples:
            pred_test.append((torch.dot(emb_norm[edge[0]], emb_norm[edge[1]])))

        true_test = [1] * len(self.positive_edges) + [0] * len(neg_samples)

        return roc_auc_score(true_test, pred_test)

if __name__ == "__main__":
    data = Planetoid(root='/tmp/' + str('name'), name='Citeseer', transform=T.NormalizeFeatures())
    model = ModelLinkPrediction(data)
    print(model.predict())
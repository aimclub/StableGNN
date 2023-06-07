import torch
from torch import device
from stable_gnn.graph import Graph
from stable_gnn.negative_sampling import NegativeSampler
import random
from torch_geometric.loader import NeighborSampler

class ModelLinkPrediction(torch.nn.Module):
    """
    Model for Link Prediction task with

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
        dataset_name: str,
        loss: dict,
        hidden_layer: int = 64,
        out_layer: int = 32,
        dropout: float = 0.0,
        num_layers: int = 2,
        emb_conv_name: str = "SAGE",

    ) -> None:
        super().__init__()

        self.hidden_layer = hidden_layer
        self.out_layer = out_layer
        self.dropout = dropout
        self.size = num_layers
        self.data = dataset[0]
        self.dataset_name = dataset_name
        self.data.edge_index = self.data.edge_index.type(torch.LongTensor)
        self.loss = loss
          # это для того чтоб тестовые негативные примеры не включали

        train_edges, test_edges = self.train_test_edges(self.data)
        self.data.edge_index = torch.LongTensor(train_edges).T

        Sampler = self.loss["Sampler"]
        self.LossSampler = Sampler(self.datasetname, self.data, device=self.device,
                              mask=torch.BoolTensor([True] * len(self.data.x)), loss_info=self.loss, help_dir="../data_help/")

        self.positive_edges = test_edges
        ###вот отюсда доделывать
        self.neg_samples_test = self.neg_samples(self.positive_edges, self.data)



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

    def neg_samples(self,positive_edges, data):
        ns = NegativeSampler(data=data)

        num_neg_samples_test = int(len(positive_edges) / len(self.data.x))
        print('first num neg samples test', (num_neg_samples_test))
        num_neg_samples_test = num_neg_samples_test if num_neg_samples_test > 0 else 1
        neg_edges = ns.negative_sampling(torch.LongTensor(list(range(len(self.data.x)))),
                             num_negative_samples=num_neg_samples_test)
        return neg_edges

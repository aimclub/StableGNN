from typing import Any, Dict, Tuple

import optuna
import torch
from optuna import Trial
from torch import device
from torch.optim import Optimizer
from torch_geometric.loader import NeighborSampler
from torch_geometric.typing import Tensor

from stable_gnn.embedding.models import ModelFactory
from stable_gnn.embedding.models.abstract_model import BaseNet
from stable_gnn.embedding.sampling.abstract_samplers import BaseSampler
from stable_gnn.graph import Graph


class ModelTrainEmbeddings:
    """Model for training Net, which building embeddings for Geom-GCN layer

    :param data: (Graph): Input Graph
    :param loss_function: (dict): Dict of parameters of unsupervised loss function
    :param conv: (str): Name of convolution (default:'GCN')
    :param device: (device): Either 'cuda' or 'cpu' (default:'cuda')
    """

    def __init__(self, data: Graph, loss_function: Dict, device: device, conv: str = "GCN") -> None:
        self.conv = conv
        self.device = device
        self.x = data.x
        self.y = data.y.squeeze()
        self.data = data.to(device)
        self.train_mask = torch.Tensor([True] * data.num_nodes)
        self.loss = loss_function
        super(ModelTrainEmbeddings, self).__init__()

    def _sampling(self, sampler: BaseSampler, epoch: int, nodes: Tensor) -> None:
        if epoch == 0:
            self.samples = sampler.sample(nodes.to(self.device))

    def _train(
        self,
        model: BaseNet,
        data: Graph,
        optimizer: Optimizer,
        sampler: BaseSampler,
        train_loader: NeighborSampler,
        dropout: float,
        epoch: int,
    ) -> Tuple[Tensor, Tensor]:
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        if self.conv == "GCN":
            out = model.inference(data.to(self.device), dp=dropout)
            loss = model.loss(out[self.train_mask], self.samples)
            total_loss += loss
        else:
            for batch_size, n_id, adjs in train_loader:
                if len(train_loader.sizes) == 1:
                    adjs = [adjs]
                adjs = [adj.to(self.device) for adj in adjs]
                out = model.forward(data.x[n_id.to(self.device)].to(self.device), adjs)
                self._sampling(sampler, epoch, n_id[:batch_size])
                loss = model.loss(out, self.samples)
                total_loss += loss

        total_loss.backward()  # type: ignore
        optimizer.step()
        return total_loss / len(train_loader), out

    def run(self, params: Dict) -> Tensor:
        """
        Learn embeddings

        :param params: dict[str,float,int,float]: Parameters for learning: size of hidden layer, dropout, number of layers for the model, learning rate
        :return: (Tensor): The output embeddings
        """
        hidden_layer = params["hidden_layer"]
        dropout = params["dropout"]
        size = params["size of network, number of convs"]
        learning_rate = params["lr"]
        train_loader = NeighborSampler(self.data.edge_index, batch_size=self.data.num_nodes, sizes=[-1] * size)

        sampler = self.loss["sampler"]

        loss_sampler = sampler(data=self.data, device=self.device, loss_info=self.loss)
        model = ModelFactory().build_model(
            num_features=self.data.x.shape[1],
            conv=self.conv,
            loss_function=self.loss,
            device=self.device,
            hidden_layer=hidden_layer,
            out_layer=2,
            num_layers=size,
            dropout=dropout,
        )
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(99):
            loss, _ = self._train(model, self.data, optimizer, loss_sampler, train_loader, dropout, epoch)
        _, out = self._train(model, self.data, optimizer, loss_sampler, train_loader, dropout, epoch)

        return out


class OptunaTrainEmbeddings(ModelTrainEmbeddings):
    """
    Model for training Net, wcich building embeddings for Geom-GCN layer

    :param loss_function: (dict): Dict of parameters of unsupervised loss function
    :param conv: (str): Name of convolution (default:'GCN')
    :param device: (device): Either 'cuda' or 'cpu' (default:'cuda')
    """

    def _objective(self, trial: Trial) -> Tensor:
        # Integer parameter
        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        size = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        learning_rate = trial.suggest_float("lr", 5e-3, 1e-2)

        loss_to_train = {}
        for name in self.loss:
            if type(self.loss[name]) == list:
                if len(self.loss[name]) == 3:
                    var = trial.suggest_int(
                        name,
                        self.loss[name][0],
                        self.loss[name][1],
                        step=self.loss[name][2],
                    )
                    loss_to_train[name] = var
                elif len(self.loss[name]) == 2:
                    var_2 = trial.suggest_float(name, self.loss[name][0], self.loss[name][1])
                    loss_to_train[name] = var_2
                else:
                    var_3 = trial.suggest_categorical(name, self.loss[name])
                    loss_to_train[name] = var_3
            else:
                loss_to_train[name] = self.loss[name]

            if name == "q" and type(self.loss[name]) == list:
                var_5 = trial.suggest_categorical("p", self.loss["p"])
                var_4 = trial.suggest_categorical("q", self.loss[name])
                if var_4 > 1:
                    var_4 = 1
                if var_5 < var_4:
                    var_5 = var_4
                loss_to_train["q"] = var_4
                loss_to_train["p"] = var_5

        sampler = loss_to_train["sampler"]
        model = ModelFactory().build_model(
            num_features=self.data.x.shape[1],
            conv=self.conv,
            loss_function=loss_to_train,
            device=self.device,
            hidden_layer=hidden_layer,
            out_layer=2,
            num_layers=size,
            dropout=dropout,
        )
        train_loader = NeighborSampler(self.data.edge_index, batch_size=int(self.data.num_nodes), sizes=[-1] * size)

        loss_sampler = sampler(
            data=self.data,
            device=self.device,
            loss_info=loss_to_train,
        )
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(50):
            loss, _ = self._train(
                model,
                self.data,
                optimizer,
                loss_sampler,
                train_loader,
                dropout,
                epoch,
            )
        return loss

    def run(self, number_of_trials: int) -> Dict[Any, Any]:  # type: ignore
        """Tuning parameters for learning embeddings

        :param number_of_trials: (int): Number of trials for optuna
        :return: (dict[str,float,int,float]): Learned parameters: size of hidden layer, dropout, number of layers for the model, learning rate
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=number_of_trials)
        trial = study.best_trial
        return trial.params

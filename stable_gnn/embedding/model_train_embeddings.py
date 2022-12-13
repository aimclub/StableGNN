from typing import Any, Dict

import optuna
import torch
import torch_geometric.transforms as T
from torch import device
from torch_geometric.loader import NeighborSampler
from torch_geometric.typing import Tensor

from stable_gnn.embedding.model import Net
from stable_gnn.graph import Graph


class ModelTrainEmbeddings:
    """
    Model for training Net, wcich building embeddings for Geom-GCN layer

    :param name: (str): Name of input Graph
    :param loss_function: (dict): Dict of parameters of unsupervised loss function
    :param conv: (str): Name of convolution (default:'GCN')
    :param device: (device): Either 'cuda' or 'cpu' (default:'cuda')
    """

    def __init__(self, name: str, loss_function: Dict, conv: str = "GCN", device: device = "cuda") -> None:
        data = Graph(
            name,
            root="./data_validation/" + str(name),
            transform=T.NormalizeFeatures(),
            adjust_flag=False,
        )[0]
        self.Conv = conv
        self.device = device
        self.x = data.x
        self.y = data.y.squeeze()
        self.data = data.to(device)
        self.train_mask = torch.Tensor([True] * data.num_nodes)
        self.loss = loss_function
        self.dataset_name = name
        self.flag = self.loss["flag_tosave"]
        self.help_data = "stableGNN/data_help/"
        super(ModelTrainEmbeddings, self).__init__()

    def _sampling(self, sampler, epoch, nodes, loss):
        if epoch == 0:
            self.samples = sampler.sample(nodes)

    def _train(self, model, data, optimizer, sampler, train_loader, dropout, epoch, loss):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        if model.conv == "GCN":
            out = model.inference(data.to(self.device), dp=dropout)
            loss = model.loss(out[self.train_mask], self.samples)
            total_loss += loss
        else:
            for batch_size, n_id, adjs in train_loader:
                if len(train_loader.sizes) == 1:
                    adjs = [adjs]
                adjs = [adj.to(self.device) for adj in adjs]
                out = model.forward(data.x[n_id.to(self.device)].to(self.device), adjs)
                self._sampling(sampler, epoch, n_id[:batch_size], loss)
                loss = model.loss(out, self.samples)
                total_loss += loss
        total_loss.backward()
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

        sampler = self.loss["Sampler"]

        loss_sampler = sampler(self.dataset_name, self.data, device=self.device, loss_info=self.loss)
        model = Net(
            dataset=self.data,
            conv=self.Conv,
            loss_function=self.loss,
            device=self.device,
            hidden_layer=hidden_layer,
            out_layer=2,
            num_layers=(size),
            dropout=dropout,
        )
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(99):
            print(epoch)
            loss, _ = self._train(
                model,
                self.data,
                optimizer,
                loss_sampler,
                train_loader,
                dropout,
                epoch,
                self.loss,
            )
        _, out = self._train(
            model,
            self.data,
            optimizer,
            loss_sampler,
            train_loader,
            dropout,
            epoch,
            self.loss,
        )

        return out


class OptunaTrainEmbeddings(ModelTrainEmbeddings):
    """
    Model for training Net, wcich building embeddings for Geom-GCN layer

    :param name: (str): Name of input Graph
    :param loss_function: (dict): Dict of parameters of unsupervised loss function
    :param conv: (str): Name of convolution (default:'GCN')
    :param device: (device): Either 'cuda' or 'cpu' (default:'cuda')
    """

    def _objective(self, trial):
        # Integer parameter
        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128, 256])
        out_layer = 2
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        size = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        Conv = self.Conv
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

        sampler = loss_to_train["Sampler"]
        model = Net(
            dataset=self.data,
            conv=Conv,
            loss_function=loss_to_train,
            device=self.device,
            hidden_layer=hidden_layer,
            out_layer=out_layer,
            num_layers=size,
            dropout=dropout,
        )
        train_loader = NeighborSampler(self.data.edge_index, batch_size=int(self.data.num_nodes), sizes=[-1] * size)

        loss_sampler = sampler(
            self.dataset_name,
            self.data,
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
                loss_to_train,
            )
        return loss

    def run(self, number_of_trials: int) -> Dict[Any, Any]:
        """
        Tuning parameters for learning embeddings

        :param number_of_trials: (int): Number of trials for optuna
        :return: (dict[str,float,int,float]): Learned parameters: size of hidden layer, dropout, number of layers for the model, learning rate
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=number_of_trials)
        trial = study.best_trial
        return trial.params

import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda import device
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Tensor

from stable_gnn.graph import Graph
from stable_gnn.model_gc import ModelName as Model_GC
from stable_gnn.model_nc import ModelName as Model_NC


class TrainModel(ABC):
    """
    Base class for Training pipeline

    :param data: (Graph): Input graph data
    :param dataset_name: (str): Name of the dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param ssl_flag: (bool): If True, self supervised loss will be used along with semi-supervised
    """

    def __init__(
        self,
        data: Graph,
        dataset_name: str,
        device: device = "cuda",
        ssl_flag: bool = False,
    ) -> None:
        self.data = data
        self.device = device
        self.ssl_flag = ssl_flag
        self.data_name = dataset_name

        super(TrainModel, self).__init__()

    @abstractmethod
    def train(self, model: Module, optimizer: Optimizer) -> None:
        """
        Train model with optimizer

        :param model (torch.nn.Module): Model to train
        :param optimizer (torch.optim.Optimizer): Optimizer used for training
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def test(self, model: Module):
        """
        Test trained model on the test data

        :param model: (torch.nn.Module): Model to test
        """
        raise NotImplementedError

    def _train_test_split(self, N):
        indices = list(range(N))
        random.seed(0)
        train_indices = random.sample(indices, int(len(indices) * 0.7))
        left_indices = list(set(indices) - set(train_indices))
        random.seed(1)
        val_indices = random.sample(left_indices, int(len(indices) * 0.1))
        test_indices = list(set(left_indices) - set(val_indices))

        train_indices = torch.tensor(train_indices)
        val_indices = torch.tensor(val_indices)
        test_indices = torch.tensor(test_indices)

        train_mask = torch.tensor([False] * N)
        train_mask[train_indices] = True

        test_mask = torch.tensor([False] * N)
        test_mask[test_indices] = True

        val_mask = torch.tensor([False] * N)
        val_mask[val_indices] = True
        return train_indices, val_indices, test_indices, train_mask, val_mask, test_mask

    @abstractmethod
    def run(self, params: Dict):
        """
        Run the training process

        :param params: (Dict): Dictionary of input parameters for the model
        """
        raise NotImplementedError


class TrainModelGC(TrainModel):
    """
    Training pipeline for Graph Classification task

    :param data: (Graph): Input graph data
    :param dataset_name: (str): Name of the dataset
    :param conv: (str): Name of the convolution
    :param device: (device): Either 'cuda' or 'cpu'
    :param ssl_flag: (bool): If True, self supervised loss will be used along with semi-supervised
    :param extrapolate_flag: (bool): If True, extrapolation technique will be used
    """

    def __init__(
        self,
        data: Graph,
        dataset_name: str,
        conv: str = "GAT",
        device: device = "cuda",
        ssl_flag: bool = False,
        extrapolate_flag: bool = True,
    ) -> None:
        TrainModel.__init__(self, data, dataset_name, device, ssl_flag)

        self.Conv = conv
        self.extrapolate_flag = extrapolate_flag
        self.Model = Model_GC

        N = len(data)

        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
            self.train_mask,
            self.val_mask,
            self.test_mask,
        ) = self._train_test_split(N)

    def train(self, model: Module, optimizer: Optimizer, train_loader: DataLoader) -> float:
        """
        Train model with optimizer

        :param model (torch.nn.Module): Model to train
        :param optimizer (torch.optim.Optimizer): Optimizer used for training
        :param train_loader: (torch_geometric.loader.DataLoader): Data loader for input data
        :return: (float): Value of the loss function on the last epoch
        """
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        for dat in train_loader:
            dat = dat.to(self.device)
            batch_edge_list = dat.edge_index
            batch_x = dat.x
            batch = dat.batch
            y = dat.y

            out, deg_pred = model.forward(batch_x, batch_edge_list, batch)

            loss = model.loss_sup(out, y)
            total_loss += loss
            if self.ssl_flag:
                loss_SSL = model.SelfSupervisedLoss(deg_pred, dat)
                total_loss += loss_SSL
                loss_SSL.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def test(self, model: Module, loader: DataLoader) -> Tuple[float, float]:
        """
        Test trained model on the test data, the quality is measured with f1-score

        :param model: (torch.nn.Module): Model to test
        :param loader: (torch_geometric.loader.DataLoader): Loader for testing data
        :return: (float, float): Micro and Macro averaged f1 score for test data
        """
        model.eval()
        accs_micro = []
        accs_macro = []
        for dat in loader:
            dat = dat.to(self.device)
            out, _ = model.forward(dat.x, dat.edge_index, dat.batch)
            y_pred = out.cpu().argmax(dim=1, keepdim=True)
            y_true = dat.y
            accs_micro.append(accuracy_score(y_true.cpu().tolist(), y_pred.squeeze().tolist()))
            accs_macro.append(f1_score(y_true.cpu().tolist(), y_pred.squeeze().tolist(), average="macro"))
        return np.mean(accs_micro), np.mean(accs_macro)

    def run(self, params: Dict) -> Tuple[Module, float, float]:
        """
        Run the training process for Graph Classification task

        :param params: (dict): Dictionary of input parameters to the model: size of hidden layer, dropout, number of layers in the model and learning rate
        :return: (torch.nn.Module, float, float): Trained Model, Micro and macro averaged f1-scores for the test data
        """
        hidden_layer = params["hidden_layer"]
        dropout = params["dropout"]
        size = params["size of network, number of convs"]
        learning_rate = params["lr"]

        model = self.Model(
            dataset=self.data,
            conv=self.Conv,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,
            ssl_flag=self.ssl_flag,
        )

        model.to(self.device)

        if self.extrapolate_flag:
            init_edges = False
            remove_init_edges = False
            white_list = False
            score_func = "MI"  # TODO придумать как их задавать пользователю
            (self.train_dataset, self.test_dataset, self.val_dataset,) = model.Extrapolate(
                self.train_indices,
                self.val_indices,
                init_edges,
                remove_init_edges,
                white_list,
                score_func,
            )

        else:
            self.train_dataset, self.test_dataset, _, _ = model.convert_dataset(
                data=self.data,
                train_indices=self.train_indices,
                val_indices=self.val_indices,
            )
        train_loader = DataLoader(self.train_dataset, batch_size=20, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=20, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        losses = []

        for epoch in range(100):

            loss = self.train(model, optimizer, train_loader)
            losses.append(loss.detach().cpu())
            print(float(loss.detach().cpu()))

        test_acc_mi, test_acc_ma = self.test(model, loader=test_loader)
        train_acc_mi, train_acc_ma = self.test(model, loader=train_loader)

        print(
            "Loss: {:.4f}, Epoch: {:03d}, train acc micro: {:.4f}, train acc macro: {:.4f}, test acc micro: {:.4f}, test acc macro: {:.4f}".format(
                loss, epoch, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma
            )
        )

        return model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma


class TrainModelOptunaGC(TrainModelGC):
    """Class for optimizing hyperparameters of training pipeline for Node Classification task"""

    def _objective(self, trial):
        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        size = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        Conv = self.Conv
        learning_rate = trial.suggest_float("lr", 5e-3, 1e-2)

        model = self.Model(
            dataset=self.data,
            conv=Conv,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,
            ssl_flag=self.ssl_flag,
        )

        if self.extrapolate_flag:
            init_edges = False
            remove_init_edges = False
            white_list = False
            score_func = "MI"  # TODO придумать как их задавать пользователю
            (self.train_dataset, self.test_dataset, self.val_dataset,) = model.Extrapolate(
                self.train_indices,
                self.val_indices,
                init_edges,
                remove_init_edges,
                white_list,
                score_func,
            )
        else:
            self.train_dataset, _, self.val_dataset, _ = model.convert_dataset(
                self.data,
                train_indices=self.train_indices,
                val_indices=self.val_indices,
            )

        train_loader = DataLoader(self.train_dataset, batch_size=20, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=20, shuffle=True)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(50):
            _ = self.train(model, optimizer, train_loader)
        val_acc_mi, val_acc_ma = self.test(model, loader=val_loader)
        return np.sqrt(val_acc_mi * val_acc_ma)

    def run(self, number_of_trials: int) -> Dict:
        """
        Optimize hyperparameters for graph classification task training pipeline

        :param number_of_trials: (int): Number of trials for Optuna
        :return: (dict): Dictionary of input parameters for Model: size of hidden layer, dropout, number of layers in the model and learning rate
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=number_of_trials)
        trial = study.best_trial
        return trial.params


class TrainModelNC(TrainModel):
    """
    Training pipeline for Node Classification task

    :param data: (Graph): Input graph data
    :param dataset_name: (str): Name of the input dataset
    :param device: (device): Either 'cuda' or 'cpu'
    :param ssl_flag: (bool): If True, self supervised loss function is optimized along with semi-supervised loss
    :param loss_name: (str): Name of the loss for embedding learning in GeomGCN layer
    """

    def __init__(
        self, data: Graph, dataset_name: str, device: device = "cuda", ssl_flag: bool = False, loss_name: str = "APP"
    ):
        TrainModel.__init__(self, data, dataset_name, device, ssl_flag)
        self.Model = Model_NC
        self.loss_name = loss_name
        self.y = self.data.y.squeeze()
        N = len(self.data.x)

        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
            self.train_mask,
            self.val_mask,
            self.test_mask,
        ) = self._train_test_split(N)
        super(TrainModel, self).__init__()

    def train(self, model: Module, optimizer: Optimizer) -> float:
        """
        Train input model with optimizer

        :param model: (torch.nn.Module): Model to train
        :param optimizer: (torch.optim.Optimizer): Optimizer for training the model
        :return: (float): Loss function of the training data on the last epoch
        """
        model.train()
        optimizer.zero_grad()
        total_loss = 0

        out, deg_pred = model.inference(self.data.to(self.device))

        y = self.y.type(torch.LongTensor)
        y = y.to(self.device)
        if self.ssl_flag:
            loss_SSL = model.SelfSupervisedLoss(deg_pred, self.train_mask)
            total_loss += loss_SSL
            loss_SSL.backward(retain_graph=True)
        loss = model.loss_sup(out[self.train_mask], y[self.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    @torch.no_grad()
    def test(self, model: Module, mask: Tensor) -> Tuple[float, float]:
        """
        Test trained model, the quality is measured with f1 score

        :param model: (torch.nn.Module): Model to test
        :param mask: (Tensor): Tensor with bool values, masking testing data
        :return: (float,float): Micro and Macro averaged f1 scores for test data
        """
        model.eval()
        out, _ = model.inference(self.data.to(self.device))
        y_pred = out.cpu().argmax(dim=-1, keepdim=True)

        accs_micro = accuracy_score(self.y.detach()[mask].cpu().numpy(), y_pred[mask])
        accs_macro = f1_score(self.y.detach()[mask].cpu().numpy(), y_pred[mask], average="macro")

        return accs_micro, accs_macro

    def run(self, params: Dict) -> Tuple[Module, float, float, float]:
        """
        Run the training pipeline for node classification task

        :param params: (dict): Dictionary of input parameters to the model: size of hidden layer, dropout, number of layers in the model and learning rate
        :return: (torch.nn.Module, float, float): Trained Model, Micro and macro averaged f1-scores for the test data
        """
        hidden_layer = params["hidden_layer"]
        dropout = params["dropout"]
        size = params["size of network, number of convs"]
        learning_rate = params["lr"]

        model = self.Model(
            dataset=self.data,
            data_name=self.data_name,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,
            ssl_flag=self.ssl_flag,
            loss_name=self.loss_name,
        )

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        losses = []
        train_accs_mi = []
        train_accs_ma = []
        log = "Loss: {:.4f}, Epoch: {:03d}, Train acc micro: {:.4f}, Train acc macro: {:.4f}"

        for epoch in range(100):
            loss = self.train(model, optimizer)
            losses.append(loss.detach().cpu())
            train_acc_mi, train_acc_ma = self.test(model, mask=self.train_mask)
            train_accs_mi.append(train_acc_mi)
            train_accs_ma.append(train_acc_ma)
            print(log.format(loss, epoch, train_acc_mi, train_acc_ma))

        test_acc_mi, test_acc_ma = self.test(model, mask=self.test_mask)

        print(
            "Loss: {:.4f}, Epoch: {:03d}, test acc micro: {:.4f}, test acc macro: {:.4f}".format(
                loss, epoch, test_acc_mi, test_acc_ma
            )
        )
        plt.plot(losses)
        plt.title(" loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
        plt.plot(train_accs_mi)
        plt.title(" train f1 micro")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

        plt.plot(train_accs_ma)
        plt.title(" train f1 macro")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
        return model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma


class TrainModelOptunaNC(TrainModelNC):
    """Class for optimizing hyperparameters of training pipeline for Node Classification task"""

    def _objective(self, trial):
        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        size = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        learning_rate = trial.suggest_float("lr", 5e-3, 1e-2)

        model = self.Model(
            dataset=self.data,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,
            ssl_flag=self.ssl_flag,
            data_name=self.data_name,
            loss_name=self.loss_name,
        )

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(50):
            _ = self.train(model, optimizer)
        val_acc_mi, val_acc_ma = self.test(model, mask=self.val_mask)

        return val_acc_mi

    def run(self, number_of_trials: int) -> Dict:
        """
        Optimize hyperparameters for graph classification task training pipeline

        :param number_of_trials: (int): Number of trials for Optuna
        :return: (dict): Dictionary of input parameters for Model: size of hidden layer, dropout, number of layers in the model and learning rate
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=number_of_trials)
        trial = study.best_trial
        return trial.params

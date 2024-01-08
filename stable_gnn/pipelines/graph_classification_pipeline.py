from typing import Any, Dict, Tuple

import numpy as np
import optuna
import torch
from optuna import Trial
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda import device
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Tensor

from stable_gnn.extrapolate import Extrapolate
from stable_gnn.graph import Graph
from stable_gnn.model_gc import ModelGraphClassification as Model_GC
from stable_gnn.pipelines.abstract_pipeline import TrainModel


class TrainModelGC(TrainModel):
    """
    Training pipelines for Graph Classification task

    :param data: (Graph): Input graph data
    :param conv: (str): Name of the convolution
    :param device: (device): Either 'cuda' or 'cpu'
    :param ssl_flag: (bool): If True, self supervised loss will be used along with semi-supervised
    :param extrapolate_flag: (bool): If True, extrapolation technique will be used
    """

    def __init__(
        self,
        data: Graph,
        device: device,
        conv: str = "GAT",
        ssl_flag: bool = False,
        extrapolate_flag: bool = True,
    ) -> None:
        super().__init__(device=device, ssl_flag=ssl_flag)

        self.data = data
        self.conv = conv
        self.extrapolate_flag = extrapolate_flag
        self.model = Model_GC

        N = len(data)

        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
            self.train_mask,
            self.val_mask,
            self.test_mask,
        ) = self._train_test_split(N)

    def train(self, model: Module, optimizer: Optimizer, loader: DataLoader, coef: int) -> Tensor:  # type: ignore
        """
        Train model with optimizer

        :param model: (torch.nn.Module) Model to train
        :param optimizer: (torch.optim.Optimizer) Optimizer used for training
        :param loader: (torch_geometric.loader.DataLoader): Data loader for input data
        :return: (float): Value of the loss function on the last epoch
        """
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        total_loss_ssl = 0
        len_loader = 0
        for dat in loader:
            len_loader += 1
            dat = dat.to(self.device)
            batch_edge_list = dat.edge_index
            batch_x = dat.x
            batch = dat.batch
            y = dat.y

            out, deg_pred = model.forward(batch_x, batch_edge_list, batch)

            loss = model.loss_sup(out, y)
            total_loss += loss

            if self.ssl_flag:
                loss_SSL = model.self_supervised_loss(deg_pred, dat)
                loss += coef * loss_SSL
                total_loss_ssl += coef * loss_SSL
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        return total_loss / len_loader, total_loss_ssl / len_loader

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

            y_true_list = y_true.cpu().tolist()
            y_pred_list = y_pred.squeeze().tolist()
            if type(y_pred_list) != list:
                y_pred_list = [y_pred_list]
            accs_micro.append(f1_score(y_true_list, y_pred_list, average="micro"))
            accs_macro.append(f1_score(y_true_list, y_pred_list, average="macro"))
        return float(np.mean(accs_micro)), float(np.mean(accs_macro))

    def run(
        self, params: Dict[Any, Any], plot_training_procces: bool = False
    ) -> Tuple[Module, float, float, float, float]:
        """
        Run the training process for Graph Classification task

        :param params: (dict): Dictionary of input parameters to the model: size of hidden layer, dropout, number of layers in the model and learning rate
        :return: (torch.nn.Module, float, float): Trained Model, Micro and macro averaged f1-scores for the test data
        """
        hidden_layer = params["hidden_layer"]
        dropout = params["dropout"]
        size = params["size of network, number of convs"]
        learning_rate = params["lr"]
        coef = params["coef"]

        model = self.model(
            dataset=self.data,
            conv=self.conv,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,
            ssl_flag=self.ssl_flag,
        )

        model.to(self.device)

        if self.extrapolate_flag:
            Extrapolation = Extrapolate(model=model, dataset=self.data)
            init_edges = False
            remove_init_edges = False
            white_list = False
            score_func = "MI"
            (
                self.train_dataset,
                self.test_dataset,
                self.val_dataset,
            ) = Extrapolation(
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
        if plot_training_procces:
            losses = []
            losses_sl = []
            train_accs_mi = []
            test_accs_mi = []
        for epoch in range(100):
            print(epoch)
            loss, loss_sl = self.train(model, optimizer, train_loader, coef)
            if plot_training_procces:
                losses.append(loss.detach().cpu())
                losses_sl.append(loss_sl)
                test_acc_mi, test_acc_ma = self.test(model, loader=test_loader)
                train_acc_mi, train_acc_ma = self.test(model, loader=train_loader)
                train_accs_mi.append(train_acc_mi)
                test_accs_mi.append(test_acc_mi)
        test_acc_mi, test_acc_ma = self.test(model, loader=test_loader)
        train_acc_mi, train_acc_ma = self.test(model, loader=train_loader)

        if plot_training_procces:
            self.plot(losses=losses, losses_sl=losses_sl, train_accs_mi=train_accs_mi, test_accs_mi=test_accs_mi)

        return model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma


class TrainModelOptunaGC(TrainModelGC):
    """Class for optimizing hyperparameters of training pipelines for Node Classification task"""

    def _objective(self, trial: Trial) -> float:
        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        size = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        conv = self.conv
        learning_rate = trial.suggest_float("lr", 5e-3, 1e-2)
        coef = trial.suggest_categorical("coef", [1, 2, 5, 10, 20])

        model = self.model(
            dataset=self.data,
            conv=conv,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,
            ssl_flag=self.ssl_flag,
        )

        if self.extrapolate_flag:
            Extrapolation = Extrapolate(model=model, dataset=self.data)
            init_edges = False
            remove_init_edges = False
            white_list = False
            score_func = "MI"  # TODO придумать как их задавать пользователю
            (
                self.train_dataset,
                self.test_dataset,
                self.val_dataset,
            ) = Extrapolation(
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
            _ = self.train(model, optimizer, train_loader, coef)
        val_acc_mi, val_acc_ma = self.test(model, loader=val_loader)
        return np.sqrt(val_acc_mi * val_acc_ma)

    # TODO нужно поправить сигнатуру наследуемого класса, для семейства классов оптюн, возможно нужны вообще другие методы
    def run(self, number_of_trials: int) -> Dict[Any, Any]:  # type: ignore
        """
        Optimize hyperparameters for graph classification task training pipelines

        :param number_of_trials: (int): Number of trials for Optuna
        :return: (dict): Dictionary of input parameters for Model: size of hidden layer, dropout, number of layers in the model and learning rate
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=number_of_trials)
        trial = study.best_trial
        return trial.params

from typing import Any, Dict, Tuple

import optuna
import torch
from optuna import Trial
from sklearn.metrics import accuracy_score, f1_score
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.typing import Tensor

from stable_gnn.graph import Graph
from stable_gnn.model_nc import ModelNodeClassification as Model_NC
from stable_gnn.pipelines.abstract_pipeline import TrainModel


class TrainModelNC(TrainModel):
    """
    Training pipelines for Node Classification task

    :param data: (Graph): Input graph data
    :param device: (device): Either 'cuda' or 'cpu'
    :param ssl_flag: (bool): If True, self supervised loss function is optimized along with semi-supervised loss
    :param loss_name: (str): Name of the loss for embedding learning in GeomGCN layer
    """

    def __init__(
        self, data: Graph, device: device, ssl_flag: bool = False, emb_conv: str = "SAGE", loss_name: str = "APP"
    ) -> None:
        TrainModel.__init__(self, device, ssl_flag)
        self.model = Model_NC
        self.emb_conv = emb_conv
        self.loss_name = loss_name
        self.data = data
        self.y = self.data[0].y.squeeze()
        N = len(self.data[0].x)

        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
            self.train_mask,
            self.val_mask,
            self.test_mask,
        ) = self._train_test_split(N)

    def train(self, model: Module, optimizer: Optimizer, coef: int) -> Tensor:
        """Train input model with optimizer

        :param model: (torch.nn.Module): Model to train
        :param optimizer: (torch.optim.Optimizer): Optimizer for training the model
        :return: (float): Loss function of the training data on the last epoch
        """
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        out, deg_pred = model.inference(self.data[0].to(self.device))

        y = self.y.type(torch.LongTensor)
        y = y.to(self.device)
        loss_ssl = torch.tensor(0)
        if self.ssl_flag:
            loss_ssl = model.self_supervised_loss(deg_pred[self.train_mask], self.train_mask)
            total_loss = loss_ssl * coef
        loss = model.loss_sup(out[self.train_mask], y[self.train_mask])
        total_loss += loss
        total_loss.backward()  # type: ignore
        optimizer.step()
        optimizer.zero_grad()
        return loss, loss_ssl

    @torch.no_grad()
    def test(self, model: Module, mask: Tensor) -> Tuple[float, float]:
        """
        Test trained model, the quality is measured with f1 score

        :param model: (torch.nn.Module): Model to test
        :param mask: (Tensor): Tensor with bool values, masking testing data
        :return: (float,float): Micro and Macro averaged f1 scores for test data
        """
        model.eval()
        out, _ = model.inference(self.data[0].to(self.device))
        y_pred = out.cpu().argmax(dim=-1, keepdim=True)

        accs_micro = accuracy_score(self.y.detach()[mask].cpu().numpy(), y_pred[mask])
        accs_macro = f1_score(self.y.detach()[mask].cpu().numpy(), y_pred[mask], average="macro")

        return accs_micro, accs_macro

    def run(
        self, params: Dict[Any, Any], plot_training_procces: bool = False
    ) -> Tuple[Module, float, float, float, float]:
        """
        Run the training pipelines for node classification task

        :param params: (dict): Dictionary of input parameters to the model: size of hidden layer, dropout, number of layers in the model and learning rate
        :return: (torch.nn.Module, float, float, float, float): Trained Model, Micro and macro averaged f1-scores for the test data
        """
        hidden_layer = params["hidden_layer"]
        dropout = params["dropout"]
        size = params["size of network, number of convs"]
        learning_rate = params["lr"]
        coef = params["coef"]

        model = self.model(
            dataset=self.data,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,
            ssl_flag=self.ssl_flag,
            loss_name=self.loss_name,
        )

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        if plot_training_procces:
            losses = []
            losses_ssl = []
            train_accs_mi = []
            test_accs_mi = []

        for epoch in range(100):
            loss, loss_ssl = self.train(model, optimizer, coef)
            if plot_training_procces:
                losses.append(loss.detach().cpu())
                losses_ssl.append(loss_ssl.detach().cpu())
                train_acc_mi, train_acc_ma = self.test(model, mask=self.train_mask)
                test_acc_mi, _ = self.test(model, mask=self.test_mask)
                train_accs_mi.append(train_acc_mi)
                test_accs_mi.append(test_acc_mi)
            # print(log.format(loss, epoch, train_acc_mi, train_acc_ma))
        train_acc_mi, train_acc_ma = self.test(model, mask=self.train_mask)
        test_acc_mi, test_acc_ma = self.test(model, mask=self.test_mask)

        if plot_training_procces:
            self.plot(losses, losses_ssl, train_accs_mi, test_accs_mi)

        return model, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma


class TrainModelOptunaNC(TrainModelNC):
    """Class for optimizing hyperparameters of training pipelines for Node Classification task"""

    def _objective(self, trial: Trial) -> float:
        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        num_layers = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        learning_rate = trial.suggest_float("lr", 5e-3, 1e-2)
        coef = trial.suggest_categorical("coef", [0, 2, 5, 10, 20])
        model = self.model(
            dataset=self.data,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=num_layers,
            dropout=dropout,
            ssl_flag=self.ssl_flag,
            loss_name=self.loss_name,
        )

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(50):
            _ = self.train(model, optimizer, coef)
        val_acc_mi, val_acc_ma = self.test(model, mask=self.val_mask)

        return val_acc_mi

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

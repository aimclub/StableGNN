import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.cuda import device
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Tensor


class TrainModel(ABC):
    """
    Base class for Training pipelines

    :param device: (device): Either 'cuda' or 'cpu'
    :param ssl_flag: (bool): If True, self supervised loss will be used along with semi-supervised
    """

    def __init__(
        self,
        device: device,
        ssl_flag: bool = False,
    ) -> None:
        self.device = device
        self.ssl_flag = ssl_flag

    @abstractmethod
    def train(
        self, model: Module, optimizer: Optimizer, loader: DataLoader
    ) -> Tuple[Module, float, float, float, float]:
        """
        Train model with optimizer

        :param model: (torch.nn.Module) Model to train
        :param optimizer: (torch.optim.Optimizer) Optimizer used for training
        :param loader: (torch_geometric.loader.DataLoader): Data loader for input data
        :returns: (torch.nn.Module, float, float, float, float): Trained Model, Micro and macro averaged f1-scores for the train data
        """
        raise NotImplementedError("implement train function")

    @abstractmethod
    @torch.no_grad()
    def test(
        self,
        model: Module,
        loader: DataLoader,
    ) -> Tuple[Module, float, float, float, float]:
        """
        Test trained model on the test data

        :param model: (torch.nn.Module): Model to test
        :param loader: (torch_geometric.loader.DataLoader): Data loader for input data
        :returns: (torch.nn.Module, float, float, float, float): Trained Model, Micro and macro averaged f1-scores for the test data
        """
        raise NotImplementedError("implement test function")

    @staticmethod
    def _train_test_split(n: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        indices = list(range(n))
        train_indices_sample = random.sample(indices, int(len(indices) * 0.7))
        left_indices = list(set(indices) - set(train_indices_sample))

        val_indices_sample = random.sample(left_indices, int(len(indices) * 0.1))
        test_indices_sample = list(set(left_indices) - set(val_indices_sample))

        train_indices = torch.tensor(train_indices_sample)
        val_indices = torch.tensor(val_indices_sample)
        test_indices = torch.tensor(test_indices_sample)

        train_mask = torch.tensor([False] * n)
        train_mask[train_indices] = True

        test_mask = torch.tensor([False] * n)
        test_mask[test_indices] = True

        val_mask = torch.tensor([False] * n)
        val_mask[val_indices] = True
        return train_indices, val_indices, test_indices, train_mask, val_mask, test_mask

    @staticmethod
    def plot(
        losses: List[float], losses_sl: List[float], train_accs_mi: List[float], test_accs_mi: List[float]
    ) -> None:
        """
        Plot training process

        :param losses: List of losses during training
        :param losses_sl: List of self-supervised losses during training
        :param train_accs_mi: List of accuracy on training data during training
        :param test_accs_mi:  List of accuracy on testing data during training
        """
        plt.plot(losses)
        plt.title("semi-supervised loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
        if sum(losses_sl) > 0:
            plt.plot(losses)
            plt.title("self-supervised loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.show()
        plt.plot(train_accs_mi)
        plt.title("micro-averaged f1 score on train data")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

        plt.plot(test_accs_mi)
        plt.title("micro-averaged f1 score on test data")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

    @abstractmethod
    def run(
        self, params: Dict[Any, Any], plot_training_process: bool = False
    ) -> Tuple[Module, float, float, float, float]:
        """
        Run the training process

        :param params: (Dict): Dictionary of input parameters for the model
        :returns: (torch.nn.Module, float, float, float, float): Trained Model, Micro and macro averaged f1-scores for the test data
        """
        raise NotImplementedError("implement run function")

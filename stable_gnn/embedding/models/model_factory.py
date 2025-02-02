from typing import Any, Dict

from torch import device

from stable_gnn.embedding.models.abstract_model import BaseNet
from stable_gnn.embedding.models.convs_factory import ConvolutionsFactory
from stable_gnn.embedding.unsupervized_loss_models.context_matrix import ContextMatrixModel
from stable_gnn.embedding.unsupervized_loss_models.factorization import FactorizationModel
from stable_gnn.embedding.unsupervized_loss_models.force_to_vec import Force2VecModel
from stable_gnn.embedding.unsupervized_loss_models.laplacian_eigen_maps import LalplacianEigenMapsModel
from stable_gnn.embedding.unsupervized_loss_models.random_walk import RandomWalkBasedModel


class ModelFactory:
    """Factory responsible for flexible model creation based on user input."""

    def __init__(self) -> None:
        """Initialize the ModelFactory instance."""
        self.conv_factory = ConvolutionsFactory()

    def build_model(
        self,
        conv: str,
        loss_function: Dict[Any, Any],
        device: device,
        num_features: int,
        hidden_layer: int,
        out_layer: int,
        num_layers: int,
        dropout: float,
        heads: int = 0,
    ) -> BaseNet:
        """Build model based on input.

        :param device: (device): Either 'cuda' or 'cpu'.
        :param hidden_layer: (int): The size of hidden layer (default: 64).
        :param out_layer: (int): The size of output layer (default: 128).
        :param dropout: (float): Dropout (default: 0.0).
        :param num_layers: (int): Number of layers in the model (default: 2).
        :param heads: (int): Number of heads in GAT conv (default: 1).
        :param conv: (str): Either 'GCN', 'GAT' or 'SAGE' convolution.
        :returns: Model.
        """
        if loss_function["loss var"] == "Random Walks":
            model = RandomWalkBasedModel(
                device=device,
                num_layers=num_layers,
                num_featurs=num_features,
                dropout=dropout,
                out_layer=out_layer,
            )
        elif loss_function["loss var"] == "Context Matrix":
            model = ContextMatrixModel(
                name=loss_function["Name"],
                device=device,
                num_layers=num_layers,
                num_featurs=num_features,
                dropout=dropout,
                out_layer=out_layer,
            )
        elif loss_function["loss var"] == "Factorization":
            model = FactorizationModel(
                lmbda=loss_function["lmbda"],
                device=device,
                num_layers=num_layers,
                num_featurs=num_features,
                dropout=dropout,
                out_layer=out_layer,
            )
        elif loss_function["loss var"] == "Laplacian EigenMaps":
            model = LalplacianEigenMapsModel(
                device=device,
                num_layers=num_layers,
                num_featurs=num_features,
                dropout=dropout,
                out_layer=out_layer,
            )
        elif loss_function["loss var"] == "Force2Vec":
            model = Force2VecModel(
                device=device,
                num_layers=num_layers,
                num_featurs=num_features,
                dropout=dropout,
                out_layer=out_layer,
            )
        else:
            raise ValueError(f"Loss is not supported: {loss_function['loss var']}")

        model.convs = self.conv_factory.build_convolutions(
            conv=conv,
            num_layers=num_layers,
            num_features=num_features,
            hidden_layer=hidden_layer,
            out_layer=out_layer,
            heads=heads,
        )
        return model

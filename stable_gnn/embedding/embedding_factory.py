from typing import Any, Dict, List

import torch
from numpy.typing import NDArray
from torch import device

from stable_gnn.embedding.model_train_embeddings import ModelTrainEmbeddings, OptunaTrainEmbeddings
from stable_gnn.embedding.sampling.samplers import SamplerAPP, SamplerContextMatrix, SamplerFactorization
from stable_gnn.graph import Graph


class EmbeddingFactory:
    """Producing unsupervised embeddings for a given dataset"""

    @staticmethod
    def _build_embeddings(
        loss: Dict[str, Any], data: Graph, conv: str, device: device, number_of_trials: int, tune_out: bool = False
    ) -> NDArray:
        optuna_training = OptunaTrainEmbeddings(
            data=data, conv=conv, device=device, loss_function=loss, tune_out=tune_out
        )
        best_values = optuna_training.run(number_of_trials=number_of_trials)

        loss_trgt = dict()
        for par in loss:
            loss_trgt[par] = loss[par]

        if "alpha" in loss_trgt:
            loss_trgt["alpha"] = best_values["alpha"]
        if "num_negative_samples" in loss_trgt:
            loss_trgt["num_negative_samples"] = best_values["num_negative_samples"]
        if "lmbda" in loss_trgt:
            loss_trgt["lmbda"] = best_values["lmbda"]

        model_training = ModelTrainEmbeddings(
            data=data, conv=conv, device=device, loss_function=loss_trgt, tune_out=tune_out
        )
        out = model_training.run(best_values)
        torch.cuda.empty_cache()
        return out.detach().cpu().numpy()

    @staticmethod
    def _get_emb_settings(loss_name: str) -> Dict[str, Any]:
        if loss_name == "APP":
            return {
                "Name": "APP",
                "C": "PPR",
                "num_negative_samples": [1, 6, 11, 16, 21],
                "loss var": "Context Matrix",
                "flat_tosave": False,
                "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "sampler": SamplerAPP,
            }  # APP
        elif loss_name == "LINE":
            return {
                "Name": "LINE",
                "C": "Adj",
                "num_negative_samples": [1, 6, 11, 16, 21],
                "loss var": "Context Matrix",
                "sampler": SamplerContextMatrix,
                "flat_tosave": False,
                "lmbda": [0.0, 1.0],
            }
        elif loss_name == "HOPE_AA":
            return {
                "Name": "HOPE_AA",
                "C": "AA",
                "loss var": "Factorization",
                "sampler": SamplerFactorization,
                "flat_tosave": False,
                "lmbda": [0.0, 1.0],
            }
        elif loss_name == "VERSE_Adj":
            return {
                "Name": "VERSE_Adj",
                "C": "Adj",
                "num_negative_samples": [1, 6, 11, 16, 21],
                "loss var": "Context Matrix",
                "flat_tosave": False,
                "sampler": SamplerContextMatrix,
                "lmbda": [0.0, 1.0],
            }
        else:
            raise NameError

    def build_embeddings(
        self,
        loss_name: str,
        conv: str,
        data: List[Graph],
        device: device,
        number_of_trials: int,
        tune_out: bool = False,
    ) -> NDArray:
        """Build embeddings based on passed dataset and settings

        :param loss_name: (str): Name of loss function for embedding learning in GeomGCN layer
        :param conv: (str) Name of convolution used in unsupervied embeddings
        :param data: (Graph): Input Graph
        :param device: (device): Device 'cuda' or 'cpu'
        :param number_of_trials (int): Number of trials for optuna tuning embeddings
        :param tune_out (bool): Flag if you want tune out layer of embeddings
        :returns: (NDArray) embeddings NumPy array of (N_nodes) x (N_emb_dim)
        """
        loss_params = self._get_emb_settings(loss_name)
        emb = self._build_embeddings(
            loss=loss_params,
            data=data[0],
            conv=conv,
            device=device,
            number_of_trials=number_of_trials,
            tune_out=tune_out,
        )
        return emb

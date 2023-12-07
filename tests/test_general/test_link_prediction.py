import pathlib

import torch

from stable_gnn.graph import Graph
from stable_gnn.pipelines.graph_classification_pipeline import TrainModelGC
from tests.data_generators import generate_gc_graphs

root = str(pathlib.Path(__file__).parent.resolve().joinpath("data_validation/")) + "/"
generate_gc_graphs(root, 30)

def test_linkpredict():
    pass
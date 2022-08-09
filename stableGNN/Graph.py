import torch
from torch_geometric.data import Data

class Graph(Data):
    def __init__(self):
        super().__init__()

    def adjust(self):
        pass
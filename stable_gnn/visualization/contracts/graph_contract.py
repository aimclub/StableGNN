from dataclasses import dataclass

from stable_gnn.visualization.contracts.base_graph_contract import BaseGraphContract


@dataclass
class GraphContract(BaseGraphContract):
    edge_num: int = 0

from dataclasses import dataclass

from stable_gnn.visualization.contracts.base_graph_contract import BaseGraphContract


@dataclass
class HypergraphContract(BaseGraphContract):
    vertex_weight: list[float] | None = None
    edge_num: int = 0

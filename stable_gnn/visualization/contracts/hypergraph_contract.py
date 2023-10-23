from dataclasses import dataclass

from stable_gnn.visualization.contracts.base_graph_contract import BaseGraphContract


@dataclass
class HypergraphContract(BaseGraphContract):
    edge_weights: float | list[float] | None = None
    vertex_weight: list[float] | None = None

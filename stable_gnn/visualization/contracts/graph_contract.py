from dataclasses import dataclass


@dataclass
class GraphContract:
    vertex_num: int
    edge_num: int
    edge_list: list[int] | list[list[int]] | None = None
    edge_weights: float | list[float] | None = None

from dataclasses import dataclass


@dataclass
class LayoutContract:
    vertex_num: int
    edge_list: list[tuple]
    push_vertex_strength: float
    push_edge_strength: float
    pull_edge_strength: float
    pull_center_strength: float

from dataclasses import dataclass


@dataclass
class LayoutContract:
    vertex_num: int
    edge_list: list[tuple] | list[list[int]]
    push_vertex_strength: float | None
    push_edge_strength: float | None
    pull_edge_strength: float | None
    pull_center_strength: float | None

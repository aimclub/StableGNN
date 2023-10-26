from dataclasses import dataclass

from stable_gnn.visualization.config.parameters.defaults import Defaults


@dataclass
class StrengthConstructorContract:
    push_vertex_strength: float = Defaults.push_vertex_strength_vis
    push_edge_strength: float = Defaults.push_edge_strength_vis
    pull_edge_strength: float = Defaults.pull_edge_strength_vis
    pull_center_strength: float = Defaults.pull_center_strength_vis

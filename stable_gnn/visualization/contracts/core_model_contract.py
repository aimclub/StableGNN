from dataclasses import dataclass

from stable_gnn.visualization.config.parameters.defaults import Defaults


@dataclass
class CoreModelContract:
    nums: int | list
    forces: dict
    centers: list
    damping_factor: float = Defaults.damping_factor

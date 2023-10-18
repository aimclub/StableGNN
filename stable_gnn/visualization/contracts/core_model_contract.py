from dataclasses import dataclass


@dataclass
class CoreModelContract:
    nums: int | list
    forces: dict
    centers: list
    damping_factor: float = 0.9999

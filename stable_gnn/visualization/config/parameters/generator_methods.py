from stable_gnn.visualization.utils.frozen_dataclass import reference
from stable_gnn.visualization.utils.reference_base import ReferenceBase


@reference
class GeneratorMethods(ReferenceBase):
    custom: str = "custom"
    uniform: str = "uniform"
    low_order_first: str = "low_order_first"
    high_order_first: str = "high_order_first"

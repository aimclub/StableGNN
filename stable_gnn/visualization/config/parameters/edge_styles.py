from stable_gnn.visualization.utils.frozen_dataclass import reference
from stable_gnn.visualization.utils.reference_base import ReferenceBase


@reference
class EdgeStyles(ReferenceBase):
    line: str = "line"
    circle: str = "circle"

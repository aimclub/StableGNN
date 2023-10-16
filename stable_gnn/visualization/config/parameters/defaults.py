from stable_gnn.visualization.config.parameters.colors import Colors
from stable_gnn.visualization.config.parameters.edge_styles import EdgeStyles
from stable_gnn.visualization.config.parameters.fonts import Fonts
from stable_gnn.visualization.utils.frozen_dataclass import reference
from stable_gnn.visualization.utils.reference_base import ReferenceBase


@reference
class Defaults(ReferenceBase):
    edge_style: str = EdgeStyles.line
    edge_color: str = Colors.gray
    edge_fill_color: str = Colors.whitesmoke
    edge_line_width: float = 1.0
    vertex_size: float = 1.0
    vertex_color: str = Colors.red
    vertex_line_width: float = 1.0
    font_size: float = 1.0
    font_family: str = Fonts.sans_serif
    push_vertex_strength: float = 1.0
    push_edge_strength: float = 1.0
    pull_edge_strength: float = 1.0
    pull_center_strength: float = 1.0
    # calculate_edge_line_width params
    edge_line_width_multiplier: float = 1.0
    edge_line_width_divider: float = 120.0
    # calculate_font_size params
    font_size_multiplier: float = 20
    font_size_divider: float = 100.0
    # calculate_vertex_line_width params
    vertex_line_width_multiplier: float = 1.0
    vertex_line_width_divider: float = 50.0
    # calculate_vertex_size params
    calculate_vertex_size_multiplier: float = 1.0
    calculate_vertex_size_divider: float = 10.0
    calculate_vertex_size_modifier: float = 0.1


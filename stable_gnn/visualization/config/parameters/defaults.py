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
    vertex_strength: float = 1.0
    font_size: float = 1.0
    font_family: str = Fonts.sans_serif
    push_vertex_strength_vis: float = 1.0
    push_edge_strength_vis: float = 1.0
    pull_edge_strength_vis: float = 1.0
    pull_center_strength_vis: float = 1.0
    damping_factor: float = 0.9999
    damping: float = 1
    radius_increment: float = 0.3
    force_modifier: float = -0.1
    force_a_max: float = 0.1
    axes_num: int = 2
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
    # calculate strength
    push_vertex_strength: float = 0.006
    push_edge_strength: float = 0.0
    pull_edge_strength: float = 0.045
    pull_center_strength: float = 0.01
    # calculate layout
    layout_scale_initial: int = 5
    vertex_coord_max: float = 5.0
    vertex_coord_min: float = -5.0
    vertex_coord_multiplier: float = 0.8
    vertex_coord_modifier: float = 0.1
    # figure
    figure_size: tuple = (6, 6)
    x_limits: tuple = (0, 1.0)
    y_limits: tuple = (0, 1.0)
    axes_on_off: str = "off"
    # core physical model
    node_attraction_key: int = 0
    node_repulsion_key: int = 1
    edge_repulsion_key: int = 2
    center_of_gravity_key: int = 3
    max_iterations: int = 400
    epsilon: float = 0.001
    delta: float = 2.0

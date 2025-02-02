from stable_gnn.visualization.contracts.style_constructor_contract import GraphStyleConstructorContract
from stable_gnn.visualization.utils.fill_color import fill_color


class GraphStyleConstructor:
    """Constructor (one action controller) for Graph style."""

    def __call__(self, contract: GraphStyleConstructorContract) -> tuple:
        vertex_color_color = contract.vertex_color
        edge_color_color = contract.edge_color
        edge_fill_color_fill_color = contract.edge_fill_color

        v_color = fill_color(contract.vertex_color, vertex_color_color, contract.vertex_num)
        e_color = fill_color(contract.edge_color, edge_color_color, contract.edges_num)
        e_fill_color = fill_color(contract.edge_fill_color, edge_fill_color_fill_color, contract.edges_num)

        return v_color, e_color, e_fill_color

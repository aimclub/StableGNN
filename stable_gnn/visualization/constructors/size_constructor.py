from stable_gnn.visualization.contracts.size_constructor_contract import SizeConstructorContract
from stable_gnn.visualization.equations.calc_edge_line_width import calculate_edge_line_width
from stable_gnn.visualization.equations.calc_font_size import calculate_font_size
from stable_gnn.visualization.equations.calc_vertex_line_width import calculate_vertex_line_width
from stable_gnn.visualization.equations.calc_vertex_size import calculate_vertex_size
from stable_gnn.visualization.utils.fill_sizes import fill_sizes


class SizeConstructor:
    """Constructor (one action controller) for Graph sizes."""

    def __call__(self, contract: SizeConstructorContract) -> tuple:
        edge_list_length = len(contract.edge_list)
        _vertex_size = calculate_vertex_size(contract.vertex_num)
        _vertex_line_width = calculate_vertex_line_width(contract.vertex_num)
        _edge_line_width = calculate_edge_line_width(edge_list_length)
        _font_size = calculate_font_size(contract.vertex_num)

        v_size = fill_sizes(contract.vertex_size, _vertex_size, contract.vertex_num)
        v_line_width = fill_sizes(contract.vertex_line_width, _vertex_line_width, contract.vertex_num)
        e_line_width = fill_sizes(contract.edge_line_width, _edge_line_width, edge_list_length)
        font_size = _font_size if contract.font_size is None else contract.font_size * _font_size

        return v_size, v_line_width, e_line_width, font_size

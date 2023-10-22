from stable_gnn.visualization.config.types import TEdgeList


def calc_arrow_head_width(edge_line_width: list, show_arrow: bool, edge_list: TEdgeList):
    return [0.015 * w for w in edge_line_width] \
            if show_arrow else [0] * len(edge_list)

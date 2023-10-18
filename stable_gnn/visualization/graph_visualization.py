from copy import deepcopy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from stable_gnn.visualization.config.parameters.edge_styles import EdgeStyles
from stable_gnn.visualization.constructors.layout_constructor import LayoutConstructor
from stable_gnn.visualization.constructors.size_constructor import SizeConstructor
from stable_gnn.visualization.constructors.strength_constructor import StrengthConstructor
from stable_gnn.visualization.constructors.style_constructor import StyleConstructor
from stable_gnn.visualization.contracts.draw_circle_edges_contract import DrawEdgesContract
from stable_gnn.visualization.contracts.draw_line_edges_contract import DrawLineEdgesContract
from stable_gnn.visualization.contracts.draw_vertex_contract import DrawVertexContract
from stable_gnn.visualization.contracts.graph_visualization_contract import GraphVisualizationContract
from stable_gnn.visualization.contracts.layout_contract import LayoutContract
from stable_gnn.visualization.contracts.size_constructor_contract import SizeConstructorContract
from stable_gnn.visualization.contracts.strength_constructor_contract import StrengthConstructorContract
from stable_gnn.visualization.contracts.style_constructor_contract import StyleConstructorContract
from stable_gnn.visualization.exceptions.exceptions_classes import ParamsValidationException
from stable_gnn.visualization.config.parameters.defaults import Defaults

from stable_gnn.graph import Graph


class GraphVisualizer:

    def __init__(self, contract: GraphVisualizationContract):
        self.contract = contract

        self._validate()

    def draw(self):
        fig, __axes = plt.subplots(figsize=Defaults.figure_size)
        __vertex_num, __edge_list = self.contract.graph.vertex_num, deepcopy(self.contract.graph.edges[0])

        default_style_contract: StyleConstructorContract = StyleConstructorContract(
            vertex_num=self.contract.graph.vertex_num,
            edges_num=self.contract.graph.edge_num,
            vertex_color=self.contract.vertex_color,
            edge_color=self.contract.edge_color,
            edge_fill_color=self.contract.edge_fill_color
        )
        default_style_constructor: StyleConstructor = StyleConstructor()

        __vertex_color, __edge_color, __edge_fill_color = default_style_constructor(default_style_contract)

        default_size_contract: SizeConstructorContract = SizeConstructorContract(
            vertex_num=__vertex_num,
            edges_list=__edge_list,
            vertex_size=self.contract.vertex_size,
            vertex_line_width=self.contract.vertex_line_width,
            edge_line_width=self.contract.edge_line_width,
            font_size=self.contract.font_size
        )
        default_size_constructor: SizeConstructor = SizeConstructor()

        __vertex_size, __vertex_line_width, __edge_line_width, __font_size = default_size_constructor(
            default_size_contract)

        default_strength_contract: StrengthConstructorContract = StrengthConstructorContract(
            self.contract.push_vertex_strength,
            self.contract.push_edge_strength,
            self.contract.pull_edge_strength,
            self.contract.pull_center_strength
        )

        default_strength_constructor: StrengthConstructor = StrengthConstructor()

        (
            __push_v_strength, __push_e_strength, __pull_e_strength,
            __pull_center_strength,
        ) = default_strength_constructor(default_strength_contract)

        layout_contract: LayoutContract = LayoutContract(
            vertex_num=__vertex_num,
            edge_list=__edge_list,
            push_vertex_strength=__push_v_strength,
            push_edge_strength=None,
            pull_edge_strength=__pull_e_strength,
            pull_center_strength=__pull_center_strength
        )

        layout_constructor: LayoutConstructor = LayoutConstructor()

        __vertex_coordinates = layout_constructor(layout_contract)

        if self.contract.edge_style == EdgeStyles.line:
            draw_line_edges_contract: DrawLineEdgesContract = DrawLineEdgesContract(
                axes=__axes,
                vertex_coordinates=__vertex_coordinates,
                vertex_size=__vertex_size,
                edge_list=__edge_list,
                show_arrow=False,
                edge_color=__edge_color,
                edge_line_width=__edge_line_width
            )
            self.__draw_line_edges(draw_line_edges_contract)
        elif self.contract.edge_style == EdgeStyles.circle:
            draw_edges_contract: DrawEdgesContract = DrawEdgesContract(
                axes=__axes,
                vertex_coordinates=__vertex_coordinates,
                vertex_size=__vertex_size,
                edge_list=__edge_list,
                edge_color=__edge_color,
                edge_fill_color=__edge_fill_color,
                edge_line_width=__edge_line_width
            )
            self.__draw_circle_edges(draw_edges_contract)
        else:
            raise ParamsValidationException

        draw_vertex_contract: DrawVertexContract = DrawVertexContract(
            axes=__axes,
            vertex_coordinates=__vertex_coordinates,
            vertex_label=self.contract.vertex_label,
            font_size=__font_size,
            font_family=self.contract.font_family,
            vertex_size=__vertex_size,
            vertex_color=__vertex_color,
            vertex_line_width=__vertex_line_width
        )
        self.__draw_vertex(draw_vertex_contract)

        plt.xlim(Defaults.x_limits)
        plt.ylim(Defaults.y_limits)
        plt.axis(Defaults.axes_on_off)
        fig.tight_layout()

    def __draw_line_edges(self, contract: DrawLineEdgesContract):
        pass

    def __draw_circle_edges(self, contract: DrawEdgesContract):
        pass

    def __draw_vertex(self, contract: DrawVertexContract):
        pass

    def _validate(self):
        graph_type_is_correct = isinstance(self.contract.graph, Graph)
        edge_style_are_valid = self.contract.edge_style in EdgeStyles.values
        node_quantity_is_positive = self.contract.graph.vertex_num > 0

        if not graph_type_is_correct or not edge_style_are_valid or not node_quantity_is_positive:
            raise ParamsValidationException

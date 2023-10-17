from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from stable_gnn.visualization.config.parameters.edge_styles import EdgeStyles
from stable_gnn.visualization.constructors.size_constructor import SizeConstructor
from stable_gnn.visualization.constructors.style_constructor import StyleConstructor
from stable_gnn.visualization.contracts.graph_visualization_contract import GraphVisualizationContract
from stable_gnn.visualization.contracts.size_constructor_contract import SizeConstructorContract
from stable_gnn.visualization.contracts.style_constructor_contract import StyleConstructorContract
from stable_gnn.visualization.exceptions.exceptions_classes import ParamsValidationException

from stable_gnn.graph import Graph


class GraphVisualizer:

    def __init__(self, contract: GraphVisualizationContract):
        self.contract = contract

        self._validate()

        default_style_contract: StyleConstructorContract = StyleConstructorContract(self.contract.graph.vertex_num,
                                                                                    self.contract.graph.edge_num,
                                                                                    self.contract.vertex_color,
                                                                                    self.contract.edge_color,
                                                                                    self.contract.edge_fill_color)
        default_style_constructor: StyleConstructor = StyleConstructor()

        v_color, e_color, e_fill_color = default_style_constructor(default_style_contract)

        default_size_contract: SizeConstructorContract = SizeConstructorContract(self.contract.graph.vertex_num,
                                                                                 self.contract.graph.edge_list,
                                                                                 self.contract.vertex_size,
                                                                                 self.contract.vertex_line_width,
                                                                                 self.contract.edge_line_width)
        default_size_constructor: SizeConstructor = SizeConstructor()

        v_size, v_line_width, e_line_width, font_size = default_size_constructor(default_size_contract)

    def draw(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        num_v, e_list = g.num_v, deepcopy(g.e[0])

    def _validate(self):
        graph_type_is_correct = isinstance(self.contract.graph, Graph)
        edge_style_are_valid = self.contract.edge_style in EdgeStyles.values
        node_quantity_is_positive = self.contract.graph.vertex_num > 0

        if not graph_type_is_correct or not edge_style_are_valid or not node_quantity_is_positive:
            raise ParamsValidationException

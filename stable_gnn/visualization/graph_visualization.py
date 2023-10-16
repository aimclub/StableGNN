from stable_gnn.visualization.config.parameters.edge_styles import EdgeStyles
from stable_gnn.visualization.contracts.graph_visualization_contract import GraphVisualizationContract
from stable_gnn.visualization.exceptions.exceptions_classes import ParamsValidationException

from stable_gnn.graph import Graph


class GraphVisualizer:

    def __init__(self, contract: GraphVisualizationContract):
        self.contract = contract
        self._validate()

    def draw(self):
        pass

    def _validate(self):
        graph_type_is_correct = isinstance(self.contract.graph, Graph)
        edge_style_are_valid = self.contract.edge_style in EdgeStyles.values
        node_quantity_is_positive = self.contract.graph.num_nodes > 0

        if not graph_type_is_correct or not edge_style_are_valid or not node_quantity_is_positive:
            raise ParamsValidationException

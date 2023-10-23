from stable_gnn.visualization.base_visualization import BaseVisualization


class GraphVisualizer(BaseVisualization):
    contract = None

    def __init__(self, contract: GraphVisualizationContract):
        self.contract = contract

        self.validate()


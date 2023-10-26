from dataclasses import dataclass

from stable_gnn.visualization.contracts.base_visualization_contract import BaseVisualizationContract
from stable_gnn.visualization.contracts.graph_contract import GraphContract


@dataclass
class GraphVisualizationContract(BaseVisualizationContract):
    graph: GraphContract = None

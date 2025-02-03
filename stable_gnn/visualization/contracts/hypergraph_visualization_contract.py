from dataclasses import dataclass

from stable_gnn.visualization.config.parameters.edge_styles import EdgeStyles
from stable_gnn.visualization.contracts.base_visualization_contract import BaseVisualizationContract
from stable_gnn.visualization.contracts.hypergraph_contract import HypergraphContract


@dataclass
class HypergraphVisualizationContract(BaseVisualizationContract):
    graph: HypergraphContract = None
    edge_style: str = EdgeStyles.circle

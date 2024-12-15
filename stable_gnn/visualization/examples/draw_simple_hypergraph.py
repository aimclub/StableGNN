from torch import tensor

from stable_gnn.visualization.contracts.hypergraph_contract import HypergraphContract
from stable_gnn.visualization.contracts.hypergraph_visualization_contract import HypergraphVisualizationContract
from stable_gnn.visualization.hypergraph_visualization import HypergraphVisualizer

graph_contract: HypergraphContract = HypergraphContract(
    vertex_num=10,
    edge_list=(  # noqa
        [(3, 4, 5, 9), (0, 4, 7), (4, 6), (0, 1, 2, 4), (3, 6), (0, 3, 9), (2, 5), (4, 7)],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ),
    edge_num=8,
    edge_weights=tensor(  # noqa
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ),
)
vis_contract: HypergraphVisualizationContract = HypergraphVisualizationContract(graph=graph_contract)

vis: HypergraphVisualizer = HypergraphVisualizer(vis_contract)
fig = vis.draw()
fig.show()

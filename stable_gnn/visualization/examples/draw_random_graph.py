from torch import tensor

from stable_gnn.visualization.contracts.graph_contract import GraphContract
from stable_gnn.visualization.contracts.graph_visualization_contract import GraphVisualizationContract
from stable_gnn.visualization.data_generation.graph_generator import GraphGenerator
from stable_gnn.visualization.graph_visualization import GraphVisualizer

VERTEX_NUM = 1000
EDGE_NUM = 1200


generator = GraphGenerator(vertex_num=VERTEX_NUM, edge_num=EDGE_NUM)
generated_data = generator()

generated_edge_weights = [1.0 for _ in range(len(generated_data))]


graph_contract: GraphContract = GraphContract(
    vertex_num=VERTEX_NUM,
    edge_list=(generated_data, generated_edge_weights),  # noqa
    edge_num=EDGE_NUM,
    edge_weights=tensor(generated_edge_weights * 2),  # noqa
)
vis_contract: GraphVisualizationContract = GraphVisualizationContract(graph=graph_contract)

vis: GraphVisualizer = GraphVisualizer(vis_contract)
fig = vis.draw()
fig.show()

print("Complete...")

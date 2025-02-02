from stable_gnn.visualization.config.parameters.generator_methods import GeneratorMethods
from stable_gnn.visualization.data_generation.hypergraph_generator import HypergraphGenerator

print("Generate random Hypergraph 'uniform'")
generator = HypergraphGenerator(vertex_num=10, edge_num=3, generation_method=GeneratorMethods.uniform)
data = generator()
print(data)

print("Generate random Hypergraph 'low_order_first'")
generator = HypergraphGenerator(vertex_num=5, edge_num=4, generation_method=GeneratorMethods.low_order_first)
data = generator()
print(data)

print("Generate random Hypergraph 'high_order_first'")
generator = HypergraphGenerator(vertex_num=5, edge_num=4, generation_method=GeneratorMethods.high_order_first)
data = generator()
print(data)

print("Generate random Hypergraph 'custom'")
prob_k_list = [0, 0, 0.8, 0.2]
generator = HypergraphGenerator(
    vertex_num=5, edge_num=4, generation_method=GeneratorMethods.custom, probability_k_list=prob_k_list
)
data = generator()
print(data)

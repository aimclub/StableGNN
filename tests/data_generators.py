import os

import networkx as nx
import networkx.generators as gen


def generate_star_graphs(root_dir, size_of_star=5, num_of_stars=20):
    name = "stars"

    if not os.path.exists(root_dir + str(name)):
        graph = nx.DiGraph()
        prev_nodes = 0
        central_nodes = []
        for num in range(num_of_stars):
            g = gen.star_graph(size_of_star)
            mapping = {}
            for i in range(len(g.nodes())):
                mapping[i] = i + prev_nodes
            central_nodes.append(prev_nodes)
            prev_nodes += len(g.nodes())

            for edge in g.edges():
                graph.add_edge(mapping[edge[0]], mapping[edge[1]])

        for node in central_nodes:
            for node2 in central_nodes:
                if node > node2:
                    graph.add_edge(node, node2)

        path_to_dir = root_dir + f"/{name}/"
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        if not os.path.exists(path_to_dir):
            os.mkdir(path_to_dir)
        if not os.path.exists(path_to_dir + "raw"):
            os.mkdir(path_to_dir + "raw")

        with open(path_to_dir + "raw/" + "labels.txt", "a") as f:
            for i in graph.nodes():
                if i in central_nodes:
                    f.write(str(1) + "\n")
                else:
                    f.write(str(0) + "\n")

        with open(path_to_dir + "raw/" + "edges.txt", "a") as f:
            for i in graph.edges():
                f.write(str(i[0]) + "," + str(i[1]) + "\n")

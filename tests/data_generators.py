import os
import random

import networkx as nx
import networkx.generators as gen
import numpy as np


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


def generate_gc_graphs(root_dir, num_graphs=50):
    name = "ba_gc"

    if not os.path.exists(root_dir + str(name)):
        types_of_graphs = ["wheel", "circ ladder", "star", "ba"]

        for k in range(num_graphs):
            type = random.choice(types_of_graphs)
            n = random.randint(10, 20)
            if type == "wheel":
                g = gen.wheel_graph(n)
                if np.random.binomial(1, 0.85, 1) == 1:
                    y = 0
                else:
                    y = random.choice([0, 1, 2, 3])
            elif type == "cir ladder":
                g = gen.circular_ladder_graph(n)
                if np.random.binomial(1, 0.85, 1) == 1:
                    y = 1
                else:
                    y = random.choice([0, 1, 2, 3])
            elif type == "star":
                g = gen.star_graph(n - 1)
                if np.random.binomial(1, 0.85, 1) == 1:
                    y = 2
                else:
                    y = random.choice([0, 1, 2, 3])
            else:
                g = gen.barabasi_albert_graph(n, 3)
                if np.random.binomial(1, 0.85, 1) == 1:
                    y = 1
                else:
                    y = random.choice([0, 1, 2, 3])

            x = np.random.rand(n, 32)

            path_to_dir = root_dir + f"/{name}/"
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)
            if not os.path.exists(path_to_dir):
                os.mkdir(path_to_dir)
            if not os.path.exists(path_to_dir + "raw"):
                os.mkdir(path_to_dir + "raw")

            with open(path_to_dir + "raw/" + "edges_" + str(k) + ".txt", "a") as f:
                for edge in g.edges():
                    f.write(str(edge[0]) + "," + str(edge[1]) + "\n")

            with open(path_to_dir + "raw/" + "attrs_" + str(k) + ".txt", "a") as f:
                for line in x:
                    for i in line:
                        f.write(str(i) + ",")
                    f.write(str(y) + "\n")

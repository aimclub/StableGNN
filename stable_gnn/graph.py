import os
import warnings
from os import listdir
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.typing import Tensor
from torch_geometric.utils import coalesce, dense_to_sparse, negative_sampling
from torch_geometric.utils.undirected import to_undirected


class Graph(InMemoryDataset):
    """Read graph data from txt files and learning structure (denoising) with adjust function. The data can be either one graph or the list of graphs. The data should be located in root/name/raw directory

    For one graph, data should contain three files:

    - attrs.txt with N lines, each line means attributes of corresponding node, attributes separated from each other with ','.

    - edges.txt consists of two columns of nodes separeted with ',', each row of this table is a pair of vertices connected by an edge.

    - labels.txt is a column of numbers, meaning labels of nodes. The size of this column is the size of input graph.

    For list of graphs, data should contain 3*M similar files, where M is the number of graphs in dataset.

    Denoising now is possible for datasets consisting of ONE graph, for Node Classification tasks.

    :param name: (string) Name of your dataset,
    :param root: (string): Root directory where the raw dataset is located and processed graph should be saved.
    :param transform: (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: `None`)
    :param pre_transform: (callable, optional): A function/transform that takes in
        an `torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: `None`)
    :param adjust_flag: (bool): When set to True, the graph is denoised (default: True)
    :param sigma_u: (float): Variance of randomly generated representations of nodes, required if adjust_flag = True (default: 0.7)
    :param sigma_e: (float, optional) Variance of randomly generated noise, required if adjust_flag = True (default: 0.4)
    """

    # TODO перепроверить можно ли на куду все перенести, в первый раз не получилось - ноль ускорения
    def __init__(
        self,
        name: str,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        adjust_flag: bool = True,
        sigma_u: float = 0.7,
        sigma_e: Optional[float] = 0.4,
    ) -> None:
        # reading input files consisting of edges.txt, attrs.txt, y.txt
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.name = name
        self.sigma_u = sigma_u
        self.sigma_e = sigma_e
        self.adjust_flag = adjust_flag
        self.num_negative_samples = 5

        if self.name == "texas" or self.name == "wisconsin":
            self.url = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/" + self.name
        elif self.name == "BACE":
            self.url = "https://raw.githubusercontent.com/anpolol/data_validation/main/BACE/"
        if self.name == "usa" or self.name == "brazil" or self.name == "europe":
            self.edge_url = "https://github.com/leoribeiro/struc2vec/raw/master/graph/{}-airports.edgelist"
            self.label_url = "https://github.com/leoribeiro/struc2vec/raw/master/graph/labels-{}-airports.txt"

        super().__init__(self.root, self.transform, self.pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """
        Find str names of the raw input data files

        :return: (list): List of str names
        """
        if self.name == "texas" or self.name == "wisconsin":
            out = ["out1_node_feature_label.txt", "out1_graph_edges.txt"]
        elif self.name == "BACE":
            out = []
            for i in range(500):
                out.append("attrs_" + str(i) + ".txt")
            for i in range(500):
                out.append("edges_" + str(i) + ".txt")
        elif self.name == "usa" or self.name == "europe" or self.name == "brazil":
            return [
                f"{self.name}-airports.edgelist",
                f"labels-{self.name}-airports.txt",
            ]
        else:
            out = [
                self.name + "_attrs.txt",
                self.name + "_edges.txt",
                self.name + "_labels.txt",
            ]
        return out

    def download(self) -> None:
        """Download the data from the link given"""
        if self.name in ["texas", "wisconsin"]:
            for f in self.raw_file_names:
                download_url(f"{self.url}/{f}", self.raw_dir)
        if self.name == "BACE":
            for f in self.raw_file_names:
                download_url(f"{self.url}/{f}", self.raw_dir)
        if self.name == "usa" or self.name == "brazil" or self.name == "europe":
            download_url(self.edge_url.format(self.name), self.raw_dir)
            download_url(self.label_url.format(self.name), self.raw_dir)

    @property
    def processed_file_names(self) -> List[str]:
        """Return the name of the file of the processed input graph

        return: ([str]): the name of the processed file
        """
        return [self.name + "_data.pt"]

    def process(self) -> None:
        """Process the raw files of the input data"""
        if self.name == "texas" or self.name == "wisconsin":
            self._process_texas()
        elif self.name == "usa" or self.name == "brazil" or self.name == "europe":
            self._process_airport()
        else:
            names_datasets = listdir(self.raw_dir)

            if (len(names_datasets) == 3) or (len(names_datasets) == 2):  # 1 graph: edge list, labels, attrs
                self._process_1graph()
            else:  # many graphs
                self._process_many_graphs()

    def _process_airport(self):
        index_map, ys = {}, []
        with open(self.raw_paths[1], "r") as f:
            data = f.read().split("\n")[1:-1]
            for i, row in enumerate(data):
                idx, y = row.split()
                index_map[int(idx)] = i
                ys.append(int(y))
        y = torch.tensor(ys, dtype=torch.long)
        x = torch.eye(y.size(0))

        edge_indices = []
        with open(self.raw_paths[0], "r") as f:
            data = f.read().split("\n")[:-1]
            for row in data:
                src, dst = row.split()
                edge_indices.append([index_map[int(src)], index_map[int(dst)]])
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_index = coalesce(edge_index, num_nodes=y.size(0))
        if self.adjust_flag:
            self.num_nodes = len(x)
            edge_index = self._adjust(edge_index=edge_index)
        if self.root is not None:
            np.save(self.root + "/X.npy", x.numpy())
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def _process_many_graphs(self) -> None:
        if self.adjust_flag:
            warnings.warn(
                "Warning! We can adjust only 1 graph, so use adjust_flag==True only for Node Classification tasks"
            )
        number_of_graphs = int(len(listdir(self.raw_dir)) / 2)
        data_list = []

        for i in range(number_of_graphs):
            lines = self._read_files("", self.raw_dir, "edges_" + str(i) + ".txt")

            edge_list = []
            for line in lines:
                f = line.split(",")
                edge_list.append([int(f[0]), int(f[1])])
            edge_index = torch.tensor(edge_list).T

            lines = self._read_files("", self.raw_dir, "attrs_" + str(i) + ".txt")

            attrs = []
            for line in lines:
                f = line.split(",")
                attr = []
                for symb in f[: len(f) - 1]:
                    attr.append(float(symb))
                attrs.append(attr)

            y = torch.tensor(int(float(lines[0].split(",")[(len(f) - 1)])))
            x = torch.tensor(attrs)
            G = Data(edge_index=edge_index, x=x, y=y)
            data_list.append(G)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _process_texas(self) -> None:
        with open(self.raw_paths[0], "r") as f:
            data_raw = f.read().split("\n")[1:-1]
            pre_x = [[float(v) for v in r.split("\t")[1].split(",")] for r in data_raw]
            x = torch.tensor(pre_x, dtype=torch.float)

            pre_y = [int(r.split("\t")[2]) for r in data_raw]
            y = torch.tensor(pre_y, dtype=torch.long)

        with open(self.raw_paths[1], "r") as f:
            data_raw = f.read().split("\n")[1:-1]
            data = [[int(v) for v in r.split("\t")] for r in data_raw]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = coalesce(edge_index, num_nodes=x.size(0))
        self.num_nodes = len(y)
        print(self.num_nodes)
        if self.adjust_flag:
            edge_index = self._adjust(edge_index=edge_index)
        data = Data(x=x, y=y, edge_index=edge_index)
        if self.root is not None:
            np.save(self.root + "/X.npy", x.numpy())
        data_list = [data]
        data, slices = self.collate(data_list)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save((data, slices), self.processed_paths[0])

    def _process_1graph(self) -> None:
        edge_index = self._read_edges(self.raw_dir)
        edge_index = to_undirected(edge_index)

        # labels reading
        y = self._read_labels(self.raw_dir)

        self.num_nodes = len(y)

        try:  # TODO: это лучше через assert? + мы же не рассматриваем несвязаные графы?
            max(
                int(torch.max(edge_index[0])), int(torch.max(edge_index[1]))
            ) == self.num_nodes - 1  # numbering starts with 0 so self.num_nodes = max_index+1
        except ValueError:
            print(
                "number of nodes in your graph differ from max index of nodes. Possible reasons (but not the only one): your graph has connected components of size = 1, or numbering starts with 1 (should with 0)"
            )

        # attributes reading
        x, d = self._read_attrs(self.raw_dir)

        if self.adjust_flag:
            edge_index = self._adjust(
                edge_index=edge_index,
            )

        data = Data(x=x, edge_index=edge_index, y=y)
        data.name = self.name
        data_list = [data]
        data, slices = self.collate(data_list)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save((data, slices), self.processed_paths[0])

    def _read_edges(self, path_initial: str) -> Tensor:
        edge_index_list = []
        for line in self._read_files("", path_initial, "edges.txt"):
            split_line = line.split(",")
            edge_index_list.append([int(split_line[0]), int(split_line[1])])
        edge_index = torch.tensor(edge_index_list)
        edge_index = edge_index.T

        return edge_index

    def _read_labels(self, path_initial: str) -> Tensor:
        y_list = []
        for line in self._read_files("", path_initial, "labels.txt"):
            y_list.append(int(line))
        y = torch.tensor(y_list)
        return y

    def _read_attrs(self, path_initial: str) -> Tuple[Tensor, int]:
        d = 128  # случай если нет атрибутов добавляем случайные из норм распределения
        if os.path.exists(path_initial + "/" + "attrs.txt"):
            x_list = []
            for line in self._read_files("", path_initial, "attrs.txt"):
                split_line = line.split(",")
                x_attr = []
                for attr in split_line:
                    x_attr.append(float(attr))
                x_list.append(x_attr)
            x = torch.tensor(x_list)
            d = x.shape[1]
            if self.root is not None:
                np.save(self.root + "/X.npy", x.numpy())
            return x, d
        else:
            x = torch.rand(self.num_nodes, d)
            if self.root is not None:
                np.save(self.root + "/X.npy", x.numpy())
            return x, d

    @staticmethod
    def _read_files(name: str, path_initial: str, txt_file_postfix: str) -> List[str]:
        path_file = path_initial + "/" + name + txt_file_postfix
        if os.path.exists(path_file):
            with open(path_file, "r") as f:
                lines = f.readlines()
        else:
            raise Exception("there is no " + str(path_file) + " file")
        return lines

    # Learn structure
    def _adjust(self, edge_index):
        # generation of genuine graph structure
        m = 64  # TODO найти какой именной тут размер, или гиперпараметр?
        u = torch.normal(
            mean=torch.zeros((self.num_nodes, m)),
            std=torch.ones((self.num_nodes, m)) * self.sigma_u,
        )
        u.requires_grad = True
        # a_approx = torch.bernoulli(torch.clamp(a_approx_prob, min=0, max=1)) #TODO в статье есть эта строчка однако я не понимаю зачем, если в ф.п. только log(prob)
        # generation of noise
        e = torch.normal(
            mean=torch.zeros((self.num_nodes, self.num_nodes)),
            std=torch.ones((self.num_nodes, self.num_nodes)) * self.sigma_e,
        )
        e.requires_grad = True

        optimizer = torch.optim.Adam([u, e], lr=0.01, weight_decay=1e-5)
        optimizer.zero_grad()
        # TODO ниже негатив семлинг для каждого позитивного ищет негативный пример. Если отдать num_negative_sample то он вернет num_negative_sample ребер ВСЕГо на весь граф. В стаье для каждой вершины строятся негативные примеры. Стоит ли этот момент тут исправить?
        # если пережать в функцию negative_sampling num_negative_samples , то у нас будет всего num_negative_samples негативных примеров, хотя хотелось бы для каждой вершины сколько-то негативных примеров

        negative_samples = negative_sampling(
            edge_index, self.num_nodes, num_neg_samples=len(edge_index[0]) * 5, method="dense"
        )

        for i in range(89):
            optimizer.zero_grad()
            loss = self._loss(u, e, edge_index, negative_samples, m)

            loss.backward(retain_graph=True)
            optimizer.step()
        # approximating genuine graph
        u_diff = u.view(1, self.num_nodes, m) - u.view(self.num_nodes, 1, m)
        a_genuine = torch.nn.Sigmoid()(-(u_diff * u_diff).sum(axis=2))

        # TODO: в этом я тоже не уверена (то что ниже)
        # print(a_genuine)

        a_genuine = torch.bernoulli(torch.clamp(a_genuine, min=0, max=1))

        np.save(self.root + "/A.npy", a_genuine.detach().numpy())
        edge_index, _ = dense_to_sparse(a_genuine)
        return edge_index

    def _loss(self, u, e, edge_index, negative_samples, m):
        u_diff = u.view(1, self.num_nodes, m) - u.view(self.num_nodes, 1, m)
        a_genuine = torch.nn.Sigmoid()(-(u_diff * u_diff).sum(axis=2))  # high assortativity assumption

        # approximating input graph structure
        a_approx_prob = a_genuine + e
        a_approx = torch.clamp(a_approx_prob, min=1e-5, max=1)

        alpha_u = 1
        alpha_e = 1
        positive_indices_flattened = torch.concat(
            [
                edge_index[0] * self.num_nodes + edge_index[1],
                edge_index[1] * self.num_nodes + edge_index[0],
            ]
        )
        loss_proximity = -torch.sum(
            torch.log(torch.take(a_approx, positive_indices_flattened))
        )  # TODO:  a_approx_prob судя по статье, мы обрезаем до [0,1], но тогда log() даст inf  loss танет inf, поэтому я обрезала не от нуля а от 10^-5 хз насколько правильно
        loss_u = torch.sum(u * u)
        loss_e = torch.sum(e * e)

        negative_indices_flattened = torch.concat(
            [
                negative_samples[0] * self.num_nodes + negative_samples[1],
                negative_samples[1] * self.num_nodes + negative_samples[0],
            ]
        )

        loss_proximity_negative = -torch.sum(torch.log(1 - torch.take(a_approx, negative_indices_flattened)))

        return loss_proximity + alpha_u * loss_u + alpha_e * loss_e + loss_proximity_negative

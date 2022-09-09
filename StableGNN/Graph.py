import torch
from torch_geometric.data import Data
import os

class Graph(): #TODO: а нужно ли тут наследование от Data? как будто нет
    '''
    reading Graph data from txt files and learning structure (denoising) with adjust

    Args
    d -- dimension of attributes. You can set it if your data has no attributes for generating random attributes
    '''
    def __init__(self, dataset_name: str, d: int = 128):
        # reading input files consisting of edges.txt, attrs.txt, y.txt
        self.name = dataset_name

        path_initial = './DataValidation/' + dataset_name #TODO: как сделать более универсальное чтение? передавать весь путь, а не только dataset_name?
        # edges reading
        self.edge_index = self.read_edges(path_initial)

        #labels reading
        self.y = self.read_labels( path_initial)

        self.num_nodes = len(self.y)

        try: #TODO: это лучше через assert? + мы же не рассматриваем несвязаные графы?
            max(int(torch.max(self.edge_index[0])), int(torch.max(self.edge_index[1]))) == self.num_nodes-1 #numbering starts with 0 so num_nodes = max_index+1
        except:
            raise Exception("number of nodes in your graph differ from max index of nodes. Possible reasons (but not the only one): your graph has connected components of size = 1, or numbering starts with 1 (should with 0)")

        #attributes reading
        self.x , self.d = self.read_attrs(path_initial,d)
        super().__init__()

    def read_edges(self, path_initial):
        path_edges = path_initial + '/' + self.name + '_edges.txt'
        if os.path.exists(path_edges):
            with open(path_edges, 'r') as f:
                lines = f.readlines()
                edge_index = []
                for line in lines:
                    split_line = line.split(',')
                    edge_index.append([int(split_line[0]), int(split_line[1])])
            edge_index = torch.tensor(edge_index)
            edge_index = edge_index.T
            return edge_index
        else:
            raise Exception("there is no " + path_edges + " file")
            # TODO: exceptions это правильно?

    def read_labels(self, path_initial):
        path_labels = path_initial + '/' + self.name + '_labels.txt'
        if os.path.exists(path_labels):
            with open(path_labels, 'r') as f:
                lines = f.readlines()
                y = []
                for line in lines:
                    y.append(int(line))
            y = torch.tensor(y)
            return y
        else:
            raise Exception("there is no " + path_labels + " file")
            # TODO: exceptions это правильно?

    def read_attrs(self, path_initial, d):
        path_attrs = path_initial + '/' + self.name + '_attrs.txt'
        if os.path.exists(path_attrs):
            with open(path_attrs, 'r') as f:
                lines = f.readlines()
                x = []
                for line in lines:
                    split_line = line.split(',')
                    x_attr = []
                    for attr in split_line:
                        x_attr.append(float(attr))
                    x.append(x_attr)
            x = torch.tensor(x)
            d = x.shape[1]
            return x, d
        else:
            x = torch.rand(self.num_nodes, d) #TODO: добавить другие варианты случайной инициации
            return x,d

    def adjust(self): #Learn Structure
        pass
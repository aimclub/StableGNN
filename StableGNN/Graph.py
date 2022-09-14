import torch
import os
import random
from torch_geometric.utils import subgraph
from torch_geometric.utils import dense_to_sparse

class Graph(): #TODO: а нужно ли тут наследование от Data? как будто нет
    '''
    reading Graph data from txt files and learning structure (denoising) with adjust

    Args
    d -- dimension of attributes. You can set it if your data has no attributes for generating random attributes
    dataset_name -- name of your dataset
    sigma_u -- variance of randomly generated representations of nodes
    sigma_e -- variance of randomly generated noise
    '''
    #TODO перепроверить можно ли на куду все перенести, в первый раз не получилось - ноль ускорения
    def __init__(self, dataset_name: str, device:str='cpu', d: int = 128, sigma_u: int = 0.8, sigma_e: int = 0.2):
        # reading input files consisting of edges.txt, attrs.txt, y.txt
        self.name = dataset_name
        self.sigma_u = sigma_u
        self.sigma_e = sigma_e
        self.device = device

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

    def to(self,device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)

    def read_edges(self, path_initial):
            edge_index = []
            for line in self.read_files(self.name,path_initial,'_edges.txt'):
                split_line = line.split(',')
                edge_index.append([int(split_line[0]), int(split_line[1])])
            edge_index = torch.tensor(edge_index)
            edge_index = edge_index.T
            return edge_index

    def read_labels(self, path_initial):
            y = []
            for line in self.read_files(self.name,path_initial, '_labels.txt'):
                 y.append(int(line))
            y = torch.tensor(y)
            return y


    def read_attrs(self, path_initial, d):
        try:
            x = []
            for line in self.read_files(self.name,path_initial,'_attrs.txt'):
                 split_line = line.split(',')
                 x_attr = []
                 for attr in split_line:
                    x_attr.append(float(attr))
                 x.append(x_attr)
            x = torch.tensor(x)
            d = x.shape[1]
            return x, d
        except:
            x = torch.rand(self.num_nodes, d)
            return x, d

    def read_files(self, name, path_initial, txt_file_postfix):
        path_file = path_initial + '/' + name + txt_file_postfix
        if os.path.exists(path_file):
            with open(path_file, 'r') as f:
                lines = f.readlines()
        else:
            raise Exception('there is no '+str(path_file) +' file') # TODO: exceptions это правильно?
        return lines


    def adjust(self): #Learn Structure
        #generation of genuine graph structure
        m = 64 #TODO найти какой именной тут размер, или гиперпараметр?
        u = torch.normal(mean=torch.zeros((self.num_nodes, m)), std=torch.ones((self.num_nodes, m))*self.sigma_u)
        u.requires_grad = True
        u_diff = u.view(1, self.num_nodes, m) - u.view(self.num_nodes, 1, m)
        a_genuine = torch.nn.Sigmoid()(-(u_diff*u_diff).sum(axis=2)) #high assortativity assumption
        # a_approx = torch.bernoulli(torch.clamp(a_approx_prob, min=0, max=1)) #TODO в статье есть эта строчка однако я не понимаю зачем, если в ф.п. только log(prob)
        #generation of noise
        e = torch.normal(mean=torch.zeros((self.num_nodes, self.num_nodes)), std=torch.ones((self.num_nodes, self.num_nodes ))*self.sigma_e)
        e.requires_grad = True
        #approximating input graph structure
        a_approx_prob = a_genuine + e
        #a_approx_prob = a_approx_prob.to(self.device)

        e = e.to(self.device)
        u = u.to(self.device)
        u_diff = u_diff.to(self.device)

        optimizer = torch.optim.Adam([u, e], lr=0.01, weight_decay=1e-5)
        optimizer.zero_grad()
        for i in range(100):
            print(i)
            loss = self.loss(u, e, torch.clamp(a_approx_prob, min=1e-5, max=1))
            loss.backward(retain_graph=True)
            optimizer.step()

        #approximating genuine graph
        u_diff = u.view(1, self.num_nodes, m) - u.view(self.num_nodes, 1, m)
        a_genuine = torch.nn.Sigmoid()(-(u_diff*u_diff).sum(axis=2))
        #TODO: в этом я тоже не уверена (то что ниже)
        a_genuine = torch.bernoulli(torch.clamp(a_genuine, min=0, max=1))

        self.edge_index, _ = dense_to_sparse(a_genuine)


    def loss(self, u, e, a_approx):
        alpha_u = 1
        alpha_e = 1
        positive_indices_flattened = torch.concat([self.edge_index[0]*self.num_nodes + self.edge_index[1], self.edge_index[1]*self.num_nodes + self.edge_index[0]])
        loss_proximity = -torch.sum(torch.log(torch.take(a_approx,positive_indices_flattened))) #TODO:  a_approx_prob судя по статье, мы обрезаем до [0,1], но тогда log() даст inf  loss танет inf, поэтому я обрезала не от нуля а от 10^-5 хз насколько правильно
        loss_u = torch.sum(u*u)
        loss_e = torch.sum(e*e)
        #self.negative_sampling(torch.tensor(list(range(self.num_nodes))), 5)

        # loss_neg_part =

        return loss_proximity + alpha_u*loss_u + alpha_e*loss_e


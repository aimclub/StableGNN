import torch
import os
import random
from torch_geometric.utils import dense_to_sparse,negative_sampling
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
#root '/tmp/Cora'

#TODO на данный момент не реализовано сохранение отдельно в папку processed_adjust графа после уточнения структуры и
#TODO отдельно в папку processed графа без уточнения структуры. Приходится удалять содержимое папки processed если хочется посчитать другое
#TODO именно поэтому в TrainModel number of negative samples for graph.adjust не влияет на результат - постоянно считывается один и тот же граф из папки

class Graph(InMemoryDataset):
    '''
    reading Graph data from txt files and learning structure (denoising) with adjust

    Args
    d -- dimension of attributes. You can set it if your data has no attributes for generating random attributes
    dataset_name -- name of your dataset
    sigma_u -- variance of randomly generated representations of nodes
    sigma_e -- variance of randomly generated noise
    '''
    #TODO перепроверить можно ли на куду все перенести, в первый раз не получилось - ноль ускорения
    def __init__(self, name, root, transform=None, pre_transform=None, pre_filter=None, adjust_flag=True, sigma_u=0.7, sigma_e=0.4, num_negative_samples=5):
        # reading input files consisting of edges.txt, attrs.txt, y.txt
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform

        self.name = name

        self.sigma_u = sigma_u
        self.sigma_e = sigma_e
        self.adjust_flag = adjust_flag
        self.num_negative_samples = 5
        super().__init__(self.root, self.transform, self.pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
            return [self.name+'_attrs.txt', self.name+'_edges.txt', self.name+'_labels.txt']

    @property
    def processed_file_names(self):
        return [self.name+'_data.pt']

    def process(self):
 #       if self.pre_transform is not None:
  #          data_list = [self.pre_transform(data) for data in data_list]
        print(self.raw_dir)
        edge_index = self.read_edges(self.raw_dir)


        #labels reading
        y = self.read_labels(self.raw_dir)

        self.num_nodes = len(y)

        try: #TODO: это лучше через assert? + мы же не рассматриваем несвязаные графы?
            max(int(torch.max(edge_index[0])), int(torch.max(edge_index[1]))) == self.num_nodes-1 #numbering starts with 0 so self.num_nodes = max_index+1
        except:
            raise Exception("number of nodes in your graph differ from max index of nodes. Possible reasons (but not the only one): your graph has connected components of size = 1, or numbering starts with 1 (should with 0)")

        #attributes reading
        x, d = self.read_attrs(self.raw_dir)


        if self.adjust_flag:
            edge_index = self.adjust(edge_index=edge_index, num_negative_samples= self.num_negative_samples*len(x))

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list = [data]
        data, slices = self.collate(data_list)


        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save((data, slices), self.processed_paths[0])

    def read_edges(self, path_initial):
            edge_index = []
            for line in self.read_files(self.name, path_initial,'_edges.txt'):
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


    def read_attrs(self, path_initial):
        d = 128 #TODO как задать?!
        try:
            x = []
            for line in self.read_files(self.name, path_initial, '_attrs.txt'):
                 split_line = line.split(',')
                 x_attr = []
                 for attr in split_line:
                    x_attr.append(float(attr))
                 x.append(x_attr)
            x = torch.tensor(x)
            d = x.shape[1]
            np.save(self.root+'/X.npy', x.numpy())
            return x, d
        except:
            x = torch.rand(self.num_nodes, d)
            np.save(self.root+'/X.npy', x.numpy())
            return x, d

    def read_files(self, name, path_initial, txt_file_postfix):
        path_file = path_initial + '/' + name + txt_file_postfix
        if os.path.exists(path_file):
            with open(path_file, 'r') as f:
                lines = f.readlines()
        else:
            raise Exception('there is no '+str(path_file) +' file')
        return lines


    def adjust(self, edge_index, num_negative_samples): #Learn Structure
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

        e = e#.to(self.device)
        u = u#.to(self.device)
        u_diff = u_diff#.to(self.device)

        optimizer = torch.optim.Adam([u, e], lr=0.01, weight_decay=1e-5)
        optimizer.zero_grad()
        #TODO ниже негатив семлинг для каждого позитивного ищет негативный пример. Если отдать num_negative_sample то он вернет num_negative_sample ребер ВСЕГо на весь граф. В стаье для каждой вершины строятся негативные примеры. Стоит ли этот момент тут исправить?
        #сли пережать в функцию negative_sampling num_negative_samples , то у нас будет всего num_negative_samples негативных примеров, хотя хотелось бы для каждой вершины сколько-то негативных примеров

        negative_samples = negative_sampling(edge_index, self.num_nodes, method='dense')

        for i in range(100):
            print(i)
            loss = self.loss(u, e, torch.clamp(a_approx_prob, min=1e-5, max=1), edge_index, negative_samples)
            loss.backward(retain_graph=True)
            optimizer.step()

        #approximating genuine graph
        u_diff = u.view(1, self.num_nodes, m) - u.view(self.num_nodes, 1, m)
        a_genuine = torch.nn.Sigmoid()(-(u_diff*u_diff).sum(axis=2))

        #TODO: в этом я тоже не уверена (то что ниже)
        a_genuine = torch.bernoulli(torch.clamp(a_genuine, min=0, max=1))
        print(self.root)
        print(self.name)
        np.save(self.root + '/A.npy', a_genuine.detach().numpy())
        edge_index, _ = dense_to_sparse(a_genuine)
        return edge_index

    def loss(self, u, e, a_approx,edge_index,negative_samples):
        alpha_u = 1
        alpha_e = 1
        positive_indices_flattened = torch.concat([edge_index[0]*self.num_nodes + edge_index[1], edge_index[1]*self.num_nodes + edge_index[0]])
        loss_proximity = -torch.sum(torch.log(torch.take(a_approx, positive_indices_flattened))) #TODO:  a_approx_prob судя по статье, мы обрезаем до [0,1], но тогда log() даст inf  loss танет inf, поэтому я обрезала не от нуля а от 10^-5 хз насколько правильно
        loss_u = torch.sum(u*u)
        loss_e = torch.sum(e*e)

        negative_indices_flattened = torch.concat([negative_samples[0] * self.num_nodes + negative_samples[1],
                                                   negative_samples[1] * self.num_nodes + negative_samples[0]])
        loss_proximity_negative = -torch.sum(torch.log(1 - torch.take(a_approx,
                                                         negative_indices_flattened)))

        return loss_proximity + alpha_u*loss_u + alpha_e*loss_e + loss_proximity_negative


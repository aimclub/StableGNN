from StableGNN.Graph import Graph
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from StableGNN.Model import ModelName
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import optuna
import numpy as np
import torch
import random

class TrainModel():
    def __init__(self, name='Cora2', conv='GAT', device='cuda'):
        data = Graph(name)
        data.adjust()
        self.Conv = conv
        self.device = device
        self.x = data.x
        self.y = data.y.squeeze()
        self.edge_index = data.edge_index

        self.data = Data(x=self.x, edge_index=self.edge_index, y=self.y)
        self.data = self.data.to(device)

        self.train_mask, self.val_mask, self.test_mask = self.train_test_split(self.x)
        super(TrainModel, self).__init__()

    def train_test_split(self, x):
        indices = list(range(len(x)))
        random.seed(0)
        train_indices = random.sample(indices, int(len(indices)*0.7))
        left_indices = list(set(indices) - set(train_indices))
        random.seed(1)
        val_indices = random.sample(left_indices, int(len(indices)*0.1))
        test_indices = list(set(left_indices)-set(val_indices))

        train_indices = torch.tensor(train_indices)
        val_indices = torch.tensor(val_indices)
        test_indices = torch.tensor(test_indices)

        train_mask = torch.tensor([False] * len(indices))
        test_mask = torch.tensor([False] * len(indices))
        val_mask = torch.tensor([False] * len(indices))
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        val_mask[val_indices] = True
        return train_mask, val_mask, test_mask

    def train(self, model, data, optimizer, train_loader, dropout):
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        y = self.y.to(self.device)
        for batch_size, n_id, adjs in train_loader:

            if len(train_loader.sizes) == 1:
                adjs = [adjs]
            adjs = [adj.to(self.device) for adj in adjs]
            out = model.forward(data.x[n_id.to(self.device)].to(self.device), adjs)
            loss = model.loss_sup(out, y[self.train_mask])
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        return total_loss / len(train_loader)


    @torch.no_grad()
    def test(self, model, data):  # ,n_estimators,learning_rate_catboost, max_depth):
        model.eval()
        out = model.inference(data.to(self.device))
        y_pred = out.cpu().argmax(dim=-1, keepdim=True)
        accs_micro = []
        accs_macro = []
        for mask in [self.train_mask, self.test_mask, self.val_mask]:
            accs_micro += [f1_score(self.y.detach()[mask].cpu().numpy(), y_pred[mask], average='micro')]
            accs_macro += [f1_score(self.y.detach()[mask].cpu().numpy(), y_pred[mask], average='macro')]

        return accs_micro, accs_macro

    def run(self, params):

        hidden_layer = params['hidden_layer']
        dropout = params['dropout']
        size = params['size of network, number of convs']
        learning_rate = params['lr']
        train_loader = NeighborSampler(self.data.edge_index, node_idx=self.train_mask, batch_size=int(sum(self.train_mask)), sizes=[-1] * size)

        model = ModelName(dataset=self.data, conv=self.Conv, device=self.device, hidden_layer=hidden_layer, num_layers=size, dropout=dropout)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        losses = []
        train_accs_mi = []
        test_accs_mi = []
        val_accs = []
        train_accs_ma = []
        test_accs_ma = []
        log = 'Loss: {:.4f}, Epoch: {:03d}, Train acc micro: {:.4f}, Test acc micro: {:.4f},Train acc macro: {:.4f}, Test acc macro: {:.4f}'

        for epoch in range(100):
            print(epoch)
            loss = self.train(model, self.data, optimizer, train_loader, dropout)
            losses.append(loss.detach().cpu())
            [train_acc_mi, test_acc_mi, val_acc_mi], [train_acc_ma, test_acc_ma, val_acc_ma] = self.test(model, self.data)
            train_accs_mi.append(train_acc_mi)
            test_accs_mi.append(test_acc_mi)
            train_accs_ma.append(train_acc_ma)
            test_accs_ma.append(test_acc_ma)
            print(log.format(loss, epoch, train_acc_mi, test_acc_mi, train_acc_ma, test_acc_ma))

            # scheduler.step()
        print(log.format(loss, epoch, train_acc_mi, test_acc_mi, train_acc_ma, test_acc_ma))
        plt.plot(losses)
        plt.title(' loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        plt.plot(test_accs_mi)
        plt.title(' test f1 micro')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

        plt.plot(test_accs_ma)
        plt.title( ' test f1 macro')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        return train_acc_mi, test_acc_mi, train_acc_ma, test_acc_ma


class TrainModelOptuna(TrainModel):
    def objective(self, trial):
        # Integer parameter
        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128, 256])
        out_layer = trial.suggest_categorical("out_layer", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        size = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        Conv = self.Conv
        learning_rate = trial.suggest_float("lr", 5e-3, 1e-2)

        model = ModelName(dataset=self.data, conv=Conv,  device=self.device,hidden_layer=hidden_layer, num_layers=size, dropout=dropout)
        train_loader = NeighborSampler(self.data.edge_index, batch_size=int(sum(self.train_mask)),
                                       node_idx=self.train_mask, sizes=[-1] * size)

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(50):
            loss = self.train(model, self.data, optimizer, train_loader, dropout)
        [train_acc_mi, test_acc_mi, val_acc_mi], [train_acc_ma, test_acc_ma, val_acc_ma] = self.test(model, self.data)
        trial.report(np.sqrt(val_acc_mi * val_acc_ma))
        return np.sqrt(val_acc_mi * val_acc_ma)

    def run(self, number_of_trials):

        study = optuna.create_study(direction="maximize",
                                    study_name=self.loss["Name"] + " loss," + str(self.Conv) + " conv")
        study.optimize(self.objective, n_trials=number_of_trials)
        trial = study.best_trial
        return trial.params
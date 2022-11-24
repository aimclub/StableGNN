from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from StableGNN.Model import ModelName
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import optuna
import numpy as np
import torch
import random
from torch_geometric.loader import DataLoader


class TrainModel:
    def __init__(
        self,
        data,
        conv="GAT",
        device="cuda",
        SSL = False,
            EXTRAPOLATE=True
    ):

        # arguments = {'d': d, 'sigma_u': sigma_u, 'sigma_e': sigma_e, 'device': device, 'name' : name, '_store': 'C:'}
       # print(data)

        self.Conv = conv
        self.device = device
        self.x = data.x
        self.y = data.y.squeeze()
        self.edge_index = data.edge_index

        self.data = data
        self.data = self.data#.to(device)
        self.SSL = SSL
        self.EXTRAPOLATE = EXTRAPOLATE
        self.train_mask, self.val_mask, self.test_mask = self.train_test_split(self.x)
        super(TrainModel, self).__init__()

    def train_test_split(self, x):
        indices = list(range(len(x)))
        random.seed(0)
        train_indices = random.sample(indices, int(len(indices) * 0.7))
        left_indices = list(set(indices) - set(train_indices))
        random.seed(1)
        val_indices = random.sample(left_indices, int(len(indices) * 0.1))
        test_indices = list(set(left_indices) - set(val_indices))

        train_indices = torch.tensor(train_indices)
        val_indices = torch.tensor(val_indices)
        test_indices = torch.tensor(test_indices)

        train_mask = torch.tensor([False] * len(indices))
        test_mask = torch.tensor([False] * len(indices))
        val_mask = torch.tensor([False] * len(indices))
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        val_mask[val_indices] = True
        return train_indices, val_indices, test_indices



    def train(self, model, optimizer, train_loader):
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        len(train_loader)
        for dat in train_loader:
            dat = dat.to(self.device)
            out, deg_pred = model.forward(dat.x, dat.edge_index, dat.edge_weight, dat.batch)
            loss = model.loss_sup(out, dat.y)
            if self.SSL:
                loss_SSL = model.SelfSupervisedLoss(deg_pred,dat)
                total_loss += loss_SSL
            total_loss += loss
            loss_SSL.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def test(self, model, loader):  # ,n_estimators,learning_rate_catboost, max_depth):
        model.eval()
        accs_micro=[]
        accs_macro=[]
        for dat in loader:
            dat = dat.to(self.device)
            out, _ = model.forward(dat.x, dat.edge_index, dat.edge_weight, dat.batch)
            y_pred = out.cpu().argmax(dim=1, keepdim=True)
            y_true = dat.y
            accs_micro.append(f1_score(
                y_true.cpu().tolist(),  y_pred.squeeze().tolist(), average="micro"
                    ))
            accs_macro.append(f1_score(
                y_true.cpu().tolist(),  y_pred.squeeze().tolist(), average="macro"
                    ))
        return np.mean(accs_micro), np.mean(accs_macro)

    def run(self, params):

        hidden_layer = params["hidden_layer"]
        dropout = params["dropout"]
        size = params["size of network, number of convs"]
        learning_rate = params["lr"]


        model = ModelName(
            dataset=self.data,
            conv=self.Conv,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,SSL=self.SSL
        )
        model.to(self.device)

        if self.EXTRAPOLATE:
            init_edges = False
            remove_init_edges = False
            white_list = False
            score_func = 'MI' #TODO придумать как их задавать пользователю
            self.train_dataset, self.test_dataset, self.val_dataset = model.Extrapolate(self.train_mask, self.val_mask, init_edges, remove_init_edges,white_list, score_func)

       # train_loader = NeighborSampler(
        #    self.data.edge_index,
         #   node_idx=self.train_mask,
          #  batch_size=int(sum(self.train_mask)),
           # sizes=[-1] * size,
        #)
        #self.data = self.train_dataset+self.test_dataset
        train_loader = DataLoader(self.train_dataset, batch_size=20, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=20, shuffle=True)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        losses = []
        train_accs_mi = []
        test_accs_mi = []
        train_accs_ma = []
        test_accs_ma = []
        log = "Loss: {:.4f}, Epoch: {:03d}, Test acc micro: {:.4f}, Test acc macro: {:.4f}"

        for epoch in range(100):
            loss = self.train(model, optimizer, train_loader)
            losses.append(loss.detach().cpu())
            test_acc_mi, test_acc_ma = self.test(model, test_loader)
           # train_accs_mi.append(train_acc_mi)
            test_accs_mi.append(test_acc_mi)
          #  train_accs_ma.append(train_acc_ma)
            test_accs_ma.append(test_acc_ma)
            print(
                log.format(
                    loss, epoch, test_acc_mi, test_acc_ma
                )
            )

            # scheduler.step()
        print(
            log.format(
                loss, epoch, test_acc_mi, test_acc_ma
            )
        )
        plt.plot(losses)
        plt.title(" loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
        plt.plot(test_accs_mi)
        plt.title(" test f1 micro")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

        plt.plot(test_accs_ma)
        plt.title(" test f1 macro")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
        return model, test_acc_mi, test_acc_ma


class TrainModelOptuna(TrainModel):
    def objective(self, trial):
        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        size = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        Conv = self.Conv
        learning_rate = trial.suggest_float("lr", 5e-3, 1e-2)

        model = ModelName(
            dataset=self.data,
            conv=Conv,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,SSL=self.SSL
        )

        if self.EXTRAPOLATE:
            init_edges = False
            remove_init_edges = False
            white_list = False
            score_func = 'MI'  # TODO придумать как их задавать пользователю
            self.train_dataset, self.test_dataset, self.val_dataset = model.Extrapolate(self.train_mask,self.val_mask, init_edges, remove_init_edges,
                                                                      white_list, score_func)

        # train_loader = NeighborSampler(
        #    self.data.edge_index,
        #   node_idx=self.train_mask,
        #  batch_size=int(sum(self.train_mask)),
        # sizes=[-1] * size,
        # )
       # self.data = self.train_dataset + self.test_dataset
        train_loader = DataLoader(self.train_dataset, batch_size=20, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=20, shuffle=True)

        model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        for epoch in range(50):
            _ = self.train(model, optimizer, train_loader)
        val_acc_mi, val_acc_ma = self.test(model, val_loader)
        return np.sqrt(val_acc_mi * val_acc_ma)

    def run(self, number_of_trials):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=number_of_trials)
        trial = study.best_trial
        return trial.params

import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader

from stable_gnn.model_gc import ModelName as Model_GC
from stable_gnn.model_nc import ModelName as Model_NC


class TrainModel:
    def __init__(
        self,
        data,
        dataset_name,
        conv="GAT",
        device="cuda",
        task="NC",
        ssl_flag=False,
        extrapolate_flag=True,
    ):

        # arguments = {'d': d, 'sigma_u': sigma_u, 'sigma_e': sigma_e, 'device': device, 'name' : name, '_store': 'C:'}
        # print(data)

        self.Conv = conv
        self.device = device

        # self.data = self.data#.to(device)
        self.ssl_flag = ssl_flag
        self.task = task
        self.extrapolate_flag = extrapolate_flag
        self.data_name = dataset_name
        if self.task == "NC":
            self.Model = Model_NC
            self.data = data[0]
            self.y = self.data.y.squeeze()
            N = len(self.data.x)
            self.test = self.test_nc

        elif self.task == "GC":
            self.Model = Model_GC
            self.data = data
            N = len(data)
            self.test = self.test_gc
        else:
            raise Exception(
                "there is no such task, try again with TASK_FLAG as NC (Node Classification) or GC (Graph Classification)"
            )

        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
            self.train_mask,
            self.val_mask,
            self.test_mask,
        ) = self.train_test_split(N)

        if self.task == "NC" and self.extrapolate_flag:
            self.extrapolate_flag = False
            warnings.warn(
                "Warning! Extraplation released only for Node Classification task, so next results will be for graph Classification without Extrapolation"
            )

        super(TrainModel, self).__init__()

    def train(self, model, optimizer, train_loader=None):
        model.train()
        optimizer.zero_grad()
        total_loss = 0

        if self.task == "NC":
            out, deg_pred = model.inference(self.data.to(self.device))

            y = self.y.type(torch.LongTensor)
            y = y.to(self.device)
            if self.ssl_flag:
                loss_SSL = model.SelfSupervisedLoss(deg_pred, self.train_mask)
                total_loss += loss_SSL
                loss_SSL.backward(retain_graph=True)
            loss = model.loss_sup(out[self.train_mask], y[self.train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss

        else:
            for dat in train_loader:
                dat = dat.to(self.device)
                batch_edge_list = dat.edge_index
                batch_x = dat.x
                batch = dat.batch
                y = dat.y

                out, deg_pred = model.forward(batch_x, batch_edge_list, batch)

                loss = model.loss_sup(out, y)
                total_loss += loss
                if self.SSL:
                    loss_SSL = model.SelfSupervisedLoss(deg_pred, dat)
                    total_loss += loss_SSL
                    loss_SSL.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            return total_loss / len(train_loader)

    @torch.no_grad()
    def test_gc(self, model, loader, **args):  # ,n_estimators,learning_rate_catboost, max_depth):
        model.eval()
        accs_micro = []
        accs_macro = []
        for dat in loader:
            dat = dat.to(self.device)
            out, _ = model.forward(dat.x, dat.edge_index, dat.edge_weight, dat.batch)
            y_pred = out.cpu().argmax(dim=1, keepdim=True)
            y_true = dat.y
            accs_micro.append(accuracy_score(y_true.cpu().tolist(), y_pred.squeeze().tolist()))
            accs_macro.append(f1_score(y_true.cpu().tolist(), y_pred.squeeze().tolist(), average="macro"))
        return np.mean(accs_micro), np.mean(accs_macro)

    @torch.no_grad()
    def test_nc(self, model, mask, **kwargs):  # ,n_estimators,learning_rate_catboost, max_depth):
        model.eval()
        out, _ = model.inference(self.data.to(self.device))
        y_pred = out.cpu().argmax(dim=-1, keepdim=True)

        accs_micro = accuracy_score(self.y.detach()[mask].cpu().numpy(), y_pred[mask])
        accs_macro = f1_score(self.y.detach()[mask].cpu().numpy(), y_pred[mask], average="macro")

        return accs_micro, accs_macro

    def train_test_split(self, N):
        indices = list(range(N))
        random.seed(0)
        train_indices = random.sample(indices, int(len(indices) * 0.7))
        left_indices = list(set(indices) - set(train_indices))
        random.seed(1)
        val_indices = random.sample(left_indices, int(len(indices) * 0.1))
        test_indices = list(set(left_indices) - set(val_indices))

        train_indices = torch.tensor(train_indices)
        val_indices = torch.tensor(val_indices)
        test_indices = torch.tensor(test_indices)

        train_mask = torch.tensor([False] * N)
        train_mask[train_indices] = True

        test_mask = torch.tensor([False] * N)
        test_mask[test_indices] = True

        val_mask = torch.tensor([False] * N)
        val_mask[val_indices] = True
        return train_indices, val_indices, test_indices, train_mask, val_mask, test_mask

    def run(self, params):

        hidden_layer = params["hidden_layer"]
        dropout = params["dropout"]
        size = params["size of network, number of convs"]
        learning_rate = params["lr"]

        model = self.Model(
            dataset=self.data,
            data_name=self.data_name,
            conv=self.Conv,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,
            ssl_flag=self.ssl_flag,
        )

        model.to(self.device)

        if self.task == "GC":
            if self.extrapolate_flag:
                init_edges = False
                remove_init_edges = False
                white_list = False
                score_func = "MI"  # TODO придумать как их задавать пользователю
                (self.train_dataset, self.test_dataset, self.val_dataset,) = model.Extrapolate(
                    self.train_indices,
                    self.val_indices,
                    init_edges,
                    remove_init_edges,
                    white_list,
                    score_func,
                )

            # train_loader = NeighborSampler(
            #    self.data.edge_index,
            #   node_idx=self.train_mask,
            #  batch_size=int(sum(self.train_mask)),
            # sizes=[-1] * size,
            # )
            # self.data = self.train_dataset+self.test_dataset
            else:
                self.train_dataset, self.test_dataset, _, _ = model.convert_dataset(
                    data=self.data,
                    train_indices=self.train_indices,
                    val_indices=self.val_indices,
                )
            train_loader = DataLoader(self.train_dataset, batch_size=20, shuffle=True)
            test_loader = DataLoader(self.test_dataset, batch_size=20, shuffle=False)
        else:
            train_loader = None
            test_loader = None

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        losses = []
        train_accs_mi = []
        train_accs_ma = []
        log = "Loss: {:.4f}, Epoch: {:03d}, Train acc micro: {:.4f}, Train acc macro: {:.4f}"

        for epoch in range(100):

            loss = self.train(model, optimizer, train_loader)
            losses.append(loss.detach().cpu())
            train_acc_mi, train_acc_ma = self.test(model, mask=self.train_mask, loader=test_loader)
            train_accs_mi.append(train_acc_mi)
            train_accs_ma.append(train_acc_ma)
            print(log.format(loss, epoch, train_acc_mi, train_acc_ma))

        test_acc_mi, test_acc_ma = self.test(model, mask=self.test_mask)
        # scheduler.step()
        print(
            "Loss: {:.4f}, Epoch: {:03d}, test acc micro: {:.4f}, test acc macro: {:.4f}".format(
                loss, epoch, test_acc_mi, test_acc_ma
            )
        )
        plt.plot(losses)
        plt.title(" loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
        plt.plot(train_accs_mi)
        plt.title(" train f1 micro")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

        plt.plot(train_accs_ma)
        plt.title(" train f1 macro")
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

        model = self.Model(
            dataset=self.data,
            conv=Conv,
            device=self.device,
            hidden_layer=hidden_layer,
            num_layers=size,
            dropout=dropout,
            ssl_flag=self.ssl_flag,
            data_name=self.data_name,
        )

        if self.task == "GC":
            if self.extrapolate_flag:
                init_edges = False
                remove_init_edges = False
                white_list = False
                score_func = "MI"  # TODO придумать как их задавать пользователю
                (self.train_dataset, self.test_dataset, self.val_dataset,) = model.Extrapolate(
                    self.train_indices,
                    self.val_indices,
                    init_edges,
                    remove_init_edges,
                    white_list,
                    score_func,
                )
            else:
                self.train_dataset, _, self.val_dataset, _ = model.convert_dataset(
                    self.data,
                    train_indices=self.train_indices,
                    val_indices=self.val_indices,
                )
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
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

            for epoch in range(50):
                _ = self.train(model, optimizer, train_loader)
            val_acc_mi, val_acc_ma = self.test(model, val_loader=val_loader, mask=self.val_mask)
            return np.sqrt(val_acc_mi * val_acc_ma)
        else:
            train_loader = None
            val_loader = None

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(50):
            _ = self.train(model, optimizer, train_loader=train_loader)
        val_acc_mi, val_acc_ma = self.test(model, mask=self.val_mask, loader=val_loader)
        return val_acc_mi

    def run(self, number_of_trials):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=number_of_trials)
        trial = study.best_trial
        return trial.params

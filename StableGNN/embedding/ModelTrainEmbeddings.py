import os
import pickle

import optuna
import torch
import torch_geometric.transforms as T
from torch.optim import lr_scheduler
from torch_geometric.loader import NeighborSampler

from StableGNN.Graph import Graph

from .model import Net
from .sampling import (SamplerAPP, SamplerContextMatrix, SamplerFactorization,
                       SamplerRandomWalk)


class ModelTrainEmbeddings:
    def __init__(self, name, conv="SAGE", device="cuda", loss_function="APP"):
        data = Graph(
            name,
            root="./DataValidation/" + str(name),
            transform=T.NormalizeFeatures(),
            ADJUST_FLAG=False,
        )[0]
        self.Conv = conv
        self.device = device
        self.x = data.x
        self.y = data.y.squeeze()
        self.data = data.to(device)
        self.train_mask = torch.Tensor([True] * data.num_nodes)
        self.loss = loss_function
        self.datasetname = name
        self.flag = self.loss["flag_tosave"]

        self.help_data = "stableGNN/data_help/"
        super(ModelTrainEmbeddings, self).__init__()

    def sampling(self, Sampler, epoch, nodes, loss):

        if epoch == 0:
            if self.flag:
                if "alpha" in self.loss:
                    name_of_file = (
                        self.datasetname
                        + "_samples_"
                        + loss["Name"]
                        + "_alpha_"
                        + str(loss["alpha"])
                        + ".pickle"
                    )
                elif "betta" in self.loss:
                    name_of_file = (
                        self.datasetname
                        + "_samples_"
                        + loss["Name"]
                        + "_betta_"
                        + str(loss["betta"])
                        + ".pickle"
                    )
                else:
                    name_of_file = (
                        self.datasetname + "_samples_" + loss["Name"] + ".pickle"
                    )

                if os.path.exists(f"{self.help_data}/" + str(name_of_file)):
                    with open(f"{self.help_data}/" + str(name_of_file), "rb") as f:
                        self.samples = pickle.load(f)
                else:
                    self.samples = Sampler.sample(nodes)
                    with open(f"{self.help_data}/" + str(name_of_file), "wb") as f:
                        pickle.dump(self.samples, f)
            else:
                self.samples = Sampler.sample(nodes)

    def train(
        self, model, data, optimizer, Sampler, train_loader, dropout, epoch, loss
    ):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        # print('train loader',len(train_loader))
        if model.conv == "GCN":
            arr = torch.nonzero(self.train_mask == True)
            indices_of_train_data = [item for sublist in arr for item in sublist]
            # print('before',data.x)
            out = model.inference(data.to(self.device), dp=dropout)
            # print('after',out, sum(sum(out)))
            samples = self.sampling(Sampler, epoch, indices_of_train_data, loss)
            loss = model.loss(out[self.train_mask], self.samples)
            # print('loss',loss)
            total_loss += loss
        else:
            for batch_size, n_id, adjs in train_loader:
                if len(train_loader.sizes) == 1:
                    adjs = [adjs]
                adjs = [adj.to(self.device) for adj in adjs]
                out = model.forward(data.x[n_id.to(self.device)].to(self.device), adjs)
                self.sampling(Sampler, epoch, n_id[:batch_size], loss)
                loss = model.loss(
                    out, self.samples
                )  # pos_batch.to(device), neg_batch.to(device))
                total_loss += loss
        total_loss.backward()
        optimizer.step()
        return total_loss / len(train_loader), out

    def run(self, params):

        hidden_layer = params["hidden_layer"]
        # out_layer = params['out_layer']
        dropout = params["dropout"]
        size = params["size of network, number of convs"]
        learning_rate = params["lr"]
        hidden_layer_for_classifier = params["hidden_layer_for_classifier"]

        # hidden_layer=64,out_layer=128,dropout=0.0,size=1,learning_rate=0.001,c=100
        classifier = "logistic regression"
        train_loader = NeighborSampler(
            self.data.edge_index, batch_size=self.data.num_nodes, sizes=[-1] * size
        )

        Sampler = self.loss["Sampler"]

        LossSampler = Sampler(
            self.datasetname,
            self.data,
            device=self.device,
            mask=self.train_mask,
            loss_info=self.loss,
            help_dir=self.help_data,
        )
        model = Net(
            dataset=self.data,
            conv=self.Conv,
            loss_function=self.loss,
            device=self.device,
            hidden_layer=hidden_layer,
            out_layer=2,
            num_layers=(size),
            dropout=dropout,
        )
        model.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

        for epoch in range(99):
            print(epoch)
            loss, _ = self.train(
                model,
                self.data,
                optimizer,
                LossSampler,
                train_loader,
                dropout,
                epoch,
                self.loss,
            )
        _, out = self.train(
            model,
            self.data,
            optimizer,
            LossSampler,
            train_loader,
            dropout,
            epoch,
            self.loss,
        )
        # np.save('../data_help/embedings_'+str(self.datasetname)+str(self.loss['name'])+'.npy', out.cpu().numpy())

        # scheduler.step()

        return out


class OptunaTrainEmbeddings(ModelTrainEmbeddings):
    def objective(self, trial):
        # Integer parameter
        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128, 256])
        out_layer = 2
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        size = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        Conv = self.Conv
        learning_rate = trial.suggest_float("lr", 5e-3, 1e-2)

        # c =trial.suggest_categorical("c",  [0.001, 0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,20,30,100])
        hidden_layer_for_classifier = trial.suggest_categorical(
            "hidden_layer_for_classifier", [32, 64, 128, 256]
        )
        # варьируем параметры
        loss_to_train = {}
        for name in self.loss:

            if type(self.loss[name]) == list:
                if len(self.loss[name]) == 3:
                    var = trial.suggest_int(
                        name,
                        self.loss[name][0],
                        self.loss[name][1],
                        step=self.loss[name][2],
                    )
                    loss_to_train[name] = var
                elif len(self.loss[name]) == 2:
                    var_2 = trial.suggest_float(
                        name, self.loss[name][0], self.loss[name][1]
                    )
                    loss_to_train[name] = var_2
                else:
                    var_3 = trial.suggest_categorical(name, self.loss[name])
                    loss_to_train[name] = var_3
            else:
                loss_to_train[name] = self.loss[name]
        if name == "q" and type(self.loss[name]) == list:
            var_5 = trial.suggest_categorical("p", self.loss["p"])
            var_4 = trial.suggest_categorical("q", self.loss[name])
            if var_4 > 1:
                var_4 = 1
            if var_5 < var_4:
                var_5 = var_4
            loss_to_train["q"] = var_4
            loss_to_train["p"] = var_5

        Sampler = loss_to_train["Sampler"]
        model = Net(
            dataset=self.data,
            conv=Conv,
            loss_function=loss_to_train,
            device=self.device,
            hidden_layer=hidden_layer,
            out_layer=out_layer,
            num_layers=size,
            dropout=dropout,
        )
        train_loader = NeighborSampler(
            self.data.edge_index, batch_size=int(self.data.num_nodes), sizes=[-1] * size
        )

        LossSampler = Sampler(
            self.datasetname,
            self.data,
            device=self.device,
            mask=self.train_mask,
            loss_info=loss_to_train,
            help_dir=self.help_data,
        )
        model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        for epoch in range(50):
            loss, _ = self.train(
                model,
                self.data,
                optimizer,
                LossSampler,
                train_loader,
                dropout,
                epoch,
                loss_to_train,
            )
        return loss

    def run(self, number_of_trials):

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=number_of_trials)
        trial = study.best_trial
        return trial.params

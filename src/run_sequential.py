import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim import SGD, Adam

from data.data import LibTXTData
from data.sparseloader import DataLoader
from models.ensemble_net import EnsembleNet
from models.mlp import MLP_1HL, MLP_2HL
from utils import auc_score, init_gbnn, root_mse

with open("config.yaml", "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


config = Config(**data["params"])


def get_data(train_path, test_path, binary=False):

    if binary:
        train = LibTXTData(train_path, config)
        test = LibTXTData(test_path, config)

        scaler = MinMaxScaler()
        scaler.fit(train.feat)
        train.feat = scaler.transform(train.feat)
        test.feat = scaler.transform(test.feat)
    else:
        train = LibTXTData(train_path, config)
        test = LibTXTData(test_path, config)

        scaler = StandardScaler()
        scaler.fit(train.feat)
        train.feat = scaler.transform(train.feat)
        test.feat = scaler.transform(test.feat)

    return train, test


def get_optim(params, lr):
    optimizer = Adam(params, lr)
    # optimizer = SGD(params, lr, weight_decay=0.0003)
    return optimizer


class Experiment:
    def __init__(self, config):
        self.config = config
        self.g_func = None
        self.lambda_func = None
        self.loss_func = None
        self.acc_func = None
        self.c0 = None

    def load_data(self, binary=False):
        if config.data == "sunspot":
            train_path = "datasets/Sunspot/train.txt"
            test_path = "datasets/Sunspot/test.txt"

            config.feat_d = 4
            config.hidden_d = 10
            config.out_d = 1
        if config.data == "rossler":
            train_path = "datasets/Rossler/train.txt"
            test_path = "datasets/Rossler/test.txt"

            config.feat_d = 4
            config.hidden_d = 10
            config.out_d = 1
        if config.data == "ionosphere":
            train_path = "datasets/Ionosphere/train.txt"
            test_path = "datasets/Ionosphere/test.txt"

            config.feat_d = 34
            config.hidden_d = 50
            config.out_d = 1
        if config.data == "cancer":
            train_path = "datasets/Cancer/train.txt"
            test_path = "datasets/Cancer/test.txt"

            config.feat_d = 9
            config.hidden_d = 12
            config.out_d = 1

        self.train, self.test = get_data(train_path, test_path, binary=binary)

    def init_experiment(self):
        if self.config.data in ["sunspot", "rossler"]:  # Regression problems
            self.load_data()
            self.init_regression()
            self.config.classification = False
        elif self.config.data in ["ionosphere", "cancer"]:  # Classification problems
            self.load_data(binary=True)
            self.init_classification()
            self.config.classification = True

    def init_regression(self):
        self.g_func = lambda y, yhat: 2 * (yhat - y)
        self.lambda_func = lambda y, fx, Fx: torch.sum(fx * (y - Fx)) / torch.sum(fx * fx)
        self.loss_func = nn.MSELoss()
        self.acc_func = root_mse

        self.c0 = np.mean(self.train.label)

    def init_classification(self):
        self.g_func = lambda y, yhat: -1 * (2 * y) / (1 + torch.exp(2 * y * yhat))
        self.h_func = lambda y, yhat: (4 * torch.exp(2 * y * yhat)) / (
            1 + torch.exp(2 * y * yhat) ** 2
        )

        self.lambda_func = lambda y, fx, Fx: torch.mean(
            -1 / fx * self.g_func(y, Fx) / self.h_func(y, Fx)
        )
        self.loss_func = nn.MSELoss()
        self.acc_func = auc_score

        self.c0 = init_gbnn(self.train)

    def run_experiment(self):

        for level in range(1, config.num_nets + 1):

            t0 = time.time()

            tr_score_l = []
            te_score_l = []
            for _ in range(config.exps):
                tr_rmse, te_rmse = self.run(self.train, self.test, level)
                tr_score_l.append(tr_rmse)
                te_score_l.append(te_rmse)

            total_time = time.time() - t0

            print(f"Boosting level: {level}")
            if self.config.classification:
                print(
                    f"Training statistics: mean - {np.mean(tr_score_l)}, std = {np.std(tr_score_l)}, best = {np.amax(tr_score_l)}"
                )
                print(
                    f"Test statistics: mean - {np.mean(te_score_l)}, std = {np.std(te_score_l)}, best = {np.amax(te_score_l)}"
                )
            else:
                print(
                    f"Training statistics: mean - {np.mean(tr_score_l)}, std = {np.std(tr_score_l)}, best = {np.amin(tr_score_l)}"
                )
                print(
                    f"Test statistics: mean - {np.mean(te_score_l)}, std = {np.std(te_score_l)}, best = {np.amin(te_score_l)}"
                )
            print(f"Total elapsed time: {total_time}")

    def run(self, train, test, num_nets):
        model_type = MLP_1HL

        net_ensemble = EnsembleNet(self.c0, config.lr)

        train_loader = DataLoader(train, 1, shuffle=True, drop_last=False, num_workers=2)
        test_loader = DataLoader(test, 1, shuffle=False, drop_last=False, num_workers=2)

        final_tr_score = 0
        final_te_score = 0
        for stage in range(num_nets):

            model = model_type.get_model(config)
            if config.cuda:
                model.cuda()

            optimizer = get_optim(model.parameters(), config.lr)
            net_ensemble.to_train()  # Set the models in ensemble net to train mode

            for i, (x, y) in enumerate(train_loader):

                if config.cuda:
                    x = x.cuda()
                    y = torch.as_tensor(y, dtype=torch.float32).cuda().view(-1, 1)
                out = net_ensemble.forward(x)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)

                grad_direction = -1 * self.g_func(y, out)

                out = model(x)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                loss = self.loss_func(out, grad_direction)

                model.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate gamma
            x = torch.tensor(train.feat).cuda()
            y = torch.tensor(train.label).cuda()

            gamma = self.lambda_func(y, model(x), net_ensemble.forward(x))

            net_ensemble.add(model, gamma)

            if config.cuda:
                net_ensemble.to_cuda()
            net_ensemble.to_eval()  # Set the models in ensemble net to eval mode

            # Train
            final_tr_score = self.acc_func(net_ensemble, train_loader)

            final_te_score = self.acc_func(net_ensemble, test_loader)

            # print(
            #     f"Stage: {stage}  RMSE@Tr: {tr_rmse:.5f}, RMSE@Val: {val_rmse:.5f}, RMSE@Te: {te_rmse:.5f}"
            # )

        # print(f"Stage: {num_nets}  RMSE@Tr: {tr_rmse:.5f}, final RMSE@Te: {te_rmse:.5f}")

        return final_tr_score, final_te_score


if __name__ == "__main__":

    exp = Experiment(config)
    exp.init_experiment()
    exp.run_experiment()


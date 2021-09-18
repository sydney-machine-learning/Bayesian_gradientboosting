import math
import time
from multiprocessing import Value

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim import SGD, Adam

from data.data import LibCSVData, LibTXTData
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


def get_data(train_path, test_path, multistep=False, binary=False):

    if binary:
        if multistep:
            train = LibCSVData(train_path + "train1.csv", config)
            test = LibCSVData(test_path + "test1.csv", config)
        else:
            train = LibTXTData(train_path + "train.txt", config)
            test = LibTXTData(test_path + "test.txt", config)

        scaler = MinMaxScaler()
        scaler.fit(train.feat)
        train.feat = scaler.transform(train.feat)
        test.feat = scaler.transform(test.feat)
    else:
        if multistep:
            train = LibCSVData(train_path + "train1.csv", config)
            test = LibCSVData(test_path + "test1.csv", config)
        else:
            train = LibTXTData(train_path + "train.txt", config)
            test = LibTXTData(test_path + "test.txt", config)

        scaler = StandardScaler()
        scaler.fit(train.feat)
        train.feat = scaler.transform(train.feat)
        test.feat = scaler.transform(test.feat)

    return train, test


def get_optim(params, lr):
    optimizer = Adam(params, lr)
    # optimizer = SGD(params, lr)
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
            train_path = "datasets/Sunspot/"
            test_path = "datasets/Sunspot/"

            config.hidden_d = 10
        elif config.data == "rossler":
            train_path = "datasets/Rossler/"
            test_path = "datasets/Rossler/"

            config.hidden_d = 10
        elif config.data == "ionosphere":
            train_path = "datasets/Ionosphere/"
            test_path = "datasets/Ionosphere/"

            config.feat_d = 34
            config.hidden_d = 50
        elif config.data == "cancer":
            train_path = "datasets/Cancer/"
            test_path = "datasets/Cancer/"

            config.feat_d = 9
            config.hidden_d = 12
        else:
            raise ValueError("Invalid dataset specified")

        self.train, self.test = get_data(
            train_path, test_path, multistep=self.config.multistep, binary=binary
        )

    def init_experiment(self):
        if self.config.data in ["sunspot", "rossler"]:  # Regression problems
            if self.config.multistep:
                self.config.feat_d = 5
                self.config.out_d = 10
            else:
                self.config.feat_d = 4
                self.config.out_d = 1

            self.load_data()
            self.init_regression()
            self.config.classification = False
        elif self.config.data in ["ionosphere", "cancer"]:  # Classification problem
            self.config.out_d = 1

            self.load_data(binary=True)
            self.init_classification()
            self.config.classification = True
        else:
            raise ValueError("Invalid dataset specified")

    def log_likelihood_timeseries(self, model, data, w, tau_sq):

        x, y = data.feat, data.label

        fx = model.evaluate_proposal(x, w)

        if y.ndims > 1:
            n = (
                y.shape[0] * y.shape[1]
            )  # number of samples x number of outputs (prediction horizon)
        else:
            n = y.shape[0]

        p1 = -(n / 2) * np.log(2 * math.pi * tau_sq)
        p2 = 1 / 2 * tau_sq

        result = p1 - (p2 * np.sum(np.square(y - fx)))

        return result

    def prior_likelihood_timeseries(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.config.hidden_d  # number hidden neurons
        d = self.config.feat_d  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def init_regression(self):
        self.g_func = lambda y, yhat: 2 * (yhat - y)
        self.lambda_func = lambda y, fx, Fx: torch.sum(fx * (y - Fx)) / torch.sum(fx * fx)
        self.loss_func = nn.MSELoss()
        self.acc_func = root_mse

        self.log_likelihood_func = self.log_likelihood_timeseries
        self.prior_func = self.prior_likelihood_timeseries

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
        """
        Runs the full experiment and prints out accuracy statistics after each stage (boosting level).
        """

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

    def train_mcmc(self, net_ensemble, model, train_data):

        x, y = train_data.feat, train_data.label
        if config.cuda:
            x = x.cuda

    def train_backprop(self, net_ensemble, model, train_data):
        optimizer = get_optim(model.parameters(), config.lr)
        net_ensemble.to_train()  # Set the models in ensemble net to train mode

        for i, (x, y) in enumerate(train_data):
            if config.cuda:
                x = torch.as_tensor(x, dtype=torch.float32).cuda()
                y = torch.as_tensor(y, dtype=torch.float32).cuda()
            out = net_ensemble.forward(x)
            # out = torch.as_tensor(out, dtype=torch.float32).cuda()

            grad_direction = -1 * self.g_func(y, out)

            out = model(x)
            out = torch.as_tensor(out, dtype=torch.float32).cuda()
            loss = self.loss_func(out, grad_direction)

            model.zero_grad()
            loss.backward()
            optimizer.step()

    def run(self, train, test, num_nets):
        """
        Run one instance of the experiment

        Args:
            train (list): Train data
            test (list): Test data
            num_nets (int): Number of weak learners to train

        Returns:
            final_tr_score (float), final_te_score (float): Training and test scores after ensemble is fully trained
        """
        model_type = MLP_1HL

        net_ensemble = EnsembleNet(self.c0, config.lr)

        train.shuffle()
        # train_loader = DataLoader(train, 1, shuffle=True, drop_last=False, num_workers=2)
        # test_loader = DataLoader(test, 1, shuffle=False, drop_last=False, num_workers=2)

        final_tr_score = 0
        final_te_score = 0
        for stage in range(num_nets):

            model = model_type.get_model(config)
            if config.cuda:
                model.cuda()

            if config.mcmc:
                self.train_mcmc(net_ensemble, model, train)
            else:
                self.train_backprop(net_ensemble, model, train)

            # Calculate gamma
            x = torch.tensor(train.feat, dtype=torch.float32).cuda()
            y = torch.tensor(train.label, dtype=torch.float32).cuda()

            gamma = self.lambda_func(y, model(x), net_ensemble.forward(x))

            net_ensemble.add(model, gamma)

            if config.cuda:
                net_ensemble.to_cuda()
            net_ensemble.to_eval()  # Set the models in ensemble net to eval mode

            # Train
            final_tr_score = self.acc_func(net_ensemble, train)

            final_te_score = self.acc_func(net_ensemble, test)

            # print(
            #     f"Stage: {stage}  RMSE@Tr: {tr_rmse:.5f}, RMSE@Val: {val_rmse:.5f}, RMSE@Te: {te_rmse:.5f}"
            # )

        # print(f"Stage: {num_nets}  RMSE@Tr: {tr_rmse:.5f}, final RMSE@Te: {te_rmse:.5f}")

        return final_tr_score, final_te_score


if __name__ == "__main__":

    exp = Experiment(config)
    exp.init_experiment()
    exp.run_experiment()


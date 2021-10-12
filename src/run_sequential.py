import math
import random
import time
from multiprocessing import Value
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import grad
from torch.optim import SGD, Adam

from data.data import LibCSVData, LibTXTData
from functions import Regression
from models.ensemble_net import EnsembleNet
from models.mlp import MLP_1HL
from utils import auc_score, gr_convergence_rate, init_gbnn, mse_torch, root_mse

with open("config.yaml", "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


config = Config(**data["params"])


def get_data_help(path, config, index_col=None):
    if ".csv" in path:
        return LibCSVData(path, config, index_col=index_col)
    elif ".txt" in path:
        return LibTXTData(path, config)
    else:
        raise ValueError("Invalid data file type provided.")


def get_data(train_path, test_path, multistep=False, classification=False):

    if classification:
        train = get_data_help(train_path, config)
        test = get_data_help(test_path, config)
    else:
        train = get_data_help(train_path, config, index_col=0)
        test = get_data_help(test_path, config, index_col=0)

    scaler = StandardScaler()
    scaler.fit(train.feat)
    train.feat = scaler.transform(train.feat)
    test.feat = scaler.transform(test.feat)

    return train, test


def get_optim(params, config):
    if config.optimizer == "adam":
        return Adam(params, config.lr)
    elif config.optimizer == "sgd":
        return SGD(params, config.lr)
    else:
        raise ValueError("Invalid optimizer specified, choose adam or sgd.")


class Experiment:
    def __init__(self, config):
        self.config = config
        self.g_func = None
        self.lambda_func = None
        self.loss_func = None
        self.acc_func = None
        self.c0 = None

    def load_data(self, classification=False):
        if config.data == "sunspot":
            if self.config.multistep:
                train_path = "datasets/Sunspot/train1.csv"
                test_path = "datasets/Sunspot/test1.csv"
            else:
                train_path = "datasets/Sunspot/train.txt"
                test_path = "datasets/Sunspot/test.txt"
        elif config.data == "rossler":
            if self.config.multistep:
                train_path = "datasets/Rossler/train1.csv"
                test_path = "datasets/Rossler/test1.csv"
            else:
                train_path = "datasets/Rossler/train.txt"
                test_path = "datasets/Rossler/test.txt"
        elif config.data == "henon":
            if self.config.multistep:
                train_path = "datasets/Henon/train1.csv"
                test_path = "datasets/Henon/test1.csv"
            else:
                train_path = "datasets/Henon/train.txt"
                test_path = "datasets/Henon/test.txt"
        elif config.data == "lazer":
            if self.config.multistep:
                train_path = "datasets/Lazer/train1.csv"
                test_path = "datasets/Lazer/test1.csv"
            else:
                train_path = "datasets/Lazer/train.txt"
                test_path = "datasets/Lazer/test.txt"
        elif config.data == "ionosphere":
            train_path = "datasets/Ionosphere/ftrain.csv"
            test_path = "datasets/Ionosphere/ftest.csv"

            config.feat_d = 34
            config.hidden_d = 50
            config.out_d = 2
        elif config.data == "cancer":
            train_path = "datasets/Cancer/ftrain.txt"
            test_path = "datasets/Cancer/ftest.txt"

            config.feat_d = 9
            config.hidden_d = 12
            config.out_d = 2
        elif config.data == "bank":
            train_path = "datasets/Bank/train.csv"
            test_path = "datasets/Bank/test.csv"

            config.feat_d = 51
            config.hidden_d = 90
            config.out_d = 2
        elif config.data == "pendigit":
            train_path = "datasets/PenDigit/train.csv"
            test_path = "datasets/PenDigit/test.csv"

            config.feat_d = 16
            config.hidden_d = 30
            config.out_d = 10
        else:
            raise ValueError("Invalid dataset specified")

        self.train, self.test = get_data(
            train_path, test_path, multistep=self.config.multistep, classification=classification
        )

    def init_experiment(self):
        if self.config.data in ["sunspot", "rossler", "henon", "lazer"]:  # Regression problems
            if self.config.multistep:
                self.config.feat_d = 5
                self.config.out_d = 10
                self.config.hidden_d = 10
            else:
                self.config.feat_d = 4
                self.config.out_d = 1
                self.config.hidden_d = 10

            self.load_data()
            self.init_regression()
            self.config.classification = False
        elif self.config.data in [
            "ionosphere",
            "cancer",
            "bank",
            "pendigit",
        ]:  # Classification problem
            self.load_data(classification=True)
            self.init_classification()
            self.config.classification = True
        else:
            raise ValueError("Invalid dataset specified")

    def init_regression(self):
        self.g_func = lambda y, yhat: 2 * (yhat - y)
        self.lambda_func = lambda y, fx, Fx: torch.sum(fx * (y - Fx), 0) / torch.sum(fx * fx, 0)
        self.loss_func = nn.MSELoss()
        self.acc_func = root_mse

        self.log_likelihood_func = Regression.log_likelihood
        self.prior_func = Regression.prior_likelihood

        self.c0 = np.mean(self.train.label, axis=0)

    def init_classification(self):
        # self.g_func = lambda y, yhat: -1 * (2 * y) / (1 + torch.exp(2 * y * yhat))
        # self.h_func = lambda y, yhat: (4 * torch.exp(2 * y * yhat)) / (
        #     1 + torch.exp(2 * y * yhat) ** 2
        # )

        # self.lambda_func = lambda y, fx, Fx: torch.mean(
        #     -1 / fx * self.g_func(y, Fx) / self.h_func(y, Fx)
        # )

        self.g_func = lambda y, yhat: 2 * (yhat - y)
        self.lambda_func = lambda y, fx, Fx: torch.sum(fx * (y - Fx), 0) / torch.sum(fx * fx, 0)

        self.loss_func = nn.MSELoss()
        self.acc_func = auc_score

        self.log_likelihood_func = Regression.log_likelihood
        self.prior_func = Regression.prior_likelihood

        self.c0 = init_gbnn(self.train)

    def run_experiment(self):
        """
        Runs the full experiment and prints out accuracy statistics after each stage (boosting level).
        """

        if self.config.plot_graphs:
            self.run(self.config.num_nets)
            return

        for level in range(1, config.num_nets + 1):

            t0 = time.time()

            tr_score_l = []
            te_score_l = []
            accepted_l = []
            all_chains = []
            for _ in range(config.exps):

                tr_rmse, te_rmse, accepted, chains = self.run(level)
                tr_score_l.append(tr_rmse)
                te_score_l.append(te_rmse)
                accepted_l.append(accepted)
                all_chains.append(chains)

            total_time = time.time() - t0

            # Compute convergence diagnostics
            if self.config.mcmc:
                converge_rate = gr_convergence_rate(all_chains, self.config)
                # print(converge_rate)

            print(f"Boosting level: {level}")
            if self.config.classification:
                print(
                    f"Training statistics: mean - {np.mean(tr_score_l)}, std = {np.std(tr_score_l)}, best = {np.amax(tr_score_l)}"
                )
                print(
                    f"Test statistics: mean - {np.mean(te_score_l)}, std = {np.std(te_score_l)}, best = {np.amax(te_score_l)}"
                )
                print(f"Accept percentage: {np.mean(accepted_l)}")
            else:
                print(
                    f"Training statistics: mean - {np.mean(tr_score_l)}, std = {np.std(tr_score_l)}, best = {np.amin(tr_score_l)}"
                )
                print(
                    f"Test statistics: mean - {np.mean(te_score_l)}, std = {np.std(te_score_l)}, best = {np.amin(te_score_l)}"
                )
                print(f"Accept percentage: {np.mean(accepted_l)}")
            print(f"Total elapsed time: {total_time}")

    def langevin_gradient(self, model, x, y, w, optimizer):

        model.decode(w)

        # for xi, yi in zip(x, y):
        #     out = model(xi)
        #     loss = self.loss_func(out, yi)

        #     model.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        out = model(x)
        loss = self.loss_func(out, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        return model.encode()

    def compute_accuracy(self, fx, Fx, data):

        with torch.no_grad():
            # Calculate gamma
            x = torch.tensor(data.feat, dtype=torch.float32).cuda()
            y = torch.tensor(data.label, dtype=torch.float32).cuda()

            gamma = self.lambda_func(y, fx, Fx)

            pred = Fx + gamma * fx
            ret = self.acc_func(None, data, out=pred.cpu().numpy())

            return ret

    def train_mcmc(self, net_ensemble, model):

        optimizer = get_optim(model.parameters(), self.config)
        # net_ensemble.to_train()  # Set the models in ensemble net to train mode

        x, y = self.train.feat, self.train.label
        x_test, y_test = self.test.feat, self.test.label
        if config.cuda:
            x = torch.as_tensor(x, dtype=torch.float32).cuda()
            y = torch.as_tensor(y, dtype=torch.float32).cuda()

            x_test = torch.as_tensor(x_test, dtype=torch.float32).cuda()
            y_test = torch.as_tensor(y_test, dtype=torch.float32).cuda()

        out = net_ensemble.forward(x)
        grad_direction = -1 * self.g_func(y, out)

        w_size = (
            self.config.feat_d * self.config.hidden_d
            + self.config.hidden_d * self.config.out_d
            + self.config.hidden_d
            + self.config.out_d
        )

        # Randomwalk Steps
        if self.config.classification:
            step_w = self.config.step_w["classification"]
        else:
            step_w = self.config.step_w["regression"]

        sigma_squared = step_w * 10
        nu_1 = 0
        nu_2 = 0
        # step_eta = self.config.step_eta

        # w = model.encode()
        w = np.random.normal(0, sigma_squared, w_size)
        model.decode(w)

        pred_train = model.evaluate_proposal(x, w)
        eta = np.log(np.var((pred_train - grad_direction).cpu().numpy()))
        tau_pro = np.exp(eta)

        prior_current = self.prior_func(
            sigma_squared, nu_1, nu_2, w, tau_pro, self.config
        )  # takes care of the gradients

        log_lik, fx_train = self.log_likelihood_func(model, x, grad_direction, w, tau_sq=tau_pro)
        _, fx_test = self.log_likelihood_func(model, x_test, y_test, w, tau_sq=tau_pro)

        with torch.no_grad():
            Fx_train = torch.as_tensor(net_ensemble.forward(x), dtype=torch.float32).cuda()
            Fx_test = torch.as_tensor(net_ensemble.forward(x_test), dtype=torch.float32).cuda()

        best_w = w
        best_rmse = -1
        best_log_likelihood = -1
        accepted = 0
        chains = np.zeros((self.config.samples, w_size))
        for i in range(self.config.samples):

            # Transition to SGD after burn in
            if i == self.config.burn_in:
                optimizer = SGD(model.parameters(), self.config.lr)

            lx = random.uniform(0, 1)
            if self.config.langevin_gradients and lx < self.config.lg_rate:
                w_gd = self.langevin_gradient(model, x, grad_direction, w, optimizer)
                w_proposal = np.random.normal(w_gd, step_w, w_size)
                w_prop_gd = self.langevin_gradient(model, x, grad_direction, w_proposal, optimizer)

                wc_delta = w - w_prop_gd
                wp_delta = w_proposal - w_gd

                sigma_sq = step_w

                first = -0.5 * np.sum(wc_delta ** 2) / sigma_sq
                second = -0.5 * np.sum(wp_delta ** 2) / sigma_sq

                diff_prop = first - second
            else:
                diff_prop = 0
                w_proposal = np.random.normal(w, step_w, w_size)

            # Fix eta
            eta_pro = eta
            # eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = np.exp(eta_pro)

            with torch.no_grad():
                log_lik_proposal, fx_train_prop = self.log_likelihood_func(
                    model, x, grad_direction, w_proposal, tau_pro
                )

                if self.config.plot_graphs:
                    _, fx_test_prop = self.log_likelihood_func(
                        model, x_test, y_test, w_proposal, tau_pro
                    )

            prior_prop = self.prior_func(
                sigma_squared, nu_1, nu_2, w_proposal, tau_pro, self.config
            )  # takes care of the gradients

            diff_prior = prior_prop - prior_current
            diff_likelihood = log_lik_proposal - log_lik

            # try:
            #     mh_prob = min(1, math.exp())
            # except OverflowError:
            #     mh_prob = 1
            temp = diff_likelihood + diff_prior + diff_prop
            if temp > 0:
                mh_prob = 1
            else:
                mh_prob = math.exp(temp)

            # mh_prob = min(1, math.exp(diff_likelihood + diff_prior + diff_prop))

            u = random.uniform(0, 1)
            if u < mh_prob:  # Accept
                accepted += 1
                log_lik = log_lik_proposal
                prior_current = prior_prop
                w = w_proposal

                eta = eta_pro

                if self.config.plot_graphs:
                    fx_train = fx_train_prop
                    fx_test = fx_test_prop

                # temp = mse_torch(model, x, grad_direction)
                temp = self.loss_func(fx_train_prop, grad_direction)
                if best_rmse == -1 or temp < best_rmse:
                    best_rmse = temp
                    best_w = w
                    best_log_likelihood = log_lik

            chains[i] = w
            # Append diagnostic results
            if self.config.plot_graphs:

                self.likelihoods.append(log_lik)
                self.tr_accs.append(self.compute_accuracy(fx_train, Fx_train, self.train))
                self.te_accs.append(self.compute_accuracy(fx_test, Fx_test, self.test))

        model.decode(best_w)

        return (
            accepted / self.config.samples * 100,
            best_log_likelihood,
            chains[self.config.burn_in :],  # Discard burn in samples
        )

    def train_backprop(self, net_ensemble, model):
        optimizer = get_optim(model.parameters(), self.config)
        net_ensemble.to_train()  # Set the models in ensemble net to train mode

        for i, (x, y) in enumerate(self.train):
            if config.cuda:
                x = torch.as_tensor(x, dtype=torch.float32).cuda()
                y = torch.as_tensor(y, dtype=torch.float32).cuda()

            out = net_ensemble.forward(x)

            grad_direction = -1 * self.g_func(y, out)
            out = model(x)

            loss = self.loss_func(out, grad_direction)

            model.zero_grad()
            loss.backward()
            optimizer.step()

    def run(self, num_nets):
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

        net_ensemble = EnsembleNet(self.c0, self.config.lr)

        self.train.shuffle()

        self.likelihoods = []
        self.tr_accs = []
        self.te_accs = []
        final_tr_score = 0
        final_te_score = 0
        accepted = 0

        chains = None
        for stage in range(num_nets):

            # self.config.step_w["classification"] *= 0.5
            # self.config.step_w["regression"] *= 0.5

            model = model_type.get_model(config)
            if config.cuda:
                model.cuda()

            log_likelihood = None

            if config.mcmc:
                temp, log_likelihood, chains = self.train_mcmc(net_ensemble, model)
                accepted += temp
            else:
                self.train_backprop(net_ensemble, model)

            # Calculate gamma
            x = torch.tensor(self.train.feat, dtype=torch.float32).cuda()
            y = torch.tensor(self.train.label, dtype=torch.float32).cuda()

            fx = model(x)
            Fx = net_ensemble.forward(x)
            Fx = torch.as_tensor(Fx, dtype=torch.float32).cuda()

            gamma = self.lambda_func(y, fx, Fx)
            # print(gamma)

            net_ensemble.add(model, gamma, log_likelihood=log_likelihood)

            if config.cuda:
                net_ensemble.to_cuda()
            net_ensemble.to_eval()  # Set the models in ensemble net to eval mode

            final_tr_score = self.acc_func(net_ensemble, self.train)
            final_te_score = self.acc_func(net_ensemble, self.test)

        # Plot results
        if self.config.mcmc and self.config.plot_graphs:
            self.save_plots(num_nets)

        return final_tr_score, final_te_score, accepted / num_nets, chains

    def save_plots(self, n):
        # Make folders
        path = f"plots/{self.config.data}/"
        Path(path).mkdir(parents=True, exist_ok=True)

        # Likelihoods for all weak learners
        for i in range(n):
            plt.figure()
            plt.plot(
                list(range(1, self.config.samples + 1)),
                self.likelihoods[i * self.config.samples : (i + 1) * self.config.samples],
            )
            plt.xlabel("Samples")
            plt.ylabel("Log-likelihoods")
            plt.savefig(path + f"likelihoods_{i}.png")

        # Training acc
        plt.figure()
        plt.plot(
            list(range(1, self.config.samples * n + 1)), self.tr_accs,
        )
        plt.xlabel("Samples")
        plt.ylabel("Train score")
        plt.savefig(path + f"train_scores.png")

        # Test acc
        # Training acc
        plt.figure()
        plt.plot(
            list(range(1, self.config.samples * n + 1)), self.te_accs,
        )
        plt.xlabel("Samples")
        plt.ylabel("Test score")
        plt.savefig(path + f"test_scores.png")


if __name__ == "__main__":

    exp = Experiment(config)
    exp.init_experiment()
    exp.run_experiment()


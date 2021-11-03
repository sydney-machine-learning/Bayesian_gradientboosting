import math
import random
import time
from multiprocessing import Value
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import grad

from data.data import LibCSVData, LibTXTData
from functions import Regression
from models.ensemble_net import EnsembleNet
from models.mlp import MLP_1HL
from parallel_tempering import ParallelTempering
from utils import (
    auc_score,
    classification_score,
    get_optim,
    gr_convergence_rate,
    init_gbnn,
    mse_torch,
    root_mse,
)

with open("config.yaml", "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


config = Config(**data)
if config.mcmc:
    config.params = Config(**config.mcmc_params)
else:
    config.params = Config(**config.backprop_params)


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
        self.g_func = Regression.g_func
        self.lambda_func = Regression.lambda_func
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

        self.g_func = Regression.g_func
        self.lambda_func = Regression.lambda_func

        self.loss_func = nn.MSELoss()
        # self.acc_func = auc_score
        self.acc_func = classification_score

        self.log_likelihood_func = Regression.log_likelihood
        self.prior_func = Regression.prior_likelihood

        self.c0 = init_gbnn(self.train)

    def run_experiment(self):
        """
        Runs the full experiment and prints out accuracy statistics after each stage (boosting level).
        """

        # Plot all diagnostic graphs
        if self.config.plot_graphs:
            self.run_sequential(self.config.num_nets)
            return

        for level in range(1, config.num_nets + 1):

            t0 = time.time()

            if config.mcmc:
                if config.parallel_tempering:
                    pt = ParallelTempering(self, self.config, level)
                    pt.init_directories()
                    pt.init_chains()
                    tr_score_l, te_score_l, accepted, chains = pt.run_chains()
                elif config.simultaneous:
                    tr_score_l, te_score_l, accepted, chains = self.run_simultaneous(level)
                else:
                    tr_score_l, te_score_l, accepted, chains = self.run_sequential(level)
            else:
                tr_score_l = []
                te_score_l = []
                for _ in range(config.params.exps):

                    tr_score, te_score, _, _ = self.run_sequential(level)
                    tr_score_l.append(tr_score)
                    te_score_l.append(te_score)

            total_time = time.time() - t0

            # Compute convergence diagnostics
            # if self.config.mcmc:
            # converge_rate = gr_convergence_rate(all_chains, self.config)
            # print(converge_rate)

            print(f"Boosting level: {level}")
            if self.config.classification:
                print(
                    f"Training statistics: mean - {np.mean(tr_score_l)}, std = {np.std(tr_score_l)}, best = {np.amax(tr_score_l)}"
                )
                print(
                    f"Test statistics: mean - {np.mean(te_score_l)}, std = {np.std(te_score_l)}, best = {np.amax(te_score_l)}"
                )
                if config.mcmc:
                    print(f"Accept percentage: {accepted}")
            else:
                print(
                    f"Training statistics: mean - {np.mean(tr_score_l)}, std = {np.std(tr_score_l)}, best = {np.amin(tr_score_l)}"
                )
                print(
                    f"Test statistics: mean - {np.mean(te_score_l)}, std = {np.std(te_score_l)}, best = {np.amin(te_score_l)}"
                )
                if config.mcmc:
                    print(f"Accept percentage: {accepted}")
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

            return ret, pred

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
            step_w = self.config.params.step_w["classification"]
        else:
            step_w = self.config.params.step_w["regression"]

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
        chains = np.zeros((self.config.params.samples - self.config.params.burn_in, w_size))
        Fx_test_samples = torch.zeros(
            (self.config.params.samples, y_test.shape[0], y_test.shape[1])
        )
        likelihoods = np.zeros((self.config.params.samples))
        for i in range(self.config.params.samples):

            lx = random.uniform(0, 1)
            if self.config.params.langevin_gradients and lx < self.config.params.lg_rate:
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

                _, fx_test_prop = self.log_likelihood_func(
                    model, x_test, y_test, w_proposal, tau_pro
                )

            prior_prop = self.prior_func(
                sigma_squared, nu_1, nu_2, w_proposal, tau_pro, self.config
            )  # takes care of the gradients

            diff_prior = prior_prop - prior_current
            diff_likelihood = log_lik_proposal - log_lik

            temp = diff_likelihood + diff_prior + diff_prop
            if temp > 0:
                mh_prob = 1
            else:
                mh_prob = math.exp(temp)

            u = random.uniform(0, 1)
            if u < mh_prob:  # Accept
                accepted += 1
                log_lik = log_lik_proposal
                prior_current = prior_prop
                w = w_proposal

                eta = eta_pro

                fx_train = fx_train_prop
                fx_test = fx_test_prop

                # temp = mse_torch(model, x, grad_direction)
                temp = self.loss_func(fx_train_prop, grad_direction)
                if best_rmse == -1 or temp < best_rmse:
                    best_rmse = temp
                    best_w = w
                    best_log_likelihood = log_lik

            if i >= self.config.params.burn_in:
                chains[i - self.config.params.burn_in] = w

            # Append diagnostic results
            # if self.config.plot_graphs:

            acc_tr, _ = self.compute_accuracy(fx_train, Fx_train, self.train)
            acc_te, pred_te = self.compute_accuracy(fx_test, Fx_test, self.test)
            # if i >= self.config.burn_in:
            Fx_test_samples[i] = pred_te.cpu().numpy()

            likelihoods[i] = log_lik

            self.tr_accs.append(acc_tr)
            self.te_accs.append(acc_te)

        model.decode(best_w)
        self.likelihoods.append(likelihoods)

        self.Fx_test_l.append(Fx_test_samples)

        return (
            accepted / self.config.params.samples * 100,
            best_log_likelihood,
            chains,
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

    def run_sequential(self, num_nets):
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

        self.net_ensemble = EnsembleNet(self.c0, self.config.params.lr)

        self.train.shuffle()

        self.likelihoods = []
        self.tr_accs = []
        self.te_accs = []
        self.Fx_test_l = []
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
                temp, log_likelihood, chains = self.train_mcmc(self.net_ensemble, model)
                accepted += temp
            else:
                self.train_backprop(self.net_ensemble, model)

            # Calculate gamma
            x = torch.tensor(self.train.feat, dtype=torch.float32).cuda()
            y = torch.tensor(self.train.label, dtype=torch.float32).cuda()

            fx = model(x)
            Fx = self.net_ensemble.forward(x)
            Fx = torch.as_tensor(Fx, dtype=torch.float32).cuda()

            gamma = self.lambda_func(y, fx, Fx)
            # print(gamma)

            self.net_ensemble.add(model, gamma)

            if config.cuda:
                self.net_ensemble.to_cuda()
            self.net_ensemble.to_eval()  # Set the models in ensemble net to eval mode

            final_tr_score = self.acc_func(self.net_ensemble, self.train)
            final_te_score = self.acc_func(self.net_ensemble, self.test)

        # Plot results
        if self.config.mcmc and self.config.plot_graphs:
            self.save_plots(num_nets)

        if self.config.mcmc:
            return (
                self.tr_accs[-(self.config.params.samples - self.config.params.burn_in) :],
                self.te_accs[-(self.config.params.samples - self.config.params.burn_in) :],
                accepted / num_nets,
                chains,
            )
        else:
            return final_tr_score, final_te_score, accepted / num_nets, chains

    def run_simultaneous(self, num_nets):

        model_type = MLP_1HL

        self.net_ensemble = EnsembleNet(self.c0, self.config.params.lr)
        self.train.shuffle()

        x, y = self.train.feat, self.train.label
        x_test, y_test = self.test.feat, self.test.label
        if config.cuda:
            x = torch.as_tensor(x, dtype=torch.float32).cuda()
            y = torch.as_tensor(y, dtype=torch.float32).cuda()

            x_test = torch.as_tensor(x_test, dtype=torch.float32).cuda()
            y_test = torch.as_tensor(y_test, dtype=torch.float32).cuda()

        out = self.net_ensemble.forward(x)
        grad_direction = -1 * self.g_func(y, out)

        w_size = (
            self.config.feat_d * self.config.hidden_d
            + self.config.hidden_d * self.config.out_d
            + self.config.hidden_d
            + self.config.out_d
        )

        # Randomwalk Steps
        if self.config.classification:
            step_w = self.config.params.step_w["classification"]
        else:
            step_w = self.config.params.step_w["regression"]

        sigma_squared = step_w * 10
        nu_1 = 0
        nu_2 = 0
        # step_eta = self.config.step_eta

        # Initialize weights for all models
        model_weights = np.zeros((num_nets, w_size))
        models = []
        eta_list = np.zeros((num_nets))
        prior_list = np.zeros((num_nets))
        self.likelihoods = np.zeros((num_nets, self.config.params.samples + 1))
        fx_train_list = torch.zeros((num_nets, len(y), self.config.out_d)).cuda()
        fx_test_list = torch.zeros((num_nets, len(y_test), self.config.out_d)).cuda()
        self.Fx_test_l = torch.zeros(
            (num_nets, self.config.params.samples, len(y_test), self.config.out_d)
        ).cuda()

        self.tr_accs = []
        self.te_accs = []
        for i in range(num_nets):
            model_weights[i] = np.random.normal(0, sigma_squared, w_size)
            model = model_type.get_model(self.config)
            if config.cuda:
                model.cuda()
            model.decode(model_weights[i])
            models.append(model)

            pred_train = model.evaluate_proposal(x, model_weights[i])
            eta_list[i] = np.log(np.var((pred_train - grad_direction).cpu().numpy()))
            tau_pro = np.exp(eta_list[i])

            prior_list[i] = self.prior_func(
                sigma_squared, nu_1, nu_2, model_weights[i], tau_pro, self.config
            )  # takes care of the gradients

            self.likelihoods[i][0], fx_train_list[i] = self.log_likelihood_func(
                model, x, grad_direction, model_weights[i], tau_sq=tau_pro
            )
            _, fx_test_list[i] = self.log_likelihood_func(
                model, x_test, y_test, model_weights[i], tau_sq=tau_pro
            )

        accepted = 0
        chains = np.zeros((self.config.params.samples - self.config.params.burn_in, w_size))
        for i in range(self.config.params.samples):

            self.net_ensemble.reset()
            with torch.no_grad():
                Fx_train = torch.as_tensor(
                    self.net_ensemble.forward(x), dtype=torch.float32
                ).cuda()
                Fx_test = torch.as_tensor(
                    self.net_ensemble.forward(x_test), dtype=torch.float32
                ).cuda()

            for j in range(num_nets):

                out = self.net_ensemble.forward(x)
                grad_direction = -1 * self.g_func(y, out)

                model = models[j]
                w = model_weights[j]
                optimizer = get_optim(model.parameters(), self.config)

                lx = random.uniform(0, 1)
                if self.config.params.langevin_gradients and lx < self.config.params.lg_rate:
                    w_gd = self.langevin_gradient(model, x, grad_direction, w, optimizer)
                    w_proposal = np.random.normal(w_gd, step_w, w_size)
                    w_prop_gd = self.langevin_gradient(
                        model, x, grad_direction, w_proposal, optimizer
                    )

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
                eta_pro = eta_list[j]
                # eta_pro = eta + np.random.normal(0, step_eta, 1)
                tau_pro = np.exp(eta_pro)

                with torch.no_grad():
                    log_lik_proposal, fx_train_prop = self.log_likelihood_func(
                        model, x, grad_direction, w_proposal, tau_pro
                    )

                    _, fx_test_prop = self.log_likelihood_func(
                        model, x_test, y_test, w_proposal, tau_pro
                    )

                prior_prop = self.prior_func(
                    sigma_squared, nu_1, nu_2, w_proposal, tau_pro, self.config
                )  # takes care of the gradients

                diff_prior = prior_prop - prior_list[j]
                diff_likelihood = log_lik_proposal - self.likelihoods[j][i]

                temp = diff_likelihood + diff_prior + diff_prop
                if temp > 0:
                    mh_prob = 1
                else:
                    mh_prob = math.exp(temp)

                u = random.uniform(0, 1)
                if u < mh_prob:  # Accept
                    accepted += 1
                    self.likelihoods[j][i + 1] = log_lik_proposal
                    prior_list[j] = prior_prop
                    model_weights[j] = w_proposal
                    eta_list[j] = eta_pro
                    fx_train_list[j] = fx_train_prop
                    fx_test_list[j] = fx_test_prop
                else:
                    self.likelihoods[j][i + 1] = self.likelihoods[j][i]

                if i >= self.config.params.burn_in:
                    chains[i - self.config.params.burn_in] = w

                acc_tr, _ = self.compute_accuracy(fx_train_list[j], Fx_train, self.train)
                acc_te, pred_te = self.compute_accuracy(fx_test_list[j], Fx_test, self.test)

                if j == num_nets - 1:
                    self.tr_accs.append(acc_tr)
                    self.te_accs.append(acc_te)

                # Add model to the ensemble
                gamma = self.lambda_func(y, fx_train_list[j], Fx_train)

                self.net_ensemble.add(model, gamma)

                Fx_train += gamma * fx_train_list[j]
                Fx_test += gamma * fx_test_list[j]

                self.Fx_test_l[j][i] = Fx_test

        if self.config.plot_graphs:
            self.save_plots(num_nets)

        return (
            self.tr_accs[-(self.config.params.samples - self.config.params.burn_in) :],
            self.te_accs[-(self.config.params.samples - self.config.params.burn_in) :],
            accepted / (self.config.params.samples * num_nets) * 100,
            chains,
        )

    def save_plots(self, n):
        # Make folders
        path = f"plots/{self.config.data}/"
        Path(path).mkdir(parents=True, exist_ok=True)

        # Font size
        matplotlib.rcParams.update({"font.size": 14})
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        # Likelihoods for all weak learners
        for i in range(n):
            plt.figure()
            plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

            plt.plot(
                list(range(1, self.likelihoods[i].shape[0] + 1)), self.likelihoods[i],
            )
            plt.xlabel("Samples")
            plt.ylabel("Log-likelihoods")
            plt.savefig(path + f"likelihoods_{i}.png")

        # Training acc
        plt.figure()
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.plot(
            list(range(1, len(self.tr_accs) + 1)), self.tr_accs,
        )
        plt.xlabel("Samples")
        plt.ylabel("Train score")
        plt.savefig(path + f"train_scores.png")

        # Test acc
        # Training acc
        plt.figure()
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.plot(
            list(range(1, len(self.te_accs) + 1)), self.te_accs,
        )
        plt.xlabel("Samples")
        plt.ylabel("Test score")
        plt.savefig(path + f"test_scores.png")

        if self.config.classification:
            return

        x_test = np.linspace(1, len(self.test.label), num=len(self.test.label))
        for i in range(len(self.Fx_test_l)):
            Fx_test = self.Fx_test_l[i].cpu().numpy()
            # Uncertainty graphs
            fx_mu = Fx_test.mean(axis=0).squeeze()
            fx_high = np.percentile(Fx_test, 99.5, axis=0).squeeze()
            fx_low = np.percentile(Fx_test, 0.5, axis=0).squeeze()

            plt.figure()
            # plt.plot(x_test, fx_low, "w", label="pred. (2.5 percen.)")
            # plt.plot(x_test, fx_high, "w", label="pred. (97.5 percen.)")
            plt.fill_between(
                x_test, fx_low, fx_high, facecolor="b", alpha=0.5, label="pred. (99 percen.)"
            )
            plt.plot(x_test, self.test.label, "r", label="actual", linewidth=0.5)
            plt.plot(x_test, fx_mu, "b", label="pred. (mean)", linewidth=0.5)

            plt.legend(loc="upper right")
            plt.xlabel("Time")
            plt.ylabel("Prediction")
            plt.savefig(path + f"uncertainty_{i}.png")


if __name__ == "__main__":

    if config.plot_graphs:

        # config.mcmc = False
        # exp_backprop = Experiment(config).init_experiment()
        # exp_backprop.run(config.num_nets)

        # config.mcmc = True
        exp_mcmc = Experiment(config)
        exp_mcmc.init_experiment()

        if config.simultaneous:
            exp_mcmc.run_simultaneous(config.num_nets)
        else:
            exp_mcmc.run_sequential(config.num_nets)
    else:
        exp = Experiment(config)
        exp.init_experiment()
        exp.run_experiment()

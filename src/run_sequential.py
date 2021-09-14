import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim import SGD, Adam

from data.data import LibTXTData
from data.sparseloader import DataLoader
from models.ensemble_net import EnsembleNet
from models.mlp import MLP_1HL, MLP_2HL

with open("config.yaml", "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


config = Config(**data["params"])


def get_data(train_path, test_path):

    train = LibTXTData(train_path, config)
    test = LibTXTData(test_path, config)

    scaler = StandardScaler()
    scaler.fit(train.feat)
    train.feat = scaler.transform(train.feat)
    test.feat = scaler.transform(test.feat)

    return train, test


def get_optim(params, lr):
    optimizer = Adam(params, lr, weight_decay=0.0003)
    # optimizer = SGD(params, lr, weight_decay=weight_decay)
    return optimizer


def root_mse(net_ensemble, loader):
    loss = 0
    total = 0

    for x, y in loader:

        if config.cuda:
            x = x.cuda()

        with torch.no_grad():
            out = net_ensemble.forward(x)

        y = y.cpu().numpy().reshape(len(y), 1)

        out = out.cpu().numpy().reshape(len(y), 1)
        loss += mean_squared_error(y, out) * len(y)
        total += len(y)
    return np.sqrt(loss / total)


def run_experiment(train, test, num_nets):
    model_type = MLP_1HL

    c0 = np.mean(train.label)
    net_ensemble = EnsembleNet(c0, config.lr)

    train_loader = DataLoader(train, 1, shuffle=True, drop_last=False, num_workers=2)
    test_loader = DataLoader(test, 1, shuffle=False, drop_last=False, num_workers=2)

    loss_f = nn.MSELoss()

    final_tr_rmse = 0
    final_te_rmse = 0
    total_time = 0
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

            grad_direction = -2 * (out - y)

            out = model(x)
            out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
            loss = loss_f(out, grad_direction)  # T

            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate gamma
        x = torch.tensor(train.feat).cuda()
        y = torch.tensor(train.label).cuda()
        fx = model(x)

        gamma = torch.sum(fx * (y - net_ensemble.forward(x))) / torch.sum(fx * fx)

        net_ensemble.add(model, gamma)

        if config.cuda:
            net_ensemble.to_cuda()
        net_ensemble.to_eval()  # Set the models in ensemble net to eval mode

        # Train
        tr_rmse = root_mse(net_ensemble, train_loader)
        final_tr_rmse = tr_rmse

        te_rmse = root_mse(net_ensemble, test_loader)
        final_te_rmse = te_rmse

        # print(
        #     f"Stage: {stage}  RMSE@Tr: {tr_rmse:.5f}, RMSE@Val: {val_rmse:.5f}, RMSE@Te: {te_rmse:.5f}"
        # )

    # print(f"Stage: {num_nets}  RMSE@Tr: {tr_rmse:.5f}, final RMSE@Te: {te_rmse:.5f}")

    return final_tr_rmse, final_te_rmse, total_time


if __name__ == "__main__":

    if config.data == "sunspot":
        train_path = "datasets/Sunspot/train.txt"
        test_path = "datasets/Sunspot/test.txt"

        config.feat_d = 4
        config.hidden_d = 10
        config.out_d = 1

    train, test = get_data(train_path, test_path)

    for level in range(1, config.num_nets + 1):

        t0 = time.time()

        tr_rmse_l = []
        te_rmse_l = []
        for _ in range(config.exps):
            tr_rmse, te_rmse, runtime = run_experiment(train, test, level)
            tr_rmse_l.append(tr_rmse)
            te_rmse_l.append(te_rmse)

        total_time = time.time() - t0

        print(f"Boosting level: {level}")
        print(
            f"Training statistics: mean - {np.mean(tr_rmse_l)}, std = {np.std(tr_rmse_l)}, best = {np.amin(tr_rmse_l)}"
        )
        print(
            f"Test statistics: mean - {np.mean(te_rmse_l)}, std = {np.std(te_rmse_l)}, best = {np.amin(te_rmse_l)}"
        )
        print(f"Total elapsed time: {total_time}")

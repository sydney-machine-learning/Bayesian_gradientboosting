import math

import numpy as np
import torch
from scipy.optimize import line_search
from scipy.special import softmax

from utils import rmse_torch


class Classification:
    def g_func(y, yhat):
        return torch.nn.functional.softmax(yhat, dim=-1) - y

    def obj_func(x):
        torch.log(1 + torch.exp())

    def lambda_func(y, fx, Fx):

        return 1

        # y = y.cpu().numpy()
        # fx = fx.cpu().numpy()
        # Fx = Fx.cpu().numpy()
        # obj_func = lambda g: np.mean(np.log(1 + np.exp(Fx + g * fx)) - y * (Fx + g * fx))
        # obj_grad = lambda g: np.mean(np.exp(Fx + g * fx) / (1 + np.exp(Fx + g * fx)) - y * fx)
        # ret = line_search(obj_func, obj_grad, 0.1, 0.1, maxiter=100)
        # alpha, _, _, _, _, _ = ret
        # print(ret)
        # return alpha

    def log_likelihood(model, x, y, w, tau_sq):

        fx = model.evaluate_proposal(x, w)

        if y.ndim > 1:
            n = (
                y.shape[0] * y.shape[1]
            )  # number of samples x number of outputs (prediction horizon)
        else:
            n = y.shape[0]

        y = y.cpu()
        probs = softmax(fx.cpu(), axis=-1)
        result = 0
        # for i in range(y.shape[0]):
        #     true = np.argmax(y[i])
        #     result += probs[i, true]

        result = np.sum(y * probs)

        return result, fx

    def prior_likelihood(sigma_squared, nu_1, nu_2, w, tausq, config):
        pass


class Regression:
    def g_func(y, yhat):
        return 2 * (yhat - y)

    def lambda_func(y, fx, Fx):
        return torch.sum(fx * (y - Fx), 0) / torch.sum(fx * fx, 0)

    def log_likelihood(model, x, y, w, tau_sq):

        fx = model.evaluate_proposal(x, w)

        # if y.ndim > 1:
        #     n = (
        #         y.shape[0] * y.shape[1]
        #     )  # number of samples x number of outputs (prediction horizon)
        # else:
        #     n = y.shape[0]
        y = y.cpu().numpy()
        fx_np = fx.cpu().numpy()

        # p1 = -(n / 2) * np.log(2 * math.pi * tau_sq)
        # p2 = 1 / (2 * tau_sq)

        # result = p1 - (p2 * np.sum(np.square((y - fx).cpu().numpy())))

        result = np.sum(-0.5 * np.log(2 * math.pi * tau_sq) - 0.5 * np.square(y - fx_np) / tau_sq)
        result = np.sum(result)

        return result, fx

    def prior_likelihood(sigma_squared, nu_1, nu_2, w, tausq, config):
        h = config.hidden_d  # number hidden neurons
        d = config.feat_d  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (np.sum(np.square(w)))

        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)

        return log_loss

        # part1 = -1 * ((len(w)) / 2) * np.log(sigma_squared)
        # part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        # log_loss = part1 - part2
        # return log_loss

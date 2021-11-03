import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from torch.optim import SGD, Adam


def get_optim(params, config):
    if config.params.optimizer == "adam":
        return Adam(params, config.params.lr)
    elif config.params.optimizer == "sgd":
        return SGD(params, config.params.lr)
    else:
        raise ValueError("Invalid optimizer specified, choose adam or sgd.")


def mse_torch(model, x, y):
    with torch.no_grad():
        return ((model(x) - y).cpu().numpy() ** 2).mean()


def rmse_torch(model, x, y):
    with torch.no_grad():
        return np.sqrt(((model(x) - y).cpu().numpy() ** 2).mean())


def root_mse(net_ensemble, data, cuda=True, out=None):

    if cuda:
        x = torch.as_tensor(data.feat, dtype=torch.float32).cuda()
    y = data.label

    if out is None:
        with torch.no_grad():
            out = net_ensemble.forward(x).cpu().numpy()
    return mean_squared_error(y, out, squared=False)


def classification_score(net_ensemble, data, cuda=True, out=None):

    x, y = data.feat, data.label
    if cuda:
        x = torch.as_tensor(x, dtype=torch.float32).cuda()

    if out is None:
        with torch.no_grad():
            out = net_ensemble.forward(x).cpu().numpy()

    out = np.argmax(softmax(out, axis=-1), axis=-1)
    y = np.argmax(y, axis=-1)

    score = accuracy_score(y, out)

    return score


def auc_score(net_ensemble, data, cuda=True, out=None):

    x, y = data.feat, data.label
    if cuda:
        x = torch.as_tensor(x, dtype=torch.float32).cuda()

    if out is None:
        with torch.no_grad():
            out = net_ensemble.forward(x).cpu().numpy()

    out = softmax(out, axis=-1)
    score = roc_auc_score(y, out)

    return score


def init_gbnn(train):
    data = torch.as_tensor(train.label, dtype=torch.float32).cuda()
    totals = torch.sum(data, 0)
    probs = torch.zeros(data.shape[1])
    probs[torch.argmax(totals)] = 1

    # probs[0] = torch.argmax(totals)
    return probs


def gr_convergence_rate(chains, config):
    # Chains shape is (M, N, weight_size)

    chains = np.array(chains)

    M, N, P = chains.shape

    B_on_n = chains.mean(axis=1).var(axis=0)
    W = chains.var(axis=1).mean(axis=0)

    sig2 = N / (N - 1) * W + B_on_n
    Vhat = sig2 + B_on_n / N
    Rhat = Vhat / W

    # return Rhat

    si2 = chains.var(axis=1)
    xi_bar = chains.mean(axis=1)
    xi2_bar = chains.mean(axis=1) ** 2
    var_si2 = chains.var(axis=1).var(axis=0)
    allmean = chains.mean(axis=1).mean(axis=0)
    cov_term1 = np.array([np.cov(si2[:, i], xi2_bar[:, i])[0, 1] for i in range(P)])
    cov_term2 = np.array(
        [-2 * allmean[i] * (np.cov(si2[:, i], xi_bar[:, i])[0, 1]) for i in range(P)]
    )
    var_Vhat = (
        ((N - 1) / N) ** 2 * 1.0 / M * var_si2
        + ((M + 1) / M) ** 2 * 2.0 / (M - 1) * B_on_n ** 2
        + 2.0 * (M + 1) * (N - 1) / (M * N ** 2) * N / M * (cov_term1 + cov_term2)
    )
    df = 2 * Vhat ** 2 / var_Vhat

    Rhat *= df / (df - 2)

    return Rhat

    # variances = []  # (M, weight_size)
    # means = []  # (M, weight_size)
    # for chain in chains:
    #     variances.append(np.var(chain, axis=0))
    #     means.append(np.mean(chain, axis=0))

    # W = np.mean(variances, axis=0)

    # g_mean = np.mean(means, axis=0)

    # B = N / (M - 1) * np.sum((means - g_mean) ** 2, axis=0)

    # var_hat = (1 - 1 / N) * W + 1 / N * B

    # scale_reduction = np.sqrt(var_hat / W)

    # return scale_reduction

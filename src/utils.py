import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import mean_squared_error, roc_auc_score


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

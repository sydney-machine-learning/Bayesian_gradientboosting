import numpy as np
import torch
from sklearn.metrics import mean_squared_error


def root_mse(net_ensemble, data, cuda=True):
    loss = 0
    total = 0

    for x, y in data:

        if np.isscalar(y):
            out_shape = 1
        else:
            out_shape = y.shape[0]

        if cuda:
            x = torch.as_tensor(x, dtype=torch.float32).cuda()

        with torch.no_grad():
            out = net_ensemble.forward(x)

        y = y.reshape(out_shape, 1)
        out = out.cpu().numpy().reshape(out_shape, 1)

        loss += mean_squared_error(y, out)
        total += 1
    return np.sqrt(loss / total)


def auc_score(net_ensemble, test_loader, cuda=True):
    actual = []
    posterior = []
    for x, y in test_loader:
        if cuda:
            x = torch.as_tensor(x, dtype=torch.float32).cuda()
        with torch.no_grad():
            out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(
            out
        )  # Why not using the scores themselve than converting to prob
        prob = prob.cpu().numpy().tolist()

        if type(prob) is float:
            posterior.append(prob)
            actual.append(y)
        else:
            posterior.extend(prob)
            actual.extend(y.numpy().tolist())
    score = auc(actual, posterior)
    return score


def tied_rank(x):
    """
    Computes the tied rank of elements in x.

    This function computes the tied rank of elements in x.

    Parameters
    ----------
    x : list of numbers, numpy array

    Returns
    -------
    score : list of numbers
            The tied rank f each element in x

    """
    sorted_x = sorted(zip(x, range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank + 1 + i) / 2.0
            last_rank = i
        if i == len(sorted_x) - 1:
            for j in range(last_rank, i + 1):
                r[sorted_x[j][1]] = float(last_rank + i + 2) / 2.0
    return r


def auc(actual, posterior):
    """
    Computes the area under the receiver-operater characteristic (AUC)

    This function computes the AUC error metric for binary classification.

    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.

    Returns
    -------
    score : double
            The mean squared error between actual and posterior

    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x == 1])
    num_negative = len(actual) - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i] == 1])
    auc = (sum_positive - num_positive * (num_positive + 1) / 2.0) / (num_negative * num_positive)
    return auc


def init_gbnn(train):
    positive = negative = 0
    for i in range(len(train)):
        if train[i][1] > 0:
            positive += 1
        else:
            negative += 1
    blind_acc = max(positive, negative) / (positive + negative)
    return float(np.log(positive / negative))

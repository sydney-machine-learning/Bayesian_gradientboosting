import math

import numpy as np


class Classification:

    pass


class Regression:
    def log_likelihood(model, x, y, w, tau_sq):

        fx = model.evaluate_proposal(x, w)

        if y.ndim > 1:
            n = (
                y.shape[0] * y.shape[1]
            )  # number of samples x number of outputs (prediction horizon)
        else:
            n = y.shape[0]

        p1 = -(n / 2) * np.log(2 * math.pi * tau_sq)
        p2 = 1 / 2 * tau_sq

        result = p1 - (p2 * np.sum(np.square((y - fx).cpu().numpy())))

        return result

    def prior_likelihood(sigma_squared, nu_1, nu_2, w, tausq, config):
        h = config.hidden_d  # number hidden neurons
        d = config.feat_d  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (np.sum(np.square(w)))

        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)

        return log_loss

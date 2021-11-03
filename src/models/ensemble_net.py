import numpy as np
import torch
import torch.nn as nn


class EnsembleNet(object):
    def __init__(self, c0, lr):
        self.models = []
        self.gammas = []
        self.c0 = c0
        self.lr = lr

    def add(self, model, gamma):
        self.models.append(model)
        self.gammas.append(gamma)

    def reset(self):
        self.models = []
        self.gammas = []

    def remove(self):
        if not self.models:
            return

        self.gammas.pop()
        return self.models.pop()

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        params.extend(self.gammas)
        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self, x):
        # print(x)
        if x.dim() > 1:
            prediction = np.tile(self.c0, (x.shape[0], 1))
        else:
            prediction = self.c0

        # print(self.c0)
        prediction = torch.as_tensor(prediction, dtype=torch.float32).cuda()
        with torch.no_grad():
            for m, g in zip(self.models, self.gammas):
                prediction += g * m(x)

        return prediction

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
        prediction = self.c0
        with torch.no_grad():
            for m, g in zip(self.models, self.gammas):
                prediction += g * m(x)
        return prediction

    def forward_grad(self, x):
        prediction = self.c0
        for m, g in zip(self.models, self.gammas):
            prediction += g * m(x)
        return prediction

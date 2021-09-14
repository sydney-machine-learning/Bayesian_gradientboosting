import torch
import torch.nn as nn


class EnsembleNet(object):
    def __init__(self, c0, lr):
        self.models = []
        self.c0 = c0
        self.lr = lr

    def add(self, model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        params.append(self.boost_rate)
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
            for m in self.models:
                prediction += m(x)
        return prediction

    def forward_grad(self, x):
        if len(self.models) == 0:
            return None, self.c0
        # at least one model
        middle_feat_cum = None
        prediction = None
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, prediction = m(x, middle_feat_cum)
            else:
                middle_feat_cum, pred = m(x, middle_feat_cum)
                prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

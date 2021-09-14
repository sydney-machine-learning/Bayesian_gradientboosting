import torch
import torch.nn as nn


class MLP_1HL(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_1HL, self).__init__()
        self.in_layer = nn.Linear(dim_in, dim_hidden)
        self.out_layer = nn.Linear(dim_hidden, dim_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.in_layer(x)
        return self.out_layer(out).squeeze()

    @classmethod
    def get_model(cls, config):
        model = MLP_1HL(config.feat_d, config.hidden_d, config.out_d)
        return model


class MLP_2HL(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_2HL, self).__init__()
        self.in_layer = nn.Linear(dim_in, dim_hidden)
        self.hidden_layer = nn.Linear(dim_hidden, dim_hidden)
        self.out_layer = nn.Linear(dim_hidden, dim_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.in_layer(x)
        out = self.hidden_layer(out)
        return self.out_layer(out).squeeze()

    @classmethod
    def get_model(cls, config):
        model = MLP_2HL(config.feat_d, config.hidden_d, config.out_d)
        return model

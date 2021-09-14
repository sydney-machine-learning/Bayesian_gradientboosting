import torch
import torch.nn as nn


class MLP_1HL(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_1HL, self).__init__()
        self.in_layer = nn.Linear(dim_in, dim_hidden)
        self.out_layer = nn.Linear(dim_hidden, dim_out)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.in_layer(x)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, config):
        model = MLP_1HL(config.feat_d, config.hidden_d, config.out_d)
        return model


class MLP_2HL(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_2HL, self).__init__()
        self.in_layer = nn.Linear(dim_in, dim_hidden)
        self.dropout_layer = nn.Dropout(0.0)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden, dim_hidden)
        self.out_layer = nn.Linear(dim_hidden, dim_out)
        self.bn = nn.BatchNorm1d(dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_in)

    def forward(self, x):
        out = self.lrelu(self.in_layer(x))
        self.eval()
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, config):
        model = MLP_2HL(config.feat_d, config.hidden_d, config.out_d)
        return model

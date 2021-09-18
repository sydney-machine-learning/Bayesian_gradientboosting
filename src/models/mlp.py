from collections import OrderedDict

import torch
import torch.nn as nn


class MLP_1HL(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_1HL, self).__init__()
        self.in_layer = nn.Linear(dim_in, dim_hidden)
        self.out_layer = nn.Linear(dim_hidden, dim_out)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.in_layer(x))
        return self.out_layer(out).squeeze()

    def evaluate_proposal(self, x, w):
        with torch.no_grad():

            # Set weights
            self.decode(w)

            return self.forward(x)

    def decode(self, w):

        in_weight, in_bias, out_weight, out_bias = w
        new_state_dict = OrderedDict(
            {
                "in_layer.weight": in_weight,
                "in_layer.bias": in_bias,
                "out_layer.weight": out_weight,
                "out_layer.bias": out_bias,
            }
        )
        self.load_state_dict(new_state_dict, strict=False)

    def encode(self):
        state_dict = self.state_dict()
        return [
            state_dict["in_layer.weight"],
            state_dict["in_layer.bias"],
            state_dict["out_layer.weight"],
            state_dict["out_layer.bias"],
        ]

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

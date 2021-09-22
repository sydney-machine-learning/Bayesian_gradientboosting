from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class MLP_1HL(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_1HL, self).__init__()
        self.in_layer = nn.Linear(dim_in, dim_hidden)
        self.out_layer = nn.Linear(dim_hidden, dim_out)
        self.relu = nn.ReLU()

        self.top = (dim_in, dim_hidden, dim_out)

    def forward(self, x):
        out = self.out_layer(self.relu(self.in_layer(x)))
        return out

    def evaluate_proposal(self, x, w):
        with torch.no_grad():

            # Set weights
            self.decode(w)

            return self.forward(x)

    def decode(self, w):

        in_size = self.top[0] * self.top[1]
        out_size = self.top[1] * self.top[2]

        in_weight = w[0:in_size]
        out_weight = w[in_size : in_size + out_size]
        in_bias = w[in_size + out_size : in_size + out_size + self.top[1]]
        out_bias = w[
            in_size + out_size + self.top[1] : in_size + out_size + self.top[1] + self.top[2]
        ]

        new_state_dict = OrderedDict(
            {
                "in_layer.weight": torch.as_tensor(
                    np.reshape(in_weight, (self.top[1], self.top[0])), dtype=torch.float32
                ).cuda(),
                "out_layer.weight": torch.as_tensor(
                    np.reshape(out_weight, (self.top[2], self.top[1])), dtype=torch.float32
                ).cuda(),
                "in_layer.bias": torch.as_tensor(in_bias, dtype=torch.float32).cuda(),
                "out_layer.bias": torch.as_tensor(out_bias, dtype=torch.float32).cuda(),
            }
        )
        self.load_state_dict(new_state_dict, strict=False)

    def encode(self):
        state_dict = self.state_dict()

        return np.concatenate(
            [
                state_dict["in_layer.weight"].cpu().ravel(),
                state_dict["out_layer.weight"].cpu().ravel(),
                state_dict["in_layer.bias"].cpu().ravel(),
                state_dict["out_layer.bias"].cpu().ravel(),
            ]
        )

    @classmethod
    def get_model(cls, config):
        model = MLP_1HL(config.feat_d, config.hidden_d, config.out_d)
        return model


import copy

import torch
from torch import nn


class Sin(torch.nn.Module):
    def forward(self, x, inverse=False):
        if not inverse:
            return torch.sin(x)
        else:
            return torch.asin(x)


class MLP(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        layer_size,
        output_size,
        activation="sigmoid",
        dropout=None,
        batch_norm=False,
    ):
        super(MLP, self).__init__()

        if activation == "sin":
            activation_module = Sin()
        elif activation == "tanh":
            activation_module = torch.nn.Tanh()
        elif activation == "relu":
            activation_module = torch.nn.ReLU()
        else:
            activation_module = torch.nn.Sigmoid()

        self.layers = nn.ModuleList([nn.Flatten()])
        for l in range(0, num_layers - 1):
            self.layers.append(nn.Linear(layer_size[l], layer_size[l+1], bias=True))
            self.layers.append(activation_module)
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(layer_size[l+1]))
            if dropout is not None:
                self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(layer_size[num_layers-1], output_size, bias=False))

    def forward(self, x, inverse=False):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def __getitem__(self, val):
        if isinstance(val, slice):
            clone = copy.deepcopy(self)
            clone.layers = clone.layers[val]
            return clone
        else:
            return self.layers[val]


def mlp_builder(seed: int, config):
    model = MLP(
        num_layers=config.num_layers,
        layer_size=config.layer_size,
        output_size=config.output_size,
        activation=config.activation,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
    )
    return model

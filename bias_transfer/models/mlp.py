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
        input_size,
        num_layers,
        layer_size,
        output_size,
        activation="sigmoid",
        dropout=0.0,
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

        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size, bias=True)])
        self.layers.append(activation_module)
        for i in range(1, num_layers - 1):
            if dropout:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(layer_size, layer_size, bias=True))
            self.layers.append(activation_module)
        if dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(layer_size, output_size, bias=False))

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

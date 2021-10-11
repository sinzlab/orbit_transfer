import copy

import torch
from torch import nn

from bias_transfer.models.mlp import Sin


class MaxOut(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert (
            x.shape[1] % self._pool_size == 0
        ), "Wrong input last dim size ({}) for Maxout({})".format(
            x.shape[1], self._pool_size
        )
        m, i = x.view(
            *x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]
        ).max(2)
        return m


class CNN(torch.nn.Module):
    def __init__(
        self,
        output_size,
        num_layers,
        channels,
        kernel_size,
        pool_size,
        activation="relu",
        dropout=None,
        max_out=None,
        batch_norm=False,
    ):
        super(CNN, self).__init__()

        if activation == "sin":
            activation_module = Sin()
        elif activation == "tanh":
            activation_module = torch.nn.Tanh()
        elif activation == "relu":
            activation_module = torch.nn.ReLU()
        else:
            activation_module = torch.nn.Sigmoid()

        self.layers = nn.ModuleList([])
        for l in range(num_layers - 1):
            self.layers.append(
                nn.Conv2d(
                    in_channels=channels[l],
                    out_channels=channels[l + 1],
                    kernel_size=kernel_size[l],
                    bias=True,
                )
            )
            if max_out and max_out[l]:
                self.layers.append(MaxOut(max_out[l]))
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(channels[l + 1]//max_out[l]))
            self.layers.append(activation_module)
            self.layers.append(nn.MaxPool2d(pool_size[l]))
            if dropout is not None:
                self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.AdaptiveAvgPool2d(1))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(channels[num_layers - 1]//max_out[num_layers-2], output_size, bias=True))

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


def cnn_builder(seed: int, config):
    model = CNN(
        output_size=config.output_size,
        num_layers=config.num_layers,
        channels=config.channels,
        kernel_size=config.kernel_size,
        pool_size=config.pool_size,
        activation=config.activation,
        dropout=config.dropout,
        max_out=config.max_out,
        batch_norm=config.batch_norm,
    )
    return model

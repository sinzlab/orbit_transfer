import torch
from torch import nn as nn
from torch.nn import functional as F

from bias_transfer.models.elrg.conv import ELRGConv2d
from bias_transfer.models.elrg.linear import ELRGLinear
from nntransfer.models.utils import concatenate_flattened


class LeNet5(
    nn.Module
):  # adapted from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def __init__(
        self,
        num_classes: int = 10,
        input_width: int = 28,
        input_height: int = 28,
        input_channels: int = 1,
        dropout: float = 0.0,
        core: str = "conv",
        rank: int = 1,
        alpha: float = None,
        train_var: bool = False,
        initial_var: float = 1e-12,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else 1 / rank
        self.input_size = (input_height, input_width)
        conv_out_width1 = (input_width - 3) + 1
        conv_out_width = (
            conv_out_width1 // 2 - 3
        ) + 1  # [(W-K+2P)/S]+1  (includes max pool before)
        conv_out_height1 = (input_height - 3) + 1
        conv_out_height = (
            conv_out_height1 // 2 - 3
        ) + 1  # [(H-K+2P)/S]+1  (includes max pool before)
        flat_feature_size = ((conv_out_height // 2) * (conv_out_width // 2)) * 16
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.core_type = core
        if core == "fc":
            flat_input_size = input_width * input_height * input_channels
            intermediate_size = (conv_out_height1 // 2) * (conv_out_width1 // 2) * 6
            self.conv1 = nn.Sequential(
                nn.Flatten(),
                ELRGLinear(
                    flat_input_size,
                    intermediate_size,
                    rank=rank,
                    alpha=self.alpha,
                    train_var=train_var,
                    initial_posterior_var=initial_var,
                ),
            )
            self.conv2 = ELRGLinear(
                intermediate_size,
                flat_feature_size,
                rank=rank,
                alpha=self.alpha,
                train_var=train_var,
                initial_posterior_var=initial_var,
            )
        else:  # core == "conv":
            if train_var:
                raise ValueError("Variance should be frozen for convolutional networks.")
            self.conv1 = ELRGConv2d(
                input_channels,
                6,
                3,
                rank=rank,
                alpha=self.alpha,
                initial_posterior_var=initial_var,
            )
            self.conv2 = ELRGConv2d(
                6,
                16,
                3,
                rank=rank,
                alpha=self.alpha,
                initial_posterior_var=initial_var,
            )
        # an affine operation: y = Wx + b
        self.fc1 = ELRGLinear(
            flat_feature_size,
            120,
            rank=rank,
            alpha=self.alpha,
            train_var=train_var,
            initial_posterior_var=initial_var,
        )
        self.fc2 = ELRGLinear(
            120,
            84,
            rank=rank,
            alpha=self.alpha,
            train_var=train_var,
            initial_posterior_var=initial_var,
        )
        self.fc3 = ELRGLinear(
            84,
            num_classes,
            rank=rank,
            alpha=self.alpha,
            train_var=train_var,
            initial_posterior_var=initial_var,
        )

        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x, num_samples=1):
        y = []
        for s in range(num_samples):
            z = F.relu(self.conv1(x))
            z = self.dropout(z) if self.dropout else z
            if not self.core_type == "fc":
                # Max pooling over a (2, 2) window
                z = F.max_pool2d(z, (2, 2))
            z = F.relu(self.conv2(z))
            z = self.dropout(z) if self.dropout else z
            if not self.core_type == "fc":
                # If the size is a square you can only specify a single number
                z = F.max_pool2d(z, 2)
            z = z.flatten(start_dim=1)
            z = F.relu(self.fc1(z))
            z = self.dropout(z) if self.dropout else z
            z = F.relu(self.fc2(z))
            z = self.dropout(z) if self.dropout else z
            y.append(self.fc3(z))
        return torch.cat(y)

    def get_parameters(self, name, keep_first_dim=False):
        return concatenate_flattened(
            [
                self.conv1._parameters.get(f"w_{name}"),
                self.conv2._parameters.get(f"w_{name}"),
                self.fc1._parameters.get(f"w_{name}"),
                self.fc2._parameters.get(f"w_{name}"),
                self.fc3._parameters.get(f"w_{name}"),
                self.conv1._parameters.get(f"b_{name}"),
                self.conv2._parameters.get(f"b_{name}"),
                self.fc1._parameters.get(f"b_{name}"),
                self.fc2._parameters.get(f"b_{name}"),
                self.fc3._parameters.get(f"b_{name}"),
            ],
            keep_first_dim=keep_first_dim,
        )

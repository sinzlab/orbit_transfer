import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from bias_transfer.models.bayes_by_backprop.linear import BayesLinear
from bias_transfer.models.bayes_by_backprop.conv import BayesConv2d
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
                BayesLinear(
                    flat_input_size,
                    intermediate_size,
                    initial_posterior_var=initial_var,
                ),
            )
            self.conv2 = BayesLinear(
                intermediate_size, flat_feature_size, initial_posterior_var=initial_var
            )
        else:  # core == "conv":
            self.conv1 = BayesConv2d(
                input_channels, 6, 3, initial_posterior_var=initial_var
            )
            self.conv2 = BayesConv2d(6, 16, 3, initial_posterior_var=initial_var)
        # an affine operation: y = Wx + b
        self.fc1 = BayesLinear(
            flat_feature_size, 120, initial_posterior_var=initial_var
        )
        self.fc2 = BayesLinear(120, 84, initial_posterior_var=initial_var)
        self.fc3 = BayesLinear(84, num_classes, initial_posterior_var=initial_var)

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

    def get_parameters(self, name):
        if "prior" in name:
            return concatenate_flattened(
                [
                    self.conv1.__getattribute__(f"w_{name}"),
                    self.conv2.__getattribute__(f"w_{name}"),
                    self.fc1.__getattribute__(f"w_{name}"),
                    self.fc2.__getattribute__(f"w_{name}"),
                    self.fc3.__getattribute__(f"w_{name}"),
                    self.conv1.__getattribute__(f"b_{name}"),
                    self.conv2.__getattribute__(f"b_{name}"),
                    self.fc1.__getattribute__(f"b_{name}"),
                    self.fc2.__getattribute__(f"b_{name}"),
                    self.fc3.__getattribute__(f"b_{name}"),
                ]
            )
        else:
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
                ]
            )

    def to(self, *args, **kwargs):
        """
        Our prior tensors are registered as buffers but the way we access them
        indirectly (through tuple attributes on the model) is causing problems
        because when we use `.to()` to move the model to a new device, the prior
        tensors get moved (because they're registered as buffers) but the
        references in the tuples don't get updated to point to the new moved
        tensors. This has no effect when running just on a cpu but breaks the
        model when trying to run on a gpu. There are a million nicer ways of
        working around this problem, but for now the easiest thing is to do
        this: override the `.to()` method and manually update our references to
        prior tensors.
        """
        self = super().to(*args, **kwargs)
        for fc in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            fc.w_prior_mean = fc.w_prior_mean.to(*args, **kwargs)
            fc.w_prior_log_var = fc.w_prior_log_var.to(*args, **kwargs)
            fc.b_prior_mean = fc.b_prior_mean.to(*args, **kwargs)
            fc.b_prior_log_var = fc.b_prior_log_var.to(*args, **kwargs)
        return self

    def reset_for_new_task(self):
        for fc in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            fc.reset_for_new_task()

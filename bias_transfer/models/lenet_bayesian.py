import math
from typing import OrderedDict, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from bias_transfer.models.utils import concatenate_flattened


class BayesLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_posterior_var: float = 1e-3,
        bias: bool = True,
    ):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_posterior_var = initial_posterior_var
        (
            self.w_prior_mean,
            self.w_prior_log_var,
            self.w_posterior_mean,
            self.w_posterior_log_var,
        ) = self.create_parameter("weight", (out_features, in_features))
        if bias:
            (
                self.b_prior_mean,
                self.b_prior_log_var,
                self.b_posterior_mean,
                self.b_posterior_log_var,
            ) = self.create_parameter("bias", (out_features,))
        else:
            self.register_parameter("b_posterior_mean", None)
            self.register_parameter("b_posterior_log_var", None)
        self.reset_parameters()

    def create_parameter(self, name, dims):
        prior_mean = torch.zeros(*dims)
        prior_log_var = torch.zeros(*dims)
        posterior_mean = nn.Parameter(torch.Tensor(*dims), requires_grad=True)
        posterior_log_var = nn.Parameter(torch.Tensor(*dims), requires_grad=True)
        # Finally, we register the prior and the posterior with the nn.Module.
        # The prior values are registered as buffers, which indicates to PyTorch
        # that they represent persistent state which should not be updated by
        # the optimizer. The posteriors are registered as parameters, which on
        # the other hand are to be modified by the optimizer.
        self.register_buffer(f"{name}", prior_mean)  # to load with the right name
        self.register_buffer(f"prior_{name}_log_var", prior_log_var)

        return prior_mean, prior_log_var, posterior_mean, posterior_log_var

    def reset_for_new_task(self):
        """
        Called after completion of a task, to reset state for the next task
        """
        # Set the value of the prior to be the current value of the posterior
        self.w_prior_mean.data.copy_(self.w_posterior_mean.data)
        self.b_prior_mean.data.copy_(self.b_posterior_mean.data)
        self.w_prior_log_var.data.copy_(self.w_posterior_log_var.data)
        self.b_prior_log_var.data.copy_(self.b_posterior_log_var.data)

    def reset_parameters(self):
        # Initialise the posterior means with a normal distribution. Note that
        # prior to training we will run a procedure to optimise these values to
        # point-estimates of the parameters for the first task.
        torch.nn.init.normal_(self.w_posterior_mean, mean=0, std=0.1)
        # Initialise the posterior variances with the given constant value.
        torch.nn.init.constant_(
            self.w_posterior_log_var, math.log(self.initial_posterior_var)
        )
        if self.bias is not None:
            torch.nn.init.normal_(self.b_posterior_mean, mean=0, std=0.1)
            # Initialise the posterior variances with the given constant value.
            torch.nn.init.constant_(
                self.b_posterior_log_var, math.log(self.initial_posterior_var)
            )

    @staticmethod
    def _sample_parameters(w_mean, b_mean, w_log_var, b_log_var):
        # sample weights and biases from normal distributions
        w_epsilon = torch.randn_like(w_mean)
        b_epsilon = torch.randn_like(b_mean)
        sampled_weight = w_mean + w_epsilon * torch.exp(0.5 * w_log_var)
        sampled_bias = b_mean + b_epsilon * torch.exp(0.5 * b_log_var)
        return sampled_weight, sampled_bias

    def forward(self, input):
        sampled_weight, sampled_bias = self._sample_parameters(
            self.w_posterior_mean,
            self.b_posterior_mean,
            self.w_posterior_log_var,
            self.b_posterior_log_var,
        )
        return F.linear(input, sampled_weight, sampled_bias)


class LeNet300100(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 28,
        input_channels: int = 1,
        dropout: float = 0.0,
    ):
        super(LeNet300100, self).__init__()
        self.fc1 = BayesLinear(input_size * input_size * input_channels, 300)
        self.fc2 = BayesLinear(300, 100)
        self.fc3 = BayesLinear(100, num_classes)
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.flat_input_size = input_size * input_size * input_channels

    def forward(self, x, num_samples=1):
        x = x.view(x.size(0), self.flat_input_size)
        y = []
        for s in range(num_samples):
            z = F.relu(self.fc1(x))
            z = self.dropout(z) if self.dropout else z
            z = F.relu(self.fc2(z))
            z = self.dropout(z) if self.dropout else z
            y.append(self.fc3(z))
        return torch.cat(y)

    def get_parameters(self, name):
        if "prior" in name:
            return concatenate_flattened(
                [
                    self.fc1.__getattribute__(f"w_{name}"),
                    self.fc2.__getattribute__(f"w_{name}"),
                    self.fc3.__getattribute__(f"w_{name}"),
                    self.fc1.__getattribute__(f"b_{name}"),
                    self.fc2.__getattribute__(f"b_{name}"),
                    self.fc3.__getattribute__(f"b_{name}"),
                ]
            )
        else:
            return concatenate_flattened(
                [
                    self.fc1._parameters.get(f"w_{name}"),
                    self.fc2._parameters.get(f"w_{name}"),
                    self.fc3._parameters.get(f"w_{name}"),
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
        for fc in [self.fc1, self.fc2, self.fc3]:
            fc.w_prior_mean = fc.w_prior_mean.to(*args, **kwargs)
            fc.w_prior_log_var = fc.w_prior_log_var.to(*args, **kwargs)
            fc.b_prior_mean = fc.b_prior_mean.to(*args, **kwargs)
            fc.b_prior_log_var = fc.b_prior_log_var.to(*args, **kwargs)
        return self

    def reset_for_new_task(self):
        for fc in [self.fc1, self.fc2, self.fc3]:
            fc.reset_for_new_task()


def lenet_builder(seed: int, config):
    if "5" in config.type:
        lenet = LeNet5
    elif "300-100" in config.type:
        lenet = LeNet300100

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    model = lenet(
        num_classes=config.num_classes,
        input_size=config.input_size,
        input_channels=config.input_channels,
        dropout=config.dropout,
    )
    return model
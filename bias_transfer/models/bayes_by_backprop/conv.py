import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from nntransfer.models.utils import concatenate_flattened


class BayesConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        initial_posterior_var: float = 1e-3,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.initial_posterior_var = initial_posterior_var
        (
            self.w_prior_mean,
            self.w_prior_log_var,
            self.w_posterior_mean,
            self.w_posterior_log_var,
        ) = self.create_parameter(
            "weight", (out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        (
            self.b_prior_mean,
            self.b_prior_log_var,
            self.b_posterior_mean,
            self.b_posterior_log_var,
        ) = self.create_parameter("bias", (out_channels,), empty=not bias)
        self.reset_parameters()

    def create_parameter(self, name, dims, empty=False):
        if empty:
            prior_mean = None
            prior_log_var = None
            posterior_mean = nn.Parameter(None)
            posterior_log_var = nn.Parameter(None)
        else:
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
        self.w_prior_log_var.data.copy_(self.w_posterior_log_var.data)
        if self.use_bias:
            self.b_prior_mean.data.copy_(self.b_posterior_mean.data)
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
        if self.use_bias:
            torch.nn.init.normal_(self.b_posterior_mean, mean=0, std=0.1)
            # Initialise the posterior variances with the given constant value.
            torch.nn.init.constant_(
                self.b_posterior_log_var, math.log(self.initial_posterior_var)
            )

    @staticmethod
    def _sample_parameters(w_mean, b_mean, w_log_var, b_log_var):
        # sample weights and biases from normal distributions
        w_epsilon = torch.randn_like(w_mean)
        sampled_weight = w_mean + w_epsilon * torch.exp(0.5 * w_log_var)
        if b_mean is not None and b_mean.nelement() != 0:
            b_epsilon = torch.randn_like(b_mean)
            sampled_bias = b_mean + b_epsilon * torch.exp(0.5 * b_log_var)
        else:
            sampled_bias = None
        return sampled_weight, sampled_bias

    def forward(self, x, num_samples=0):
        if num_samples:
            y = []
            for s in range(num_samples):
                y.append(self.forward(x))
            return torch.cat(y)
        sampled_weight, sampled_bias = self._sample_parameters(
            self.w_posterior_mean,
            self.b_posterior_mean,
            self.w_posterior_log_var,
            self.b_posterior_log_var,
        )
        return F.conv2d(
            x,
            sampled_weight,
            sampled_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def get_parameters(self, name):
        if "prior" in name:
            return concatenate_flattened(
                [
                    self.__getattribute__(f"w_{name}"),
                    self.__getattribute__(f"b_{name}"),
                ]
            )
        else:
            return concatenate_flattened(
                [
                    self._parameters.get(f"w_{name}"),
                    self._parameters.get(f"b_{name}"),
                ]
            )

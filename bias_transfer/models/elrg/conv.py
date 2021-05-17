import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from nntransfer.models.utils import concatenate_flattened


class ELRGConv2d(nn.Module):
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
        initial_posterior_var: float = 1e-12,
        rank: int = 1,
        alpha: float = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.rank = rank
        self.alpha = alpha
        self.sqrt_alpha = torch.sqrt(torch.tensor(alpha))
        self.initial_posterior_var = initial_posterior_var
        (
            self.w_posterior_mean,
            self.w_posterior_v,
            self.w_posterior_log_var,
        ) = self.create_parameter(
            "weight", (out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            (
                self.b_posterior_mean,
                self.b_posterior_v,
                self.b_posterior_log_var,
            ) = self.create_parameter("bias", (out_channels,))
        else:
            self.register_parameter("b_posterior_mean", None)
            self.register_parameter("b_posterior_v", None)
            self.register_parameter("b_posterior_log_var", None)
        self.reset_parameters()

        # to be compatible with deterministic model
        self.weight = self.w_posterior_mean
        if bias:
            self.bias = self.b_posterior_mean

        # To freeze var:
        self.w_posterior_log_var.requires_grad = False
        if bias:
            self.b_posterior_log_var.requires_grad = False

    def create_parameter(self, name, dims):
        posterior_mean = nn.Parameter(torch.Tensor(*dims), requires_grad=True)
        posterior_v = nn.Parameter(torch.Tensor(self.rank, *dims), requires_grad=True)
        posterior_log_var = nn.Parameter(torch.Tensor(*dims), requires_grad=True)
        return posterior_mean, posterior_v, posterior_log_var

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.w_posterior_mean, gain=1.0)
        torch.nn.init.xavier_normal_(self.w_posterior_v, gain=0.5)
        # Initialise the posterior variances with the given constant value.
        torch.nn.init.constant_(
            self.w_posterior_log_var, math.log(self.initial_posterior_var)
        )
        if self.use_bias:
            torch.nn.init.normal_(self.b_posterior_mean, mean=0, std=0.1)
            torch.nn.init.normal_(self.b_posterior_v, mean=0, std=0.1)
            # Initialise the posterior variances with the given constant value.
            torch.nn.init.constant_(
                self.b_posterior_log_var, math.log(self.initial_posterior_var)
            )

    def forward(self, x, num_samples=0):
        if num_samples:
            y = []
            for s in range(num_samples):
                y.append(self.forward(x))
            return torch.cat(y)
        sampled_output = F.conv2d(
            x,
            self.w_posterior_mean,
            self.b_posterior_mean,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        epsilon_var = torch.randn(sampled_output.shape, device=x.device)
        sampled_output += epsilon_var * torch.sqrt(
            F.conv2d(
                x ** 2,
                torch.exp(self.w_posterior_log_var),
                torch.exp(self.b_posterior_log_var) if self.use_bias else None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        )

        epsilon_v = torch.randn((x.shape[0], self.rank), device=x.device)
        v_output = torch.zeros_like(sampled_output)
        for k in range(self.rank):
            epsilon_v_k = epsilon_v[:, k][(...,) + (None,) * 3]  # unsqueeze last 3 dims
            v_output += epsilon_v_k * F.conv2d(
                x,
                self.w_posterior_v[k],
                self.b_posterior_v[k] if self.use_bias else None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        return sampled_output + self.sqrt_alpha * v_output

    def get_parameters(self, name, keep_first_dim=False):
        return concatenate_flattened(
            [
                self._parameters.get(f"w_{name}"),
                self._parameters.get(f"b_{name}"),
            ],
            keep_first_dim=keep_first_dim,
        )

import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from nntransfer.models.utils import concatenate_flattened


class ELRGLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_posterior_var: float = 1e-12,
        bias: bool = True,
        rank: int = 1,
        alpha: float = None,
        train_var=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.rank = rank
        self.alpha = alpha
        self.sqrt_alpha = torch.sqrt(torch.tensor(alpha))
        self.initial_posterior_var = initial_posterior_var
        (
            # self.w_prior_mean,
            # self.w_prior_log_var,
            self.w_posterior_mean,
            self.w_posterior_v,
            self.w_posterior_log_var,
        ) = self.create_parameter("weight", (out_features, in_features))
        if bias:
            (
                # self.b_prior_mean,
                # self.b_prior_log_var,
                self.b_posterior_mean,
                self.b_posterior_v,
                self.b_posterior_log_var,
            ) = self.create_parameter("bias", (out_features,))
        else:
            self.register_parameter("b_posterior_mean", None)
            self.register_parameter("b_posterior_v", None)
            self.register_parameter("b_posterior_log_var", None)
        self.reset_parameters()

        # to be compatible with deterministic model
        self.weight = self.w_posterior_mean
        if bias:
            self.bias = self.b_posterior_mean

        if not train_var:
            self.w_posterior_log_var.requires_grad = False
            if bias:
                self.b_posterior_log_var.requires_grad = False

    def create_parameter(self, name, dims):
        # prior_mean = torch.zeros(*dims)
        # prior_log_var = torch.zeros(*dims)
        posterior_mean = nn.Parameter(torch.Tensor(*dims), requires_grad=True)
        posterior_v = nn.Parameter(torch.Tensor(self.rank, *dims), requires_grad=True)
        posterior_log_var = nn.Parameter(torch.Tensor(*dims), requires_grad=True)
        # Finally, we register the prior and the posterior with the nn.Module.
        # The prior values are registered as buffers, which indicates to PyTorch
        # that they represent persistent state which should not be updated by
        # the optimizer. The posteriors are registered as parameters, which on
        # the other hand are to be modified by the optimizer.
        # self.register_buffer(f"{name}", prior_mean)  # to load with the right name
        # self.register_buffer(f"prior_{name}_log_var", prior_log_var)

        # return prior_mean, prior_log_var, posterior_mean, posterior_log_var
        return posterior_mean, posterior_v, posterior_log_var

    # def reset_for_new_task(self):
    #     """
    #     Called after completion of a task, to reset state for the next task
    #     """
    #     # Set the value of the prior to be the current value of the posterior
    #     self.w_prior_mean.data.copy_(self.w_posterior_mean.data)
    #     self.b_prior_mean.data.copy_(self.b_posterior_mean.data)
    #     self.w_prior_log_var.data.copy_(self.w_posterior_log_var.data)
    #     self.b_prior_log_var.data.copy_(self.b_posterior_log_var.data)

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
        epsilon_var = torch.randn((x.shape[0], self.out_features), device=x.device)
        epsilon_v = torch.randn((x.shape[0], self.rank), device=x.device)
        sampled_output = F.linear(x, self.w_posterior_mean, self.b_posterior_mean)
        sampled_output += epsilon_var * torch.sqrt(
            F.linear(
                x ** 2,
                torch.exp(self.w_posterior_log_var),
                torch.exp(self.b_posterior_log_var) if self.use_bias else None,
            )
        )
        v_output = torch.zeros_like(sampled_output)
        for k in range(self.rank):
            v_output += epsilon_v[:, k].unsqueeze(dim=-1) * F.linear(
                x,
                self.w_posterior_v[k],
                self.b_posterior_v[k] if self.use_bias else None,
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



import math
from typing import OrderedDict, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        train_var = True,
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
        epsilon_v = torch.randn(self.rank)
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
            v_output += epsilon_v[k] * F.linear(
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


class LeNet300100(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_height: int = 28,
        input_width: int = 28,
        input_channels: int = 1,
        dropout: float = 0.0,
        rank: int = 1,
        alpha: float = None,
        train_var: bool=True,
    ):
        super(LeNet300100, self).__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else 1 / rank
        self.input_size = (input_height, input_width)
        self.flat_input_size = input_width * input_height * input_channels
        self.fc1 = ELRGLinear(self.flat_input_size, 300, rank=rank, alpha=self.alpha, train_var=train_var)
        self.fc2 = ELRGLinear(300, 100, rank=rank, alpha=self.alpha, train_var=train_var)
        self.fc3 = ELRGLinear(100, num_classes, rank=rank, alpha=self.alpha, train_var=train_var)
        self.dropout = nn.Dropout(p=dropout) if dropout else None

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

    def get_parameters(self, name, keep_first_dim=False):
        # if "prior" in name:
        #     return concatenate_flattened(
        #         [
        #             self.fc1.__getattribute__(f"w_{name}"),
        #             self.fc2.__getattribute__(f"w_{name}"),
        #             self.fc3.__getattribute__(f"w_{name}"),
        #             self.fc1.__getattribute__(f"b_{name}"),
        #             self.fc2.__getattribute__(f"b_{name}"),
        #             self.fc3.__getattribute__(f"b_{name}"),
        #         ]
        #     )
        # else:
        return concatenate_flattened(
            [
                self.fc1._parameters.get(f"w_{name}"),
                self.fc2._parameters.get(f"w_{name}"),
                self.fc3._parameters.get(f"w_{name}"),
                self.fc1._parameters.get(f"b_{name}"),
                self.fc2._parameters.get(f"b_{name}"),
                self.fc3._parameters.get(f"b_{name}"),
            ],
            keep_first_dim=keep_first_dim,
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
        # for fc in [self.fc1, self.fc2, self.fc3]:
        #     fc.w_prior_mean = fc.w_prior_mean.to(*args, **kwargs)
        #     fc.w_prior_log_var = fc.w_prior_log_var.to(*args, **kwargs)
        #     fc.b_prior_mean = fc.b_prior_mean.to(*args, **kwargs)
        #     fc.b_prior_log_var = fc.b_prior_log_var.to(*args, **kwargs)
        return self

    # def reset_for_new_task(self):
    #     for fc in [self.fc1, self.fc2, self.fc3]:
    #         fc.reset_for_new_task()


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
        input_height=config.input_height,
        input_width=config.input_width,
        input_channels=config.input_channels,
        dropout=config.dropout,
        rank=config.rank,
        alpha=config.alpha,
        train_var=config.train_var
    )
    return model

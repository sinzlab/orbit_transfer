import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal


class FRCL(nn.Module):
    def __init__(
        self,
        input_size,
        input_channels,
        h_dim,
        coreset_size,
        num_classes: int = 10,
        dropout: float = 0.0,
        sigma_prior=1,
        init_mu_std=1.0,
    ):
        """
        Adapted from the implementation of https://github.com/AndreevP/FRCL
        Args:
            input_size:
            input_channels:
            h_dim:
            coreset_size:
            num_classes:
            dropout:
            sigma_prior:
            init_mu_std:
        """
        super(FRCL, self).__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout) if dropout else None

        self.sigma_prior = sigma_prior
        self.w_prior = MultivariateNormal(
            torch.zeros(h_dim), covariance_matrix=sigma_prior * torch.eye(h_dim),
        )
        self.pred_func = nn.Softmax()
        self.init_mu_std = init_mu_std

        self.L = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(h_dim, h_dim), requires_grad=True)
                for _ in range(num_classes)
            ]
        )
        self.mu = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(h_dim), requires_grad=True,)
                for _ in range(num_classes)
            ]
        )
        self.mu_prev, self.cov_prev = [], []
        for i in range(num_classes):
            self.register_buffer(f"mu_prev_{i}", torch.zeros(coreset_size))
            self.register_buffer(
                f"cov_prev_{i}", torch.zeros(coreset_size, coreset_size)
            )
        self.register_buffer(
            "coreset", torch.zeros(coreset_size, input_channels, input_size, input_size)
        )
        self.register_buffer(
            "coreset_prev",
            torch.zeros(coreset_size, input_channels, input_size, input_size),
        )
        self.device = self.coreset.device
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_classes):
            torch.nn.init.eye_(self.L[i])
            torch.nn.init.normal_(self.mu[i], mean=0, std=self.init_mu_std)
        # TODO reset the rest?

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for i in range(self.num_classes):
            self._buffers[f"mu_prev_{i}"] = self._buffers[f"mu_prev_{i}"].to(
                *args, **kwargs
            )
            self._buffers[f"cov_prev_{i}"] = self._buffers[f"cov_prev_{i}"].to(
                *args, **kwargs
            )
        self.w_prior = MultivariateNormal(
            self.w_prior.mean.to(*args, **kwargs),
            covariance_matrix=self.w_prior.covariance_matrix.to(*args, **kwargs),
        )
        self.coreset = self.coreset.to(*args, **kwargs)
        self.coreset_prev = self.coreset_prev.to(*args, **kwargs)
        self.device = self.coreset.device
        return self

    def reset_for_new_task(self):
        """
        Called after completion of a task, to reset state for the next task
        """
        self.coreset_prev = self.coreset
        phi_z = self.core_forward(self.coreset_prev)
        for i in range(self.num_classes):
            mu, cov = self._get_inducing_distribution(phi_z, i)
            self._buffers[f"mu_prev_{i}"] = mu
            self._buffers[f"cov_prev_{i}"] = cov

    @property
    def prev(self):
        try:
            return self._prev
        except AttributeError:
            self._prev = torch.any(self.coreset_prev != 0)
            return self._prev

    def forward(self, x, num_samples=8):
        phi = self.core_forward(x)
        if self.training:
            return self._train_forward(phi, num_samples)
        else:
            return self._eval_forward(phi, num_samples)

    def core_forward(self, x):
        raise NotImplementedError()

    def _train_forward(self, phi, num_samples):
        """
        Return -ELBO
        N_k = len(dataset), required for unbiased estimate through minibatch
        """
        mu = self.mu
        cov = [self.L[i] @ self.L[i].T for i in range(len(self.L))]
        means = torch.stack([phi @ mu[i] for i in range(len(mu))], dim=1)
        #  variances = torch.cat([((phi @ cov[i]) * phi).sum(-1) for i in range(len(cov))], axis = 0)
        variances = torch.stack(
            [torch.diagonal(phi @ cov[i] @ phi.T, 0) for i in range(len(cov))], dim=1
        )
        samples = torch.cat(
            [
                means
                + torch.sqrt(variances + 1e-6)
                * torch.randn(means.shape).to(self.device)
                for i in range(num_samples)
            ]
        )
        return samples

    def _get_inducing_distribution(self, phi_z, i):
        mu_u = phi_z @ self.mu[i]
        L_u = phi_z @ self.L[i]
        cov_u = L_u @ L_u.T
        cov_u = cov_u + torch.eye(cov_u.shape[0]).to(self.device) * 1e-4
        return mu_u, cov_u

    def _get_predictive(self, phi_x):
        """ Computes predictive distribution according to section 2.5
            x - batch of data
            k - index of task
            Return predictive distribution q_\theta(f)
        """
        phi_z = self.core_forward(self.coreset)
        k_xx = phi_x @ phi_x.T * self.sigma_prior
        k_xz = phi_x @ phi_z.T * self.sigma_prior
        k_zz = phi_z @ phi_z.T * self.sigma_prior
        k_zz_ = torch.inverse(k_zz + torch.eye(phi_z.shape[0]).to(self.device) * 1e-3)

        mu_u, cov_u = (
            [None for _ in range(self.num_classes)],
            [None for _ in range(self.num_classes)],
        )
        for i in range(self.num_classes):
            mu_u[i], cov_u[i] = self._get_inducing_distribution(phi_z, i)

        mu = [phi_x @ phi_z.T @ k_zz_ @ mu_u[i] for i in range(self.num_classes)]
        sigma = [
            k_xx
            + (
                k_xz
                @ k_zz_
                @ (cov_u[i] - k_zz + torch.eye(k_zz.shape[0]).to(self.device) * 1e-4)
                @ k_zz_
                @ k_xz.T
            )
            for i in range(self.num_classes)
        ]
        sigma = [
            sigma[i] * torch.eye(sigma[i].shape[0]).to(self.device)
            + torch.eye(sigma[i].shape[0]).to(self.device) * 1e-6
            for i in range(self.num_classes)
        ]
        # print([s.min() for s in sigma])
        sigma = [
            torch.clamp(sigma[i], min=0, max=10000.0)
            + torch.eye(sigma[i].shape[0]).to(self.device) * 1e-6
            for i in range(self.num_classes)
        ]
        # we are interested only
        # in diagonal part for inference ?
        return [
            MultivariateNormal(loc=mu[i], covariance_matrix=sigma[i])
            for i in range(self.num_classes)
        ]

    def _eval_forward(self, phi, num_samples):
        """
        Compute p(y) by MC estimate from q_\theta(f)?
        """
        distr = self._get_predictive(phi)
        # TODO: speedup possible if you precompute distr before an eval epoch (i.e. not recompute for each batch)
        predicted = []
        for _ in range(num_samples):
            sample = [distr[i].sample() for i in range(self.num_classes)]
            predicted.append(self.pred_func(torch.stack(sample, dim=1)))
        return torch.cat(predicted)


class LeNet5(
    FRCL
):  # adapted from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def __init__(self, input_size: int = 28, input_channels: int = 1, *args, **kwargs):
        super(LeNet5, self).__init__(
            input_size=input_size,
            input_channels=input_channels,
            h_dim=100,
            *args,
            **kwargs,
        )
        conv_out_size = int(
            ((((input_size - 3) + 1) / 2 - 3) + 1) / 2
        )  # [(W-K+2P)/S]+1 / MP
        self.flat_feature_size = (conv_out_size ** 2) * 16
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(input_channels, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.flat_feature_size, 120)
        self.fc2 = nn.Linear(120, 84)

    def core_forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x) if self.dropout else x
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = self.dropout(x) if self.dropout else x
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.flat_feature_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if self.dropout else x
        x = F.relu(self.fc2(x))
        x = self.dropout(x) if self.dropout else x
        return x


class LeNet300100(FRCL):
    def __init__(self, input_size: int = 28, input_channels: int = 1, *args, **kwargs):
        super(LeNet300100, self).__init__(
            input_size=input_size,
            input_channels=input_channels,
            h_dim=100,
            *args,
            **kwargs,
        )
        self.fc1 = nn.Linear(input_size * input_size * input_channels, 300)
        self.fc2 = nn.Linear(300, 100)
        self.flat_input_size = input_size * input_size * input_channels

    def core_forward(self, x):
        x = x.view(x.size(0), self.flat_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if self.dropout else x
        x = F.relu(self.fc2(x))
        x = self.dropout(x) if self.dropout else x
        return x


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
        coreset_size=config.coreset_size,
    )
    return model

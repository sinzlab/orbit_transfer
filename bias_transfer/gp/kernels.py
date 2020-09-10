from scipy.spatial.distance import cdist
import numpy as np
import torch
from .nn_kernel import compute_cov_matrix


def linear(x1, x2, sigma_b=1, sigma_v=1, c=0, **kwargs):
    """
    Linear Kernel: $k(x_1,x_2) =\sigma_b^2 + \sigma_v^2 (x_1 - c)(x_2-c)$
    """
    return sigma_b ** 2 + sigma_v ** 2 * np.inner(x1 - c, x2 - c)


def rbf(x1, x2, l=1, sigma_f=1, **kwargs):
    """
    RBF Kernel: $k(x_1,x_2) =\sigma^2 \exp\left( - \frac{||x_1-x_2||^2}{2l^2} \right)$
    """
    dists = cdist(x1 / l, x2 / l, metric="sqeuclidean")
    return sigma_f ** 2 * np.exp(-0.5 * dists)


def periodic(x1, x2, l=1.0, sigma_f=1.0, p=5.0, **kwargs):
    """
    Periodic Kernel: $k(x_1,x_2) =\sigma^2 \exp\left( - \frac{2\sin^2(\pi|x_1-x_2|/p)}{l^2} \right)$
    """
    dists = cdist(x1, x2, metric="euclidean")
    return sigma_f ** 2 * np.exp(-2 * (np.sin(np.pi / p * dists) / l) ** 2)


def locally_periodic(x1, x2, l=1, sigma_f=0.5, p=2.0):
    """
    Locally Periodic Kernel: $k(x_1,x_2) =\sigma^2 \exp\left( - \frac{2\sin^2(\pi|x_1-x_2|/p)}{l^2} \right) \exp\left(-\frac{||x_1-x_2||^2}{2l^2}\right)$
    """
    return periodic(x1, x2, l, sigma_f, p) * rbf(x1, x2, l, sigma_f) / sigma_f ** 2


def white_noise(x1, x2, sigma=0.1, **kwargs):
    """
    White Noise Kernel: $k(x_1,x_2) = \sigma^2 \cdot I_n$
    """
    if x1 is x2:
        return sigma ** 2 * np.eye(len(x1))
    else:
        return np.zeros((len(x1), len(x2)))


def add_white_noise(kernel):
    return lambda x1, x2, sigma_noise=0.1, **opts: white_noise(x1, x2, sigma_noise) + kernel(
        x1, x2, **opts
    )

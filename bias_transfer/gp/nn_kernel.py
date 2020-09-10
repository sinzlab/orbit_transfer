from functools import partial

import numpy as np
import torch
from scipy.optimize import minimize
from torch.autograd import Variable
from tqdm import tqdm


def compute_cov_matrix(x1, x2, sigma=None):
    x1_flat = x1.reshape((x1.shape[0], -1))
    centered1 = x1_flat  # - x1_flat.mean(axis=1).reshape((-1, 1))
    x2_flat = x2.reshape((x2.shape[0], -1))
    centered2 = x2_flat  # - x2_flat.mean(axis=1).reshape((-1, 1))
    if sigma is not None:
        result = (
            centered1
            @ sigma
            @ centered2.T
            # / np.outer(np.linalg.norm(centered1, 2, axis=1), np.linalg.norm(centered2, 2, axis=1))
        )  # see https://de.mathworks.com/help/images/ref/corr2.html
    else:
        result = (
            centered1
            @ centered2.T
            # / np.outer(np.linalg.norm(centered1, 2, axis=1), np.linalg.norm(centered2, 2, axis=1))
        )  # see https://de.mathworks.com/help/images/ref/corr2.html
    return result


def nn_kernel(x1, x2, net, train_reps=None, weights=None, device="cpu", sigma=None):
    def get_reps(x):
        if np.count_nonzero(x) == 0:
            phi = train_reps
        else:
            x = torch.tensor(x, dtype=torch.float).to(device)
            phi = net[:-1](x).detach().cpu().numpy()
        return phi

    phi1 = get_reps(x1)
    phi2 = get_reps(x2)
    RSM = compute_cov_matrix(
        phi1,
        phi2,
        sigma=sigma
        if np.count_nonzero(x1) != 0 and np.count_nonzero(x2) != 0
        else None,
    )  # .cpu().numpy()
    if np.count_nonzero(x2) == 0 and weights is not None:
        RSM = RSM @ weights
    elif np.count_nonzero(x1) == 0 and weights is not None:
        RSM = weights @ RSM
    return RSM


def optimize_noise(kernel, X_train, Y_train):
    # -log liklihood
    def nll_fn(x, y):
        def step(theta):
            K = kernel(x, x, sigma_noise=theta[0])
            return (
                np.sum(np.log(np.diagonal(np.linalg.cholesky(K))))
                + 0.5 * y.T @ np.linalg.inv(K) @ y
                + 0.5 * len(x) * np.log(2 * np.pi)
            )

        return step

    # minimize -log liklihood
    res = minimize(
        nll_fn(X_train, Y_train), [0.01], bounds=((1e-5, 1e1),), method="L-BFGS-B"
    )

    sigma_noise_opt = res.x
    fitted_kernel = partial(kernel, sigma_noise=sigma_noise_opt)
    return fitted_kernel


def get_nn_eigen_kernel(net, device):
    v = net[-1].weight.detach().cpu().numpy().T
    # v -= np.mean(v)
    sigma = v @ v.T
    eig_vals, eig_vecs = np.linalg.eigh(sigma)
    # eig_vals (n) with possibly complex entries
    # eig_vecs (n x n) where [:,j] corresponds to eig_vals[j]
    # sort:
    eig_sorting = np.argsort(-eig_vals)
    eig_vals = eig_vals[eig_sorting[:1]]
    eig_vecs = eig_vecs[:, eig_sorting[:1]]
    weights = np.diag(eig_vals)
    kernel = partial(
        nn_kernel,
        net=net,
        device=device,
        train_reps=eig_vecs.T,
        weights=weights,
        sigma=sigma,
    )
    base_point_preds = eig_vecs.T @ v
    # base_points = inverse_computation(net, torch.tensor(eig_vecs.T, device=device))
    return kernel, base_point_preds, None


def inverse_computation(net, out_vecs):
    print(net)
    print(net[1:-1])
    print(net[:1])
    first_layer_out = net[1:-1](out_vecs, inverse=True).detach()
    print("first_layer", first_layer_out)
    x = Variable(
        100 * torch.randn(first_layer_out.shape[0], 1).cuda(), requires_grad=True
    )
    params = net.parameters()
    optim = torch.optim.Adam([x], 0.001)
    for param in params:
        param.requires_grad = False
    if hasattr(
        tqdm, "_instances"
    ):  # To have tqdm output without line-breaks between steps
        tqdm._instances.clear()
    net.train()
    t = tqdm(range(100))
    for batch in t:
        y = net[:1](x)
        loss = torch.mean((first_layer_out - y) ** 2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        t.set_postfix(
            loss=loss.item(),
            eig_vec_0=first_layer_out[0][:4].cpu().numpy(),
            phi_0=y[0][:4].detach().cpu().numpy(),
        )
    net.eval()
    return x.detach().cpu().numpy()


def optimize_base_points(net):
    v = net[-1].weight.detach().T
    sigma = v @ v.T
    eig_vals, eig_vecs = torch.eig(sigma, eigenvectors=True)
    # eig_vals (n x 2) with entries (real,imaginary)
    # eig_vecs (n x n) where [:,j] corresponds to eig_vals[j]
    eig_vecs = eig_vecs.T
    x = Variable(15 * torch.randn(eig_vecs.shape[0], 1).cuda(), requires_grad=True)
    params = net.parameters()
    optim = torch.optim.Adam([x], 0.001)
    for param in params:
        param.requires_grad = False
    if hasattr(
        tqdm, "_instances"
    ):  # To have tqdm output without line-breaks between steps
        tqdm._instances.clear()
    t = tqdm(range(100))
    net.train()
    for batch in t:
        y = net[:-1](x)
        loss = torch.mean((eig_vecs - y) ** 2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        net.eval()
        phi = net[:-1](x)
        t.set_postfix(
            loss=loss.item(),
            eig_vec_0=eig_vecs[0][:4].cpu().numpy(),
            phi_0=phi[0][:4].detach().cpu().numpy(),
        )
        net.train()
    net.eval()
    return x.detach().cpu().numpy(), eig_vals.cpu().numpy()

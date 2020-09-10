import numpy as np
from scipy.optimize import minimize
from functools import partial


def posterior_predictive(X, X_train, Y_train, kernel, **opts):
    K = kernel(X_train, X_train, **opts)
    K_s = kernel(X_train, X, **opts)
    K_ss = kernel(X, X, **opts)

    K_inv = np.linalg.inv(K)

    mu_s = K_s.T @ K_inv @ Y_train
    cov_s = K_ss - K_s.T @ K_inv @ K_s

    return mu_s, cov_s


def optimize_hyper_params(kernel,X_train, Y_train):
    # -log liklihood
    def nll_fn(x, y):
        def step(theta):
            K = kernel(x, x, sigma=theta[0], l=theta[1], sigma_f=theta[2], p=theta[3])
            return np.sum(np.log(np.diagonal(np.linalg.cholesky(K)))) + \
                   0.5 * y.T @ np.linalg.inv(K) @ y + \
                   0.5 * len(x) * np.log(2 * np.pi)

        return step

    # minimize -log liklihood
    res = minimize(nll_fn(X_train, Y_train), [0.01, 1, 1, 5.0],
                   bounds=((1e-5, 1e1), (1e-5, None), (1e-5, None), (1e-2, 1e1)),
                   method='L-BFGS-B')

    sigma_opt, l_opt, sigma_f_opt, p_opt = res.x
    fitted_kernel = partial(kernel, sigma_f=sigma_f_opt, l=l_opt, sigma=sigma_opt, p=p_opt)
    return fitted_kernel
    # mu_s, cov_s = posterior_predictive(X_plot, X_train, Y_train_noisy, l=l_opt, sigma_f=sigma_f_opt, p=p_opt,
    #                                    sigma=sigma_opt)
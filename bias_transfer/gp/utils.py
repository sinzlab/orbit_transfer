import numpy as np
import matplotlib.pyplot as plt

def plot_gp(mu, cov, X, samples=[], Y=None, X_train=None, Y_train=None, save=""):
    if Y is not None:
        plt.plot(X, Y, color='orange', lw=2, label='True')
    if X_train is not None and Y_train is not None:
        plt.plot(X_train, Y_train, color='red', label="Traning data")
    X = X.reshape(-1)
    mu = mu.reshape(-1)

    # cov *= 100000
    # gp_samples = np.random.multivariate_normal(mu, cov, size=1000)
    # uncertainty = 2 * np.std(gp_samples, axis=0)
    # 95% confidence interval
    uncertainty = 1.96 * np.sqrt(np.abs(np.diag(cov)))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.4)
    plt.plot(X, mu, label='Mean')

    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label='sample_{}'.format(i))

    plt.legend()
    if save:
        fig = plt.gcf()
        fig.savefig(save, dpi=200)

def plot_kernel(kernel, x):
    K_plot = kernel(x,x)
    plt.imshow(K_plot)
    # if np.count_nonzero(x) > 0:
    #     _ = plt.xticks(np.arange(0,x.shape[0], 15),x[::15,0].astype(np.int))
    #     _ = plt.yticks(np.arange(0,x.shape[0], 15),x[::15,0].astype(np.int))
    plt.colorbar()

def plot_nn(pred, X, save=""):
    plt.plot(X, pred, label='Prediction')
    plt.legend()
    if save:
        fig = plt.gcf()
        fig.savefig(save, dpi=200)

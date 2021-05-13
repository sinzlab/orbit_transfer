import torch
import numpy as np
from scipy.linalg import eigh


def kpca_numpy(X, n_components):
    """
    Implementation of kernel PCA with a linear kernel.

    Arguments:
        X: A MxN dataset as NumPy array where the samples are stored as rows (M),
           and the attributes defined as columns (N).
        n_components: The number of components to be returned.

    Returns the k eigenvectors (alphas) that correspond to the k largest
        eigenvalues (lambdas).

    """
    K = X.T @ X

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_norm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K_norm)

    # Obtaining the i eigenvectors (alphas) that corresponds to the i highest eigenvalues (lambdas).
    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
    lambdas = eigvals[-n_components:]

    vs = X @ alphas * (1 / np.sqrt(lambdas))

    return vs, lambdas


def kpca(X, n_components):
    """
    Implementation of kernel PCA with a linear kernel.

    Arguments:
        X: A MxN dataset as NumPy array where the samples are stored as rows (M),
           and the attributes defined as columns (N).
        n_components: The number of components to be returned.

    Returns the k eigenvectors (vs) that correspond to the k largest
        eigenvalues (lambdas) for the covariance matrix.

    """
    dtype = X.dtype
    K = X.T @ X

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = torch.ones((N, N), dtype=dtype) / N
    K_norm = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = torch.symeig(K_norm, eigenvectors=True)

    #     eigvals = eigvals[:,0]  # use only the real part
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[idx]

    # Obtaining the i eigenvectors (alphas) that corresponds to the i highest eigenvalues (lambdas).
    #     alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    alphas = eigvecs[:, :n_components]
    lambdas = eigvals[:n_components]

    vs = X @ alphas * (1 / torch.sqrt(lambdas))

    return vs, lambdas

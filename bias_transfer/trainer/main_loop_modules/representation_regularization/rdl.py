from torch import nn
import torch

from . import RepresentationRegularization
from nntransfer.trainer.utils import arctanh


class RDL(RepresentationRegularization):
    @staticmethod
    def centering(K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=K.device)
        I = torch.eye(n, device=K.device)
        H = I - unit / n

        return torch.mm(
            torch.mm(H, K), H
        )  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
        # return np.dot(H, K)  # KH

    @staticmethod
    def rbf(X, sigma=None):
        GX = torch.dot(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    @staticmethod
    def kernel_HSIC(X, Y, sigma):
        return torch.sum(
            RDL.centering(RDL.rbf(X, sigma)) * RDL.centering(RDL.rbf(Y, sigma))
        )

    @staticmethod
    def linear_HSIC(X, Y):
        L_X = torch.mm(X, X.T)
        L_Y = torch.mm(Y, Y.T)
        return torch.sum(RDL.centering(L_X) * RDL.centering(L_Y))

    @staticmethod
    def linear_CKA(X, Y):
        hsic = RDL.linear_HSIC(X, Y)
        var1 = torch.sqrt(RDL.linear_HSIC(X, X))
        var2 = torch.sqrt(RDL.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    @staticmethod
    def kernel_CKA(X, Y, sigma=None):
        hsic = RDL.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(RDL.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(RDL.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

    @staticmethod
    def compute_mse_matrix(x, y=None):
        """
        see: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y = x
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return dist

    @staticmethod
    def compute_rdm(x, dist_measure="corr"):
        x_flat = x.flatten(1, -1)
        centered = x_flat - x_flat.mean(dim=0).view(
            1, -1
        )  # centered by mean over images
        if dist_measure == "corr":
            result = (centered @ centered.transpose(0, 1)) / torch.ger(
                torch.norm(centered, 2, dim=1), torch.norm(centered, 2, dim=1)
            )  # see https://de.mathworks.com/help/images/ref/corr2.html
        else:
            result = RDL.compute_mse_matrix(centered)
        return result

    @staticmethod
    def rdm_comparison(x, y, criterion, dist_measure="corr", use_arctanh=False):
        rdm_x = RDL.compute_rdm(x, dist_measure).flatten()
        rdm_y = RDL.compute_rdm(y, dist_measure).flatten()
        rdm_x = rdm_x.triu(diagonal=1)
        rdm_y = rdm_y.triu(diagonal=1)
        if use_arctanh:
            rdm_x = arctanh(rdm_x)
            rdm_y = arctanh(rdm_y)
        return criterion(rdm_x, rdm_y)

    def __init__(self, trainer):
        super().__init__(trainer, name="RDL")
        self.criterion = nn.MSELoss()
        self.dist_measure = self.config.regularization.get("dist_measure")

    def rep_distance(self, output, target):
        if self.dist_measure == "CKA":
            return RDL.linear_CKA(output, target)
        else:
            return RDL.rdm_comparison(
                output,
                target,
                self.criterion,
                self.dist_measure,
                self.config.regularization.get("use_arctanh"),
            )

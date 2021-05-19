import math

from torch import nn
import torch
import torch.nn.functional as F


from .representation import RepresentationRegularization


class FunctionDistance(RepresentationRegularization):
    def __init__(self, trainer):
        super().__init__(trainer, name="FD")
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.T = self.config.regularization.get("softmax_temp", 1.0)
        self.use_softmax = self.config.regularization.get("use_softmax", True)
        self.eps = self.config.regularization.get("cov_eps", 1e-12)
        self.marginalize_over_hidden = self.config.regularization.get(
            "marginalize_over_hidden", True
        )
        self.regularize_mean = self.config.regularization.get("regularize_mean", True)
        self.add_determinant = self.config.regularization.get("add_determinant", False)

    def pre_forward(self, model, inputs, task_key, shared_memory):
        self.inputs = inputs.flatten()
        return super().pre_forward(model, inputs, task_key, shared_memory)

    def rep_distance(self, output, target, targets, rep_name):
        if self.use_softmax:
            output = F.softmax(output / self.T, dim=1)
        # Invert Covariance matrix for full training set
        A, B = targets.get(f"{rep_name}_cov_A"), targets.get(f"{rep_name}_cov_B")
        if A is not None and B is not None:
            A, B = A.reshape((-1, A.shape[2])), B.reshape((-1, B.shape[2]))
            eps_inv = (1 / self.eps) * torch.eye(A.shape[0], device=self.device)
            importance = (eps_inv - A @ B.T).type(torch.FloatTensor).to(self.device)
            d = (output - target).reshape(-1)
            fd_loss = d @ importance @ d.T
            return fd_loss / A.shape[0]  # normalize by (train_samples * out_classes)
        # Invert Covariance matrix for batch only
        V = targets.get(f"{rep_name}_cov_V")
        if V is not None:
            rep_name_ = rep_name.replace(".", "__")

            if self.regularize_mean:
                d = output - target
            else:
                d = output

            if self.marginalize_over_hidden:
                V = V.reshape((V.shape[0], -1)).double()  # N x (MD)
                d = d.mean(dim=1).double()
            else:
                V = V.reshape((-1, V.shape[2])).double()  # ND x M
                d = d.reshape(-1).double()
            V = (V - torch.mean(V, dim=1, keepdim=True)) / math.sqrt(V.shape[1])

            if hasattr(self.trainer.model, f"{rep_name_}_cov_lambdas"):
                lambdas = self.trainer.model.__getattribute__(
                    f"{rep_name_}_cov_lambdas"
                )
            else:
                lambdas = torch.ones(V.shape[-1], device=self.device).double()

            if V.shape[0] > V.shape[1]:
                eps_inv = 1 / self.eps
                fd_loss = (d * eps_inv) @ d.T
                fd_loss -= (
                    d
                    @ (eps_inv * V)
                    @ torch.inverse(torch.diag(1 / lambdas) + (V.T * eps_inv) @ V)
                    @ (V.T * eps_inv)
                    @ d.T
                )
                if self.add_determinant:
                    n, m = V.shape
                    logdet = n * math.log(self.eps) + torch.logdet(
                        torch.eye(m, device=self.device) + (V.T * 1 / self.eps) @ V
                    )

                    fd_loss += logdet  # + n * math.log(2 * math.pi)
            else:
                importance = torch.inverse(
                    V @ V.T
                    + self.eps * torch.eye(V.shape[0], device=self.device).double()
                )
                fd_loss = d @ importance @ d.T
                if self.add_determinant:
                    n, m = V.shape
                    fd_loss += torch.logdet(
                        torch.eye(n, device=self.device) * self.eps + V @ V.T
                    )
            return (
                fd_loss.float() / V.shape[0]
            )  # normalize by (train_samples * out_classes)
        var = targets.get(f"{rep_name}_var")
        if var is not None:
            importance = 1 / (var + self.eps)
        else:
            importance = 1
        fd_loss = (importance * (output - target) ** 2).mean()
        return fd_loss

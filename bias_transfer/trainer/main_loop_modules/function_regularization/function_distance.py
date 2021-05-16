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
        self.marginalize_over_hidden = self.config.regularization.get("marginalize_over_hidden", True)
        self.regularize_mean = self.config.regularization.get("regularize_mean", True)

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

            if self.marginalize_over_hidden:
                V = V.reshape((V.shape[0],-1)).double()
            else:
                V = V.reshape((-1, V.shape[2])).double()
            V = (V - torch.mean(V, dim=1, keepdim=True)) / math.sqrt(V.shape[1])

            if hasattr(self.trainer.model, f"{rep_name_}_cov_lambdas"):
                lambdas = self.trainer.model.__getattribute__(
                    f"{rep_name_}_cov_lambdas"
                )
            else:
                lambdas = torch.ones(V.shape[-1], device=self.device).double()

            eps_inv = 1 / self.eps
            importance = eps_inv * torch.eye(
                V.shape[0], device=self.device
            ).double() - (eps_inv * V) @ torch.inverse(
                torch.diag(1 / lambdas) + (V.T * eps_inv) @ V
            ) @ (
                V.T * eps_inv
            )
            importance = importance.type(torch.FloatTensor).to(self.device)

            # n = output.shape[0]
            # covariance = torch.zeros((n,n), device=self.device)
            # for i in range(n):
            #     # covariance[i, :] = torch.exp(-2 * torch.sin(math.pi * (self.inputs[i] - self.inputs)) ** 2)
            #     covariance[i,:] = torch.cos(self.inputs[i] - self.inputs)
            # covariance += torch.eye(n, device=self.device) * 0.1
            # importance = torch.inverse(covariance)

            if self.regularize_mean:
                d = output - target
            else:
                d = output

            if self.marginalize_over_hidden:
                d = d.sum(dim=1)
            else:
                d = d.reshape(-1)

            fd_loss = d @ importance @ d.T
            # loss_without_det = fd_loss

            # compute determinant:
            # n, m = V.shape
            # logdet = math.log(self.eps ** n) + torch.logdet(
            #     torch.eye(m, device=self.device) + (V.T * eps_inv) @ V
            # )
            #
            # fd_loss += logdet + n * math.log(2 * math.pi)

            # print("Regularizer: ", loss_without_det.item(), 0.5 * fd_loss.item(), logdet.item())

            return fd_loss / V.shape[0]  # normalize by (train_samples * out_classes)
        var = targets.get(f"{rep_name}_var")
        if var is not None:
            importance = 1 / (var + self.eps)
        else:
            importance = 1
        fd_loss = (importance * (output - target) ** 2).mean()
        return fd_loss

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
        self.log_var =     torch.tensor(
                self.config.regularization.get("log_var", 0), device=self.device
            )
        if self.config.regularization.get("learn_log_var", False):
            self.log_var = nn.Parameter(self.log_var)
            trainer.model._model.fd_log_var = self.log_var
            trainer.optimizer.param_groups[0]["params"].append(self.log_var)
        self.marginalize_over_hidden = self.config.regularization.get(
            "marginalize_over_hidden", True
        )
        self.regularize_mean = self.config.regularization.get("regularize_mean", True)
        self.add_determinant = self.config.regularization.get("add_determinant", False)
        objectives = {
            "Training": {"FD": {"std": 0}},
            "Validation": {"FD": {"std": 0}},
            "Test": {"FD": {"std": 0}},
        }
        self.tracker.add_objectives(objectives, init_epoch=True)

    def pre_forward(self, model, inputs, task_key, shared_memory):
        self.inputs = inputs.flatten()
        return super().pre_forward(model, inputs, task_key, shared_memory)

    def rep_distance(self, output, target):
        if self.use_softmax:
            output = F.softmax(output / self.T, dim=1)

        batch_size, hidden_dim, ensemble_members = target.shape
        if self.marginalize_over_hidden:
            V = target.reshape((batch_size, -1)).double()  # N x (MD)
        else:
            V = target.reshape((-1, ensemble_members)).double()  # ND x M

        if self.regularize_mean:
            mean = torch.mean(V, dim=-1, keepdim=True)  # ? x 1
            if not self.marginalize_over_hidden:
                output = output.reshape(1, -1)  # 1 x MD
                mean = mean.T  # 1 x MD
            d = output.double() - mean
        else:
            d = output.double()

        if self.marginalize_over_hidden:
            d = d.T
            margin_dim = d.shape[0]
        else:
            d = d.reshape(1, -1)
            margin_dim = 1

        V = (V - torch.mean(V, dim=1, keepdim=True)) / math.sqrt(V.shape[1])

        if V.shape[0] > V.shape[1]:
            var_inv = torch.exp(-self.log_var)
            importance_left = var_inv * V
            importance_middle = torch.inverse(
                torch.eye(V.shape[-1], device=self.device).double()
                + (V.T * var_inv) @ V
            )
            fd_loss = torch.zeros([], device=self.device)
            fd_loss += torch.trace((d * var_inv) @ d.T)
            fd_loss -= torch.trace(
                d @ importance_left @ importance_middle @ importance_left.T @ d.T
            )
            if self.add_determinant:
                n, m = V.shape
                logdet = n * self.log_var + torch.logdet(
                    torch.eye(m, device=self.device) + (V.T * var_inv) @ V
                )
                fd_loss += margin_dim * logdet
        else:
            importance = torch.inverse(
                V @ V.T
                + torch.exp(self.log_var)
                * torch.eye(V.shape[0], device=self.device).double()
            )

            fd_loss = torch.zeros([], device=self.device)
            fd_loss += torch.trace(d @ importance @ d.T)
            if self.add_determinant:
                n, m = V.shape
                logdet = torch.logdet(
                    torch.eye(n, device=self.device) * torch.exp(self.log_var) + V @ V.T
                )
                fd_loss += margin_dim * logdet

        self.tracker.log_objective(
            (torch.exp(self.log_var) ** 0.5).item() * batch_size,
            (
                self.mode,
                "FD",
                "std",
            ),
        )
        return fd_loss.float() / hidden_dim
        # return fd_loss.float()#/ (
        # V.shape[0] * hidden_dim
        # )  # normalize by (train_samples * out_classes)
        #
        # var = targets.get(f"{rep_name}_var")
        # if var is not None:
        #     importance = 1 / (var + self.eps)
        # else:
        #     importance = 1
        # fd_loss = (importance * (output - target) ** 2).mean()
        # return fd_loss

import math

from torch import nn
import torch


class MSELikelihood(nn.Module):
    def __init__(self, model, log_var=0.0):
        super().__init__()
        self.log_var = nn.Parameter(
            torch.ones((1,), dtype=torch.float32, requires_grad=True) * log_var
        )
        model._model.log_var = self.log_var

    def forward(self, input, target):
        precision = torch.exp(-self.log_var)
        diff = (input - target) ** 2.0
        loss = torch.sum(precision * diff + self.log_var, -1)
        return torch.sum(loss)


class CELikelihood(nn.CrossEntropyLoss):
    def __init__(self, model, log_var=0.0, *args, **kwargs):
        super().__init__(reduction="none", *args, **kwargs)
        self.log_var = nn.Parameter(
            torch.ones((1,), dtype=torch.float32, requires_grad=True) * log_var
        )
        model._model.log_var = self.log_var

    def forward(self, input, target):
        precision = torch.exp(-self.log_var)
        ce = super().forward(input, target)
        loss = torch.sum(precision * ce + 0.5 * self.log_var, -1)
        return torch.sum(loss)

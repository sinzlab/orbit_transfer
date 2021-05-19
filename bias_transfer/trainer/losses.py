from torch import nn
import torch


class MSELikelihood(nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_var = nn.Parameter(torch.zeros([], dtype=torch.float32))

    def forward(self, input, target):
        l = super().forward(input, target)
        l *= 0.5 * torch.exp(-self.log_var)
        l += 0.5 * self.log_var
        return l


class CELikelihood(nn.Module):
    def __init__(self, *args, **kwargs):
        self.loss = nn.CrossEntropyLoss(*args, **kwargs)
        self.log_var = nn.Parameter(torch.zeros([], dtype=torch.float32))

    def forward(self, input, target):
        l = self.loss(input, target)
        l *= torch.exp(-self.log_var)
        l += 0.5 * self.log_var
        return l

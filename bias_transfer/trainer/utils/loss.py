import math

import torch
from torch.nn import Module


class CircularDistanceLoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super(CircularDistanceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        distances = 0.5 * (
            torch.ones_like(input) - torch.cos(math.pi * (input - target) / 180)
        )
        if self.reduction == "mean":
            return distances.mean()
        else:
            return distances.sum()

import math

from torch import nn
import torch
import torch.nn.functional as F

from bias_transfer.models.learned_equiv import is_square
from .representation import RepresentationRegularization


class AttentionTransfer(RepresentationRegularization):
    def __init__(self, trainer):
        super().__init__(trainer, name="Attention")

    def to_spatial(self, x):
        shape = x.shape
        # if len(shape) < 3 and shape[1] == self.output_size:  # We are in the final layer
        #     x = x.unsqueeze(1)
        if len(shape) < 4:
            # find decomposition into channels and spatial dimensions that works
            h = shape[1]
            c = 1
            while not is_square(h // c):
                c += 1
                if h / c < 1:
                    raise ValueError("Hidden dimension not divisible")
            s = int(math.sqrt(h // c))
            x = x.reshape(-1, c, s, s)
        elif len(shape) > 4:
            b, _, _, w, h = shape
            x = x.reshape(b, -1, w, h)
        return x

    def rep_distance(self, output, target, *args, **kwargs):
        output = self.to_spatial(output)
        target = self.to_spatial(target)
        attention_map_t = torch.sum(torch.abs(target) ** 2, dim=1, keepdim=True)
        attention_map_s = torch.sum((torch.abs(output) ** 2), dim=1, keepdim=True)
        if attention_map_s.shape[-1] != attention_map_t.shape[-1]:
            attention_map_t = F.interpolate(
                attention_map_t, attention_map_s.shape[-2:], None, mode="nearest"
            )
        attention_map_s = attention_map_s.flatten(1)
        attention_map_t = attention_map_t.flatten(1)
        attention_map_t = attention_map_t / torch.norm(
            attention_map_t, dim=1, keepdim=True
        )
        attention_map_s = attention_map_s / torch.norm(
            attention_map_s, dim=1, keepdim=True
        )
        loss = F.mse_loss(attention_map_s, attention_map_t, reduction="mean")
        return loss

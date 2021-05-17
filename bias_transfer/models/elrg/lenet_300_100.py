import torch
import torch.nn as nn
import torch.nn.functional as F

from bias_transfer.models.elrg.linear import ELRGLinear
from nntransfer.models.utils import concatenate_flattened


class LeNet300100(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_height: int = 28,
        input_width: int = 28,
        input_channels: int = 1,
        dropout: float = 0.0,
        rank: int = 1,
        alpha: float = None,
        train_var: bool = True,
        initial_var: float = 1e-12,
    ):
        super(LeNet300100, self).__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else 1 / rank
        self.input_size = (input_height, input_width)
        self.flat_input_size = input_width * input_height * input_channels
        self.fc1 = ELRGLinear(
            self.flat_input_size,
            300,
            rank=rank,
            alpha=self.alpha,
            train_var=train_var,
            initial_posterior_var=initial_var,
        )
        self.fc2 = ELRGLinear(
            300,
            100,
            rank=rank,
            alpha=self.alpha,
            train_var=train_var,
            initial_posterior_var=initial_var,
        )
        self.fc3 = ELRGLinear(
            100,
            num_classes,
            rank=rank,
            alpha=self.alpha,
            train_var=train_var,
            initial_posterior_var=initial_var,
        )
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x, num_samples=1):
        x = x.view(x.size(0), self.flat_input_size)
        y = []
        for s in range(num_samples):
            z = F.relu(self.fc1(x))
            z = self.dropout(z) if self.dropout else z
            z = F.relu(self.fc2(z))
            z = self.dropout(z) if self.dropout else z
            y.append(self.fc3(z))
        return torch.cat(y)

    def get_parameters(self, name, keep_first_dim=False):
        return concatenate_flattened(
            [
                self.fc1._parameters.get(f"w_{name}"),
                self.fc2._parameters.get(f"w_{name}"),
                self.fc3._parameters.get(f"w_{name}"),
                self.fc1._parameters.get(f"b_{name}"),
                self.fc2._parameters.get(f"b_{name}"),
                self.fc3._parameters.get(f"b_{name}"),
            ],
            keep_first_dim=keep_first_dim,
        )



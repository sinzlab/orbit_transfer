import torch
import torch.nn as nn
import torch.nn.functional as F

from bias_transfer.models.bayes_by_backprop.linear import BayesLinear
from nntransfer.models.utils import concatenate_flattened


class LeNet300100(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_height: int = 28,
        input_width: int = 28,
        input_channels: int = 1,
        dropout: float = 0.0,
    ):
        super(LeNet300100, self).__init__()
        self.input_size = (input_height, input_width)
        self.flat_input_size = input_width * input_height * input_channels
        self.fc1 = BayesLinear(self.flat_input_size, 300)
        self.fc2 = BayesLinear(300, 100)
        self.fc3 = BayesLinear(100, num_classes)
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

    def get_parameters(self, name):
        if "prior" in name:
            return concatenate_flattened(
                [
                    self.fc1.__getattribute__(f"w_{name}"),
                    self.fc2.__getattribute__(f"w_{name}"),
                    self.fc3.__getattribute__(f"w_{name}"),
                    self.fc1.__getattribute__(f"b_{name}"),
                    self.fc2.__getattribute__(f"b_{name}"),
                    self.fc3.__getattribute__(f"b_{name}"),
                ]
            )
        else:
            return concatenate_flattened(
                [
                    self.fc1._parameters.get(f"w_{name}"),
                    self.fc2._parameters.get(f"w_{name}"),
                    self.fc3._parameters.get(f"w_{name}"),
                    self.fc1._parameters.get(f"b_{name}"),
                    self.fc2._parameters.get(f"b_{name}"),
                    self.fc3._parameters.get(f"b_{name}"),
                ]
            )

    def to(self, *args, **kwargs):
        """
        Our prior tensors are registered as buffers but the way we access them
        indirectly (through tuple attributes on the model) is causing problems
        because when we use `.to()` to move the model to a new device, the prior
        tensors get moved (because they're registered as buffers) but the
        references in the tuples don't get updated to point to the new moved
        tensors. This has no effect when running just on a cpu but breaks the
        model when trying to run on a gpu. There are a million nicer ways of
        working around this problem, but for now the easiest thing is to do
        this: override the `.to()` method and manually update our references to
        prior tensors.
        """
        self = super().to(*args, **kwargs)
        for fc in [self.fc1, self.fc2, self.fc3]:
            fc.w_prior_mean = fc.w_prior_mean.to(*args, **kwargs)
            fc.w_prior_log_var = fc.w_prior_log_var.to(*args, **kwargs)
            fc.b_prior_mean = fc.b_prior_mean.to(*args, **kwargs)
            fc.b_prior_log_var = fc.b_prior_log_var.to(*args, **kwargs)
        return self

    def reset_for_new_task(self):
        for fc in [self.fc1, self.fc2, self.fc3]:
            fc.reset_for_new_task()



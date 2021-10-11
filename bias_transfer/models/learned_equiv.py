import math

import torch
from torch import nn
import torch.nn.functional as F


def is_square(apositiveint):
    #https://stackoverflow.com/a/2489519
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True



class LearnedEquivariance1D(nn.Module):
    def __init__(self, kernel_size=5, group_size=40, num_layers=0, output_size=10):
        super().__init__()
        self.kernels = torch.nn.Parameter(torch.randn((group_size, kernel_size)))
        if num_layers:
            self.layer_transforms = nn.Sequential(
                *[
                    nn.Linear(kernel_size, kernel_size, bias=True)
                    for _ in range(num_layers - 1)
                ],
                nn.Linear(kernel_size, output_size, bias=True),
            )
        else:
            self.layer_transforms = None
        self.full_padding = self.get_padding(kernel_size)
        self.reduced_padding = self.get_padding(output_size)

    def get_padding(self, kernel_size):
        # copied from: https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d
        kernel_size = (kernel_size,)
        dilation = (1,)
        padding = [0, 0] * len(kernel_size)
        for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            padding[2 * i] = left_pad
            padding[2 * i + 1] = total_padding - left_pad
        return padding

    def forward(self, x, g=None, l=0, n=1):
        if g is None:
            return 0

        shape = x.shape
        # print("x", x.shape)
        x = x.view(shape[0], 1, -1)
        # print("x reshaped", x.shape)

        # print("G in", g.shape)
        g %= self.kernels.shape[0]

        x = x.permute(
            1, 0, 2
        )  # switch channel with batch dimension to apply different kernel to each sample (based on g)
        # print("x permuted", x.shape)
        kernel = self.kernels[g]
        # print("kernel", kernel.shape)
        if self.layer_transforms is not None and l > 0:
            kernel = self.layer_transforms[: l + 1](kernel)
            if l == len(self.layer_transforms) - 1:
                padding = self.reduced_padding
            else:
                padding = self.full_padding
        else:
            padding = self.full_padding
        # print("Kernel transformed", kernel.shape)
        kernel = kernel.unsqueeze(
            1
        )  # [batch_size, 1, k] -> [out_channels, in_channels/groups, k]
        # print("Kernel unsqueeze", kernel.shape)
        # print("G", g)
        # print("Kernel", kernel)
        for i in range(n):
            x = F.conv1d(
                F.pad(x, padding, mode="circular"),
                kernel,
                groups=kernel.shape[0],
            )
        # print("out", out.shape)
        x = x.permute(1, 0, 2)
        return x.reshape(shape)


class LearnedEquivariance(nn.Module):
    def __init__(self, kernel_size=5, group_size=40, num_layers=0, output_size=10):
        super().__init__()
        self.kernels = torch.nn.Parameter(
            torch.randn((group_size, kernel_size, kernel_size))
        )
        if num_layers:
            self.layer_transforms = nn.Sequential(
                *[
                    nn.Linear(kernel_size ** 2, kernel_size ** 2, bias=True)
                    for _ in range(num_layers - 1)
                ],
                nn.Linear(kernel_size ** 2, output_size, bias=True),
            )
        else:
            self.layer_transforms = None
        self.full_padding = self.get_padding((kernel_size,kernel_size))
        self.reduced_padding = self.get_padding((output_size,))
        self.output_size = output_size

    def get_padding(self, kernel_size):
        # copied from: https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d
        dilation = (1,1)
        padding = [0, 0] * len(kernel_size)
        for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            padding[2 * i] = left_pad
            padding[2 * i + 1] = total_padding - left_pad
        return padding

    def forward(self, x, g=None, l=0, n=1):
        if g is None:
            return 0

        shape = x.shape
        # print("x", x.shape)
        if len(shape) < 3 and shape[1] == self.output_size:  # We are in the final layer
            x = x.unsqueeze(1)
        elif len(shape) < 3:
            # find decomposition into channels and spatial dimensions that works
            h = shape[1]
            c = 1
            while not is_square(h // c):
                c += 1
                if h / c < 1:
                    raise ValueError("Hidden dimension not divisible")
            s = int(math.sqrt(h // c))
            x = x.reshape(-1, c, s, s)

        # x = x.view(shape[0], 1, )  # TODO make this adaptable to more than 1 input channel
        # print("x reshaped", x.shape)

        # print("G in", g.shape)
        g %= self.kernels.shape[0]

        x = x.transpose(
            0, 1
        )  # switch channel with batch dimension to apply different kernel to each sample (based on g)
        # print("x permuted", x.shape)
        kernel = self.kernels[g]
        # print("kernel", kernel.shape)
        padding = self.full_padding
        conv_op = F.conv2d
        if self.layer_transforms is not None and l > 0:
            kernel_shape = kernel.shape
            kernel = self.layer_transforms[: l + 1](kernel.flatten(1))
            if l == len(self.layer_transforms) - 1:
                padding = self.reduced_padding
                conv_op = F.conv1d
            else:
                kernel = kernel.reshape(kernel_shape)
        # print("Kernel transformed", kernel.shape)
        kernel = kernel.unsqueeze(
            1
        )  # [batch_size, 1, k, k] -> [out_channels, in_channels/groups, k, k]
        # print("Kernel unsqueeze", kernel.shape)
        # print("G", g.shape)
        # print("Kernel", kernel.shape)
        for i in range(n):
            # print("x before", x.shape)
            x_padded = F.pad(x, padding, mode="circular")  # Troublesome if spatial dimension smaller than padding! (for cirular padding)
            # print("x_padded", x_padded.shape)
            x = conv_op(
                x_padded,
                kernel,
                groups=kernel.shape[0],
            )
        # print("out", x.shape)
        x = x.transpose(0, 1)
        return x.reshape(shape)


def equiv_builder(seed: int, config):
    model = LearnedEquivariance(
        kernel_size=config.kernel_size,
        group_size=config.group_size,
        num_layers=config.num_layers,
        output_size=config.output_size,
    )
    return model

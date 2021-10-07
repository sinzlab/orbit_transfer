from bias_transfer.configs.model.mnist_1d import MNIST1DModelConfig
from nntransfer.models.utils import get_model_parameters
from nntransfer.models.wrappers.intermediate_layer_getter import IntermediateLayerGetter
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



class StaticLearnedEquivariance(nn.Module):
    def __init__(self, kernel_size=5, group_size=40, layers=0):
        super().__init__()
        self.kernels = torch.nn.Parameter(torch.randn((group_size, kernel_size)))
        if layers:
            self.layer_transforms = nn.Sequential(
                *[
                    nn.Linear(kernel_size, kernel_size, bias=True)
                    for _ in range(layers - 1)
                ],
                nn.Linear(kernel_size, 10, bias=True),
            )
        else:
            self.layer_transforms = None
        self.full_padding = self.get_padding(kernel_size)
        self.reduced_padding = self.get_padding(10)

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
        if len(x.shape) < 3:
            x = x.view(-1, 1, x.shape[-1])

        # print("G in", g)
        g %= self.kernels.shape[0]

        x = x.permute(
            1, 0, 2
        )  # switch channel with batch dimension to apply different kernel to each sample (based on g)
        kernel = self.kernels[g]
        if self.layer_transforms is not None and l > 0:
            kernel = self.layer_transforms[: l + 1](kernel)
            if l == len(self.layer_transforms) - 1:
                padding = self.reduced_padding
            else:
                padding =self.full_padding
        else:
            padding = self.full_padding
        kernel = kernel.unsqueeze(
            1
        )  # [batch_size, 1, k] -> [out_channels, in_channels/groups, k]
        # print("G", g)
        # print("Kernel", kernel)
        for i in range(n):
            x = F.conv1d(
                F.pad(x, padding, mode="circular"),
                kernel,
                groups=kernel.shape[0],
            )
        # print("out", out.shape)
        return x.permute(1, 0, 2)

class LearnedEquivariance(nn.Module):
    def __init__(self, kernel_size=5, group_size=40, layers=0):
        super().__init__()
        self.group_size = group_size
        self.kernels = torch.nn.Parameter(
            torch.randn((group_size * 2 + 1, kernel_size))
        )
        if layers:
            self.layer_transforms = nn.Sequential(
                *[
                    nn.Linear(kernel_size, kernel_size, bias=True)
                    for _ in range(layers - 1)
                ],
                nn.Linear(kernel_size, 10, bias=True),
            )
        else:
            self.layer_transforms = None
        self.full_padding = self.get_padding(kernel_size)
        self.reduced_padding = self.get_padding(10)

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

    def forward(self, x, g=None, l=0):
        if g is None:
            return 0
        if len(x.shape) < 3:
            x = x.view(-1, 1, x.shape[-1])
        x = x.permute(
            1, 0, 2
        )  # switch channel with batch dimension to apply different kernel to each sample (based on g)
        if self.layer_transforms is not None and l > 0:
            if l == len(self.layer_transforms) - 1:
                padding = self.reduced_padding
            else:
                padding = self.full_padding
        else:
            padding = self.full_padding
        # g %= self.kernels.shape[0]
        # print("G in", g)

        # print("g", g)
        iterations = torch.abs(g / self.group_size).to("cuda").reshape(-1,1,1)
        # print("iterations", iterations)
        max_iterations = torch.max(iterations)
        # print("max iterations", max_iterations)
        sign = torch.sign(g)
        # print("sign", sign)
        max_kernel = self.kernels[sign * self.group_size + self.group_size]
        identity_kernel = self.kernels[self.group_size]
        if self.layer_transforms is not None and l > 0:
            max_kernel = self.layer_transforms[: l + 1](max_kernel)
            identity_kernel = self.layer_transforms[: l + 1](identity_kernel)
        max_kernel = max_kernel.unsqueeze(
            1
        )  # [batch_size, 1, k] -> [out_channels, in_channels/groups, k]
        identity_kernel = identity_kernel.reshape(1,1,-1).expand(max_kernel.shape[0],-1,-1)
        for i in range(max_iterations):
            kernel = torch.where(i >= iterations, identity_kernel, max_kernel)
            x = F.conv1d(
                F.pad(x, padding, mode="circular"),
                kernel,
                groups=kernel.shape[0],
            )

        remainder = torch.abs(g) % self.group_size
        # print("remainder", remainder)
        kernel = self.kernels[remainder * sign + self.group_size]
        # print("kernel", kernel)
        if self.layer_transforms is not None and l > 0:
            kernel = self.layer_transforms[: l + 1](kernel)
        kernel = kernel.unsqueeze(1)
        # print("G", g)
        # print("Kernel", kernel)
        out = F.conv1d(
            F.pad(x, padding, mode="circular"),
            kernel,
            groups=kernel.shape[0],
        )
        # print("out", out.shape)
        return out.permute(1, 0, 2)


class ConvBase(nn.Module):
    def __init__(self, output_size, channels=25, linear_in=125):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.linear = nn.Linear(
            linear_in, output_size
        )  # flattened channels -> 10 (assumes input has dim 50)

    def forward(self, x, verbose=False):  # the print statements are for debugging
        x = x.view(-1, 1, x.shape[-1])
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1)  # flatten the conv features
        return self.linear(h3)  # a linear classifier goes on top


class MLPBase(nn.Module):
    def __init__(self):
        super(MLPBase, self).__init__()
        self.linear1 = nn.Linear(40, 475)
        self.linear2 = nn.Linear(475, 250)
        self.linear3 = nn.Linear(250, 125)
        self.linear4 = nn.Linear(125, 10)

    def forward(self, x):
        h = self.linear1(x).relu()
        h = self.linear2(h).relu()
        h = self.linear3(h).relu()
        return self.linear4(h)


class ConvSingleLayer(nn.Module):
    def __init__(self, channels, linear_in, padding, stride, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(
            1,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            padding_mode="circular",
        )
        self.linear = nn.Linear(
            linear_in, 10, bias=False
        )  # flattened channels -> 10 (assumes input has dim 50)
        print(
            "Initialized ConvBase model with {} parameters".format(self.count_params())
        )

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False):  # the print statements are for debugging
        x = x.view(-1, 1, x.shape[-1])
        h1 = self.conv1(x).relu()
        h1 = h1.view(h1.shape[0], -1)  # flatten the conv features
        return self.linear(h1)  # a linear classifier goes on top


from nntransfer.models.layers import LocallyConnected1d


class LCSingleLayer(nn.Module):
    def __init__(self, channels, linear_in, padding, stride, kernel_size):
        super().__init__()
        self.lc1 = LocallyConnected1d(
            1, channels, linear_in, kernel_size, stride=stride, bias=False
        )
        self.linear = nn.Linear(
            linear_in * channels, 10, bias=False
        )  # flattened channels -> 10 (assumes input has dim 50)

    def forward(self, x, verbose=False):  # the print statements are for debugging
        x = x.view(-1, 1, x.shape[-1])
        h1 = self.lc1(x).relu()
        h1 = h1.view(h1.shape[0], -1)  # flatten the conv features
        return self.linear(h1)  # a linear classifier goes on top



class FullyConvLCN(nn.Module):
    def __init__(self, channels, linear_in, padding, stride, kernel_size):
        super().__init__()
        self.linear1 = LocallyConnected1d(
            1,
            channels,
            36,
            kernel_size,
            stride=stride,
            # padding=padding,
            bias=False,
            # padding_mode="circular",
        )

        self.linear2 = LocallyConnected1d(
            channels,
            10,
            32,
            kernel_size,
            stride=stride,
            # padding=padding,
            bias=False,
            # padding_mode="circular",
        )

    def forward(self, x, verbose=False):  # the print statements are for debugging
        x = x.view(-1, 1, x.shape[-1])
        h1 = self.linear1(x).relu()
        h2 = self.linear2(h1)
        return torch.max(h2, dim=2)[0]


class FullyConv(nn.Module):
    def __init__(self, channels, linear_in, padding, stride, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(
            1,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            channels,
            10,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            padding_mode="circular",
        )

    def forward(self, x, verbose=False):  # the print statements are for debugging
        x = x.view(-1, 1, x.shape[-1])
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1)
        return torch.max(h2, dim=2)[0]


class SimpleFullyConv(nn.Module):
    def __init__(self, input_size, channels, kernel_size, stride, padding, layers=2):
        super().__init__()
        conv_stack = [
            nn.Conv1d(
                1,
                channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                padding_mode="circular",
            ),
            nn.ReLU(),
        ]
        for l in range(layers - 2):
            conv_stack.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    padding_mode="circular",
                )
            )
            conv_stack.append(nn.ReLU())
        conv_stack.append(
            nn.Conv1d(
                channels,
                10,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                padding_mode="circular",
            )
        )
        self.conv_stack = nn.Sequential(*conv_stack)

    def forward(self, x, verbose=False):  # the print statements are for debugging
        x = x.view(-1, 1, x.shape[-1])
        h = self.conv_stack(x)
        out = torch.max(h, dim=2)[0]
        return out


class SimpleConv(nn.Module):
    def __init__(self, input_size, channels, kernel_size, stride, padding, layers=2):
        super().__init__()
        conv_stack = [
            nn.Conv1d(
                1,
                channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                padding_mode="circular",
            ),
            nn.ReLU(),
        ]
        for l in range(layers - 2):
            conv_stack.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    padding_mode="circular",
                )
            )
            conv_stack.append(nn.ReLU())
        self.conv_stack = nn.Sequential(*conv_stack)
        self.linear = nn.Linear(input_size * channels, 10, bias=False)

    def forward(self, x, verbose=False):  # the print statements are for debugging
        x = x.view(-1, 1, x.shape[-1])
        h = self.conv_stack(x)
        out = self.linear(h.flatten(1))
        return out


class SimpleFC(nn.Module):
    def __init__(self, input_size, channels, layers=2):
        super().__init__()
        linear_stack = [
            nn.Linear(input_size, input_size * channels, bias=False),
            nn.ReLU(),
        ]
        for l in range(layers - 2):
            linear_stack.append(
                nn.Linear(input_size * channels, input_size * channels, bias=False)
            )
            linear_stack.append(nn.ReLU())
        linear_stack.append(
            nn.Linear(input_size * channels, input_size * 10, bias=False)
        )
        self.linear_stack = nn.Sequential(*linear_stack)
        self.linear = nn.Linear(input_size * 10, 10, bias=False)

    def forward(self, x, verbose=False):  # the print statements are for debugging
        h = self.linear_stack(x)
        out = self.linear(h)
        return out


class FCSingleLayer(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, 10 * input_size, bias=False)

    def forward(self, x, verbose=False):  # the print statements are for debugging
        h1 = self.linear1(x).relu()
        h2 = self.linear2(h1)
        return torch.max(h2.reshape(h2.shape[0], 10, -1), dim=2)[0]


class FCwithFullyConvReadout(nn.Module):
    def __init__(self, input_size, hidden_dim, channels, kernel_size, stride, padding):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_dim, bias=False)
        self.channels = channels
        self.conv2 = nn.Conv1d(
            channels,
            10,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            padding_mode="circular",
        )

    def forward(self, x, verbose=False):  # the print statements are for debugging
        h1 = self.linear1(x).relu()
        h1 = h1.view(h1.shape[0], self.channels, -1)
        h2 = self.conv2(h1)
        return torch.max(h2, dim=2)[0]


def model_fn(data_loader, seed: int, **config):
    config = MNIST1DModelConfig.from_dict(config)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if config.type == "equivariance_transfer":
        model = LearnedEquivariance(
            kernel_size=config.kernel_size, group_size=config.group_size, layers=config.layers
        )
        get_layers = {}
    elif config.type == "static_equivariance_transfer":
        model = StaticLearnedEquivariance(
            kernel_size=config.kernel_size, group_size=config.group_size, layers=config.layers
        )
        get_layers = {}
    elif config.type == "fc_single":
        linear_in = (
            (
                (config.input_size - config.kernel_size + 2 * config.padding)
                // config.stride
            )
            + 1
        ) * config.channels  # ([(W-K+2P)/S]+1 ) * channels
        model = FCSingleLayer(config.input_size, config.hidden_dim)
        get_layers = {"linear1": "linear1", "linear2": "linear2"}
    elif config.type == "conv_single":
        linear_in = (
            (
                (config.input_size - config.kernel_size + 2 * config.padding)
                // config.stride
            )
            + 1
        ) * config.channels  # ([(W-K+2P)/S]+1 ) * channels
        model = ConvSingleLayer(
            channels=config.channels,
            linear_in=linear_in,
            padding=config.padding,
            stride=config.stride,
            kernel_size=config.kernel_size,
        )
        get_layers = {"conv1": "conv1", "conv2": "conv2"}
    elif config.type == "lc_single":
        linear_in = (
            (
                (config.input_size - config.kernel_size + 2 * config.padding)
                // config.stride
            )
            + 1
        ) * config.channels  # ([(W-K+2P)/S]+1 ) * channels
        model = LCSingleLayer(
            channels=config.channels,
            linear_in=linear_in,
            padding=config.padding,
            stride=config.stride,
            kernel_size=config.kernel_size,
        )
        get_layers = {"conv1": "conv1", "conv2": "conv2"}
    elif config.type == "fully_conv_single_lcn":
        linear_in = (
                            (
                                    (config.input_size - config.kernel_size + 2 * config.padding)
                                    // config.stride
                            )
                            + 1
                    ) * config.channels  # ([(W-K+2P)/S]+1 ) * channels
        model = FullyConvLCN(
            channels=config.channels,
            linear_in=linear_in,
            padding=config.padding,
            stride=config.stride,
            kernel_size=config.kernel_size,
        )
        get_layers = {"linear1": "linear1", "linear2": "linear2"}
    elif config.type == "fully_conv_single":
        linear_in = (
            (
                (config.input_size - config.kernel_size + 2 * config.padding)
                // config.stride
            )
            + 1
        ) * config.channels  # ([(W-K+2P)/S]+1 ) * channels
        model = FullyConv(
            channels=config.channels,
            linear_in=linear_in,
            padding=config.padding,
            stride=config.stride,
            kernel_size=config.kernel_size,
        )
        get_layers = {"conv1": "conv1", "conv2": "conv2"}
    elif config.type == "conv_base":
        model = ConvBase(output_size=config.num_classes)
        get_layers = {"conv1": "conv1", "conv2": "conv2", "conv3": "conv3"}
    elif config.type == "fc_base":
        model = MLPBase()
        get_layers = {"linear1": "linear1", "linear2": "linear2", "linear3": "linear3"}
    elif config.type == "simple_fully_conv":
        linear_in = (
            (
                (config.input_size - config.kernel_size + 2 * config.padding)
                // config.stride
            )
            + 1
        ) * config.channels  # ([(W-K+2P)/S]+1 ) * channels
        print(linear_in)
        model = SimpleFullyConv(
            input_size=config.input_size,
            channels=config.channels,
            padding=config.padding,
            stride=config.stride,
            kernel_size=config.kernel_size,
            layers=config.layers,
        )
        get_layers = {
            f"conv_stack.{l * 2}": f"conv_stack.{l * 2}" for l in range(config.layers)
        }
    elif config.type == "simple_conv":
        linear_in = (
            (
                (config.input_size - config.kernel_size + 2 * config.padding)
                // config.stride
            )
            + 1
        ) * config.channels  # ([(W-K+2P)/S]+1 ) * channels
        print(linear_in)
        model = SimpleConv(
            input_size=config.input_size,
            channels=config.channels,
            padding=config.padding,
            stride=config.stride,
            kernel_size=config.kernel_size,
            layers=config.layers,
        )
        get_layers = {
            f"conv_stack.{l*2}": f"conv_stack.{l*2}" for l in range(config.layers)
        }
    elif config.type == "simple_fc":
        model = SimpleFC(
            input_size=config.input_size, channels=config.channels, layers=config.layers
        )
        get_layers = {
            f"linear_stack.{l*2}": f"linear_stack.{l*2}" for l in range(config.layers)
        }
    print(model)
    print(get_layers)

    print("Model with {} parameters.".format(get_model_parameters(model)))
    # Add wrappers
    if get_layers:
        model = IntermediateLayerGetter(
            model, return_layers=get_layers, keep_output=True
        )
    return model

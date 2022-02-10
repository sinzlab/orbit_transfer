import math

import numpy as np
import torch
from scipy.linalg import expm
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer
from einops import rearrange, repeat
from torchvision.transforms.functional import _get_inverse_affine_matrix
from functools import reduce


def get_factors(n):
    return reduce(
        list.__add__,
        ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0),
    )


def reshape_flat_to_chw(x, four_dims=False):
    assert len(x.shape) == 2
    # find decomposition into channels and spatial dimensions that works
    h = x.shape[1]
    c = 1
    while not is_square(h // c):
        c += 1
        if h / c < 1:
            raise ValueError("Hidden dimension not divisible")
    s = int(math.sqrt(h // c))

    if four_dims:
        factors = sorted(get_factors(c))
        x = x.reshape(-1, c // factors[-1], factors[-1], s, s)
    else:
        x = x.reshape(-1, c, s, s)
    return x


def is_square(apositiveint):
    # https://stackoverflow.com/a/2489519
    if apositiveint == 1:
        return True
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True


def get_padding(kernel_size):
    # copied from: https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d
    dilation = [1] * len(kernel_size)
    padding = [0, 0] * len(kernel_size)
    for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
        total_padding = d * (k - 1)
        left_pad = total_padding // 2
        padding[2 * i] = left_pad
        padding[2 * i + 1] = total_padding - left_pad
    return padding



class LearnedEquivariance(nn.Module):
    def __init__(
        self,
        kernel_size=5,
        group_size=40,
        num_layers=0,
        output_size=10,
        gold_init=False,
        vit_input=False,
        handle_output_layer=True,
        first_layer_transform=True,
    ):
        super().__init__()
        self.kernels = torch.nn.Parameter(
            torch.randn((group_size, kernel_size, kernel_size))
        )
        self.first_layer_no_transform = not first_layer_transform
        if num_layers:
            if handle_output_layer:
                num_layers -= 1
            if self.first_layer_no_transform:
                num_layers -= 1
            self.layer_transforms = nn.ModuleList(
                [
                    nn.Linear(kernel_size ** 2, kernel_size ** 2, bias=True)
                    for _ in range(num_layers)
                ]
                + (
                    [nn.Linear(kernel_size ** 2, output_size, bias=True)]
                    if handle_output_layer
                    else []
                )
            )
        else:
            self.layer_transforms = None
        self.full_padding = get_padding((kernel_size, kernel_size))
        self.reduced_padding = get_padding((output_size,))
        self.output_size = output_size
        self.vit_input = vit_input
        self.handle_output_layer = handle_output_layer

    def reshape_input(self, x):
        shape = x.shape
        if len(shape) < 3 and shape[1] == self.output_size:  # We are in the final layer
            x = x.unsqueeze(1)
        elif len(shape) < 3:
            x = reshape_flat_to_chw(x)
        elif len(shape) > 4:
            b, _, _, w, h = shape
            x = x.reshape(b, -1, w, h)
        return x

    def forward(self, x, g=None, l=0, n=1):
        not_input = l > 0
        if self.first_layer_no_transform and not_input:
            l -= 1
        if g is None:
            return 0
        last_layer = l == len(self.layer_transforms) and self.handle_output_layer
        if self.vit_input and not_input and not last_layer:
            cls_token = x[:, -1:]
            x = x[:, :-1]
            s = x.shape[1]
            x = rearrange(
                x, "b (h w) c -> b c h w", h=int(math.sqrt(s)), w=int(math.sqrt(s))
            )
        shape = x.shape
        x = self.reshape_input(x)
        g = g % self.kernels.shape[0]

        x = x.transpose(
            0, 1
        )  # switch channel with batch dimension to apply different kernel to each sample (based on g)
        kernel = self.kernels[g]
        padding = self.full_padding
        conv_op = F.conv2d
        if (
            self.layer_transforms is not None and l > 0
        ):  # not input and in some cases not first layer
            kernel_shape = kernel.shape
            kernel = self.layer_transforms[l - 1](kernel.flatten(1))
            if last_layer:
                padding = self.reduced_padding
                conv_op = F.conv1d
            else:
                kernel = kernel.reshape(kernel_shape)
        kernel = kernel.unsqueeze(
            1
        )  # [batch_size, 1, k, k] -> [out_channels, in_channels/groups, k, k]
        for i in range(n):
            x_padded = F.pad(
                x, padding, mode="circular"
            )  # Troublesome if spatial dimension smaller than padding! (for cirular padding)
            x = conv_op(
                x_padded,
                kernel,
                groups=kernel.shape[0],
            )
        x = x.transpose(0, 1)
        x = x.reshape(shape)
        if self.vit_input and not_input and not last_layer:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = torch.cat([x, cls_token], dim=1)
        return x, None




class LearnedEquivarianceSTSimple(nn.Module):
    def __init__(
        self,
        num_layers,
        group_size,
        only_translation,
        handle_output_layer=True,
        include_channels=False,
        prevent_translation=False,
        gaussian_transform=False,
        random_transform_init=False,
        gaussian_std_init=0.1,
    ):
        super().__init__()
        assert not (only_translation and prevent_translation)
        self.only_translation = only_translation
        self.prevent_translation = prevent_translation
        if self.only_translation:
            self.transform_params = 3 if include_channels else 2
        elif self.prevent_translation:
            self.transform_params = 9 if include_channels else 4
        else:
            self.transform_params = 12 if include_channels else 6
        # self.layer_transforms = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(self.transform_params, self.transform_params, bias=True),
        #             nn.Tanh(),
        #             nn.Linear(self.transform_params, self.transform_params, bias=True),
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )
        # for transform in self.layer_transforms:
        #     transform[0].weight.data = torch.eye(self.transform_params)
        #     transform[0].bias.data = torch.zeros(self.transform_params)
        #     transform[2].weight.data = torch.eye(self.transform_params)
        #     transform[2].bias.data = torch.zeros(self.transform_params)

        if prevent_translation:
            identity_tensor = torch.eye(3 if include_channels else 2)
        else:
            identity_tensor = torch.eye(
                3 if include_channels else 2, 4 if include_channels else 3
            )
        init = torch.zeros(
            num_layers + 1,
            group_size,
            3 if include_channels else 2,
            4 if include_channels else 3,
        )  # 0
        # Gold:
        # init[1] = torch.tensor([0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0])  # 90
        # init[2] = torch.tensor([-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0])  # 180
        # init[3] = torch.tensor([0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0])  # 270
        # Random rotations:
        # for g in range(group_size):
        #     angle = np.random.uniform(0,2*math.pi)
        #     init[g] = torch.tensor([math.cos(angle), -math.sin(angle), 0, 0, math.sin(angle), math.cos(angle), 0, 0, 0, 0, 1, 0], dtype=torch.float)
        # init = torch.normal(0, 0.3, (group_size, self.transform_params))
        for l in range(num_layers + 1):
            for g in range(group_size):
                transform = identity_tensor.clone()
                angle, translate, scale, shear = self.get_init_params(
                    degrees=(0, 360),
                    translate=(0.1, 0.1),  # dx, dy
                    scale_ranges=(0.8, 1.2),  # min, max
                    shears=(0.0, 15.0, 0.0, 15.0),
                )
                translate_f = [1.0 * t for t in translate]
                m = torch.tensor(
                    _get_inverse_affine_matrix(
                        [0.0, 0.0], angle, translate_f, scale, shear
                    )
                ).reshape(2, 3)
                if include_channels:
                    transform = torch.eye(4, 4)
                    if random_transform_init:
                        transform[:2, :2] = m[:, :2]
                        transform[:2, 3:] = m[:, 2:]
                        transform[2, 3] = torch.empty(1).uniform_(-0.1, 0.1).item()
                        x_angle = float(torch.empty(1).uniform_(0.0, 360.0).item())
                        y_angle = float(torch.empty(1).uniform_(0.0, 360.0).item())
                        x_rot = math.radians(x_angle)
                        y_rot = math.radians(y_angle)
                        transform = transform @ torch.tensor(
                            [
                                [1, 0, 0, 0],
                                [0, math.cos(x_rot), -math.sin(x_rot), 0],
                                [0, math.sin(x_rot), math.cos(x_rot), 0],
                                [0,0,0,1]
                            ]
                        )
                        transform = transform @ torch.tensor(
                            [
                                [math.cos(y_rot), 0, math.sin(y_rot), 0],
                                [0, 1, 0, 0],
                                [-math.sin(y_rot), 0, math.cos(y_rot), 0],
                                [0, 0, 0, 1]
                            ]
                        )
                    transform = transform[:3]
                elif random_transform_init:
                    transform = m
                init[l][g] = transform

        init = init.flatten(2)
        self.gaussian_transform = gaussian_transform
        if self.gaussian_transform:
            self.bias = nn.Parameter(init)
            self.weight = nn.Parameter(
                torch.ones(num_layers + 1, self.transform_params) * gaussian_std_init
            )
        else:
            self.theta = nn.Parameter(init)
        # self.group_enc = nn.Parameter(torch.randn(group_size,self.transform_params))
        self.include_channels = include_channels
        self.handle_output_layer = handle_output_layer
        self.num_layers = num_layers
        self.group_size = group_size

    @staticmethod
    def get_init_params(degrees, translate, scale_ranges, shears):
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(
            torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item()
        )
        if translate is not None:
            max_dx = float(translate[0])
            max_dy = float(translate[1])
            tx = torch.empty(1).uniform_(-max_dx, max_dx).item()
            ty = torch.empty(1).uniform_(-max_dy, max_dy).item()
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(
                torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item()
            )
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    # Spatial transformer network forward function
    def forward(self, x, g, l=0, *args, **kwargs):
        if self.gaussian_transform:
            bias = self.bias[l][g]
            weight = self.weight[l]
            theta = torch.randn(bias.shape, device=weight.device) * weight + bias
        else:
            theta = self.theta[l][g]
        # if l > 0:
        #     theta = self.layer_transforms[l - 1](theta)
        # print("Theta", theta)
        if self.only_translation:
            theta = torch.cat(
                [
                    torch.eye(3 if self.include_channels else 2, device=theta.device)
                    .unsqueeze(0)
                    .repeat(theta.shape[0], 1, 1),
                    theta.unsqueeze(2),
                ],
                dim=2,
            )
        elif self.prevent_translation:
            shape = [theta.shape[0]] + [3, 3] if self.include_channels else [2, 2]
            theta = torch.cat(
                [
                    theta.view(*shape),
                    torch.zeros(shape[0], shape[1], 1, device=theta.device),
                ],
                dim=2,
            )

        elif self.include_channels:
            theta = theta.view(-1, 3, 4)
        else:
            theta = theta.view(-1, 2, 3)

        squeeze_after = False
        if len(x.shape) == 2 and l == self.num_layers and self.handle_output_layer:
            x = x.unsqueeze(1).unsqueeze(3)
        elif len(x.shape) == 2:
            x = reshape_flat_to_chw(x, self.include_channels)
            squeeze_after = self.include_channels
        if self.include_channels and len(x.shape) < 5:
            x = x.unsqueeze(2)
            squeeze_after = True
        # if self.include_channels:
        #     padding = get_padding((x.shape[2], x.shape[3], x.shape[4]))
        # else:
        #     padding = get_padding((x.shape[2], x.shape[3]))
        # x_padded = F.pad(x, padding, mode="circular")
        x_padded = x
        grid = F.affine_grid(theta, x_padded.size())
        x_padded = F.grid_sample(x_padded, grid)
        # if self.include_channels:
        #     x = x_padded[
        #         :,
        #         :,
        #         padding[4] : padding[4] + x.shape[2],
        #         padding[2] : padding[2] + x.shape[3],
        #         padding[0] : padding[0] + x.shape[4],
        #     ]
        # else:
        #     x = x_padded[
        #         :,
        #         :,
        #         padding[2] : padding[2] + x.shape[2],
        #         padding[0] : padding[0] + x.shape[3],
        #     ]
        x = x_padded
        if squeeze_after:
            x = rearrange(x, "b o c h w -> b (o c) h w")  # squeeze this dim
        return x, theta



class LearnedEquivariance1D(LearnedEquivariance):
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
        self.full_padding = get_padding(kernel_size)
        self.reduced_padding = get_padding(output_size)

    def forward(self, x, g=None, l=0, n=1):
        if g is None:
            return 0

        shape = x.shape
        x = x.view(shape[0], 1, -1)

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
                padding = self.full_padding
        else:
            padding = self.full_padding
        kernel = kernel.unsqueeze(
            1
        )  # [batch_size, 1, k] -> [out_channels, in_channels/groups, k]
        for i in range(n):
            x = F.conv1d(
                F.pad(x, padding, mode="circular"),
                kernel,
                groups=kernel.shape[0],
            )
        x = x.permute(1, 0, 2)
        return x.reshape(shape)



def equiv_builder(seed: int, config):
    if config.spatial_transformer:
        # model = LearnedEquivarianceSTSimple(
        #     patch_size=config.patch_size,
        #     num_layers=config.num_layers,
        #     group_size=config.group_size,
        #     only_translation=config.only_translation,
        #     prevent_translation=config.prevent_translation,
        #     include_channels=config.include_channels,
        #     use_layer_transforms=config.use_layer_transforms,
        # )
        model = LearnedEquivarianceSTSimple(
            num_layers=config.num_layers,
            group_size=config.group_size,
            only_translation=config.only_translation,
            prevent_translation=config.prevent_translation,
            include_channels=config.include_channels,
            gaussian_transform=config.gaussian_transform,
            random_transform_init=config.random_transform_init,
            gaussian_std_init=config.gaussian_std_init,
        )
    else:
        model = LearnedEquivariance(
            kernel_size=config.kernel_size,
            group_size=config.group_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
            vit_input=config.vit_input,
            first_layer_transform=config.first_layer_transform,
        )
    return model

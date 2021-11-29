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


class LearnedEquivarianceFC(nn.Module):
    def __init__(
        self,
        group_size=40,
        layer_sizes=(),
    ):
        super().__init__()
        self.layer_transforms = nn.ParameterList(
            [nn.Parameter(torch.randn((group_size, l, l))) for l in layer_sizes]
        )

    def forward(self, x, g=None, l=0, n=1):
        if g is None:
            return 0

        shape = x.shape
        x = x.flatten(1).unsqueeze(1)  # [batch_size, 1, l]
        g %= self.layer_transforms[0].shape[0]

        transforms = self.layer_transforms[l][g]  # [batch_size, l, l]
        # apply different params to each sample (based on g)
        x = torch.bmm(x, transforms)  # [batch_size, 1, l]
        return x.reshape(shape)


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
    ):
        super().__init__()
        self.kernels = torch.nn.Parameter(
            torch.randn((group_size, kernel_size, kernel_size))
        )
        if num_layers:
            if handle_output_layer:
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
        if g is None:
            return 0
        last_layer = l == len(self.layer_transforms) and self.handle_output_layer
        if self.vit_input and l != 0 and not last_layer:
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
        if self.layer_transforms is not None and l > 0:
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
        if self.vit_input and l != 0 and not last_layer:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = torch.cat([x, cls_token], dim=1)
        return x, None


class LearnedEquivarianceUNet(LearnedEquivariance):
    def __init__(
        self,
        base_kernel_size,
        spatial_kernel_size,
        channel_kernel_size,
        spatial_pooling,
        channel_pooling,
        latent_size=5,
        num_layers=10,
    ):
        super().__init__()
        self.base_kernels_spatial = nn.Parameter(
            torch.randn(num_layers, base_kernel_size)
        )
        self.base_kernels_channel = nn.Parameter(
            torch.randn(num_layers, base_kernel_size)
        )
        self.layer_transforms_spatial = nn.ModuleList([])
        for s_kern in spatial_kernel_size:
            self.layer_transforms_spatial.append(
                nn.Linear(base_kernel_size, s_kern * s_kern * latent_size, bias=True)
            )
        self.layer_transforms_channel = nn.ModuleList([])
        for c_kern in channel_kernel_size:
            self.layer_transforms_channel.append(
                nn.Linear(base_kernel_size, c_kern * latent_size, bias=True)
            )
        self.spatial_pooling = spatial_pooling
        self.channel_pooling = channel_pooling
        self.num_layers = num_layers
        self.latent_size = latent_size
        assert self.num_layers % 2 == 0

    def reshape_input(self, x):
        shape = x.shape
        if len(shape) < 3 and shape[1] == self.output_size:  # We are in the final layer
            x = x.unsqueeze(2).unsqueeze(2)
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
        elif len(shape) > 4:
            b, _, _, w, h = shape
            x = x.reshape(b, -1, w, h)
        return x

    def forward(self, x, g=None, l=0, n=1):
        if g is None:
            return 0

        shape = x.shape
        x = self.reshape_input(x)
        g = g.view([g.shape[0]] + [1 for _ in range(len(x.shape) - 1)])
        g = g.expand([-1, 1] + list(x.shape[2:])).float().to(x.device)
        x = torch.cat([x, g], dim=1)  # g as channel dimension
        x = x.unsqueeze(1)  # additional dummy channel dimension (to allow 3D conv)

        indices = [None] * (self.num_layers // 2)
        prev_output = []
        pool_input_shape = []
        for i in range(self.num_layers):
            # create kernel for layer i of U-Net (to apply to layer l output of teacher)
            c_kern = self.base_kernels_channel[i]
            s_kern = self.base_kernels_spatial[i]
            s_kern = self.layer_transforms_spatial[l](s_kern).reshape(
                self.latent_size, -1
            )
            c_kern = self.layer_transforms_channel[l](c_kern).reshape(
                self.latent_size, -1
            )
            s = int(math.sqrt(s_kern.shape[1]))
            kernel = (c_kern.T @ s_kern).reshape(1, 1, -1, s, s)  # [1,1,c,s,s]

            # apply kernel in 3D convolution
            if i < self.num_layers / 2:
                x = F.conv3d(x, kernel)
                x = F.relu(x)
                if i < self.num_layers / 2 - 1:
                    pool_input_shape.append(x.shape)
                    prev_output.append(x)
                    x, indices[i] = F.max_pool3d(
                        x,
                        [
                            self.channel_pooling[l][i],
                            self.spatial_pooling[l][i],
                            self.spatial_pooling[l][i],
                        ],
                        return_indices=True,
                    )
            else:
                mirrored_i = (
                    self.num_layers - 2
                ) - i  # -2 because last layer has no pooling
                x = F.conv_transpose3d(x, kernel)
                x = F.relu(x)
                if i < self.num_layers - 1:
                    x = F.max_unpool3d(
                        x,
                        indices[mirrored_i],
                        [
                            self.channel_pooling[l][mirrored_i],
                            self.spatial_pooling[l][mirrored_i],
                            self.spatial_pooling[l][mirrored_i],
                        ],
                        output_size=pool_input_shape[mirrored_i],
                    )
                    x += prev_output[mirrored_i]
        x = x[:, :, :-1]
        return x.reshape(shape)


class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_lens=()):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        d_embed = d_model // 3
        # for d, len in enumerate(max_lens):
        pe = torch.zeros(1, max(max_lens), d_embed)
        position = torch.arange(0, max(max_lens), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if pe.shape[-1] % 2 == 0:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)[:, :-1]
            # pe[d] = pe[d].unsqueeze(0)
        threed_pe = torch.zeros(*max_lens, d_model)
        for tpos in range(max_lens[0]):
            for xpos in range(max_lens[1]):
                for ypos in range(max_lens[2]):
                    threed_pe[tpos, xpos, ypos, : (3 * d_embed)] = torch.cat(
                        [pe[0, tpos], pe[0, xpos], pe[0, ypos]], dim=-1
                    )

        # threed_pe = threed_pe.flatten(0, 2).T
        threed_pe = threed_pe.permute(3, 0, 1, 2)
        self.register_buffer("threed_pe", threed_pe)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x + self.threed_pe[:, :t, :h, :w].unsqueeze(0)
        return self.dropout(x)


class Patching(nn.Module):
    def __init__(self, patch_size, flatten=True):
        super(Patching, self).__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.output_size = 10
        self.up_projection = nn.Linear(self.output_size, self.patch_size, bias=False)

    def patching_2d_input(self, x):
        if len(x.shape) != 4 and x.flatten(1).shape[1] == self.output_size:
            x = self.up_projection(x)
            if not self.flatten:
                return x.reshape(-1, self.patch_size, 1, 1, 1)
        b, c, h, w = orig_shape = x.shape
        p = self.patch_size
        s_p = 0
        pad = [0, 0, 0, 0, 0, 0]
        while c * (s_p + 1) ** 2 <= p:
            s_p += 1
        if s_p != 0 and h % s_p != 0:
            total_pad = h % s_p
            top_pad = total_pad // 2
            pad[2:4] = [top_pad, total_pad - top_pad]
            h += total_pad
        if s_p != 0 and w % s_p != 0:
            total_pad = w % s_p
            left_pad = total_pad // 2
            pad[0:2] = [left_pad, total_pad - left_pad]
            w += total_pad
        if s_p != 0:
            p = p // s_p ** 2
        else:
            s_p = 1
        if c % p != 0:
            total_pad = c % p
            up_pad = total_pad // 2
            pad[4:6] = [up_pad, total_pad - up_pad]
            c += total_pad
        x = F.pad(x, pad)
        x = x.reshape((b, p, c // p, s_p, h // s_p, s_p, w // s_p))
        x = x.permute(0, 1, 3, 5, 2, 4, 6)  # [b, p, s_p, s_p, c//p, h//s_p, w//s_p]
        x = x.reshape((b, self.patch_size, c // p, h // s_p, w // s_p))
        if self.flatten:
            x = x.reshape(b, self.patch_size, -1)
        return x, (p, s_p, orig_shape)

    def invert_patching_2d_input(self, x, p, s_p, orig_shape):
        b, _, c, h, w = x.shape
        _, c_, h_, w_ = orig_shape
        x = x.reshape(b, p, s_p, s_p, c, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)  # [b, p, c, s_p, h, s_p, w]
        x = x.reshape(b, p * c, s_p * h, s_p * w)
        x = x[:, :c_, :h_, :w_]
        return x

    def invert_patching_flat_input(self, x, orig_shape):
        _, s_ = orig_shape
        x = x.flatten(1)
        x = x[:, :s_]
        return x

    def patching_flat_input(self, x):
        assert x.shape[1] >= self.patch_size
        # find decomposition into channels and spatial dimensions that works
        b, s = orig_shape = x.shape
        total_pad = s % self.patch_size
        r = (s + total_pad) // self.patch_size
        c = 1
        while not is_square(r // c):
            c += 1
            if r / c < 1:
                raise ValueError("Hidden dimension not divisible")
        s = int(math.sqrt(r // c))

        x = F.pad(x, (total_pad // 2, total_pad - total_pad // 2))
        x = x.reshape(b, self.patch_size, c, s, s)
        return x, (None, None, orig_shape)

    def forward(self, x):
        if len(x.shape) == 4:
            return self.patching_2d_input(x)
        elif len(x.shape) == 2:
            return self.patching_flat_input(x)

    def inverse(self, x, p, s_p, orig_shape):
        if len(orig_shape) == 4:
            return self.invert_patching_2d_input(x, p, s_p, orig_shape)
        elif len(orig_shape) == 2:
            return self.invert_patching_flat_input(x, orig_shape)


# class LearnedEquivarianceSTSimple(nn.Module):
#     def __init__(
#         self,
#         patch_size,
#         num_layers,
#         group_size,
#         only_translation,
#         handle_output_layer=True,
#         include_channels=False,
#         prevent_translation=False,
#         use_layer_transforms=False,
#     ):
#         super().__init__()
#         assert not (only_translation and prevent_translation)
#         self.only_translation = only_translation
#         self.prevent_translation = prevent_translation
#         self.group_enc = nn.Parameter(torch.randn(group_size, patch_size))
#         if self.only_translation:
#             self.transform_params = 3 if include_channels else 2
#         elif self.prevent_translation:
#             self.transform_params = 9 if include_channels else 4
#         else:
#             self.transform_params = 12 if include_channels else 6
#         if use_layer_transforms:
#             self.num_layers = num_layers
#             self.layer_transforms = nn.ModuleList(
#                 [
#                     nn.Linear(self.transform_params, self.transform_params, bias=True)
#                     for _ in range(num_layers)
#                 ]
#             )
#             for transform in self.layer_transforms:
#                 transform.weight.data = torch.eye(self.transform_params)
#                 transform.bias.data = torch.zeros(self.transform_params)
#             num_layers = 0
#         else:
#             self.layer_transforms = None
#         self.fc_loc = nn.Sequential(
#             nn.Linear(patch_size, 32),
#             nn.ReLU(True),
#             nn.Linear(32, self.transform_params * (num_layers + 1)),
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data = torch.normal(
#             0, 0.2, self.fc_loc[2].weight.data.shape
#         )
#         if prevent_translation:
#             identity_tensor = (
#                 [1, 0, 0, 0, 1, 0, 0, 0, 1] if include_channels else [1, 0, 0, 1]
#             )
#
#         else:
#             identity_tensor = (
#                 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
#                 if include_channels
#                 else [1, 0, 0, 0, 1, 0]
#             )
#         if only_translation:
#             init = torch.normal(0, 0.1, self.fc_loc[2].bias.data.shape)
#         else:
#             init = torch.normal(0, 0.4, self.fc_loc[2].bias.data.shape)
#             # init = torch.tensor(
#             #     identity_tensor * (num_layers + 1), dtype=torch.float
#             # ) + torch.normal(0, 0.1, self.fc_loc[2].bias.data.shape)
#         self.fc_loc[2].bias.data = torch.clone(init)
#         self.include_channels = include_channels
#         self.handle_output_layer = handle_output_layer
#
#     # Spatial transformer network forward function
#     def forward(self, x, g, l=0, *args, **kwargs):
#         g_enc = self.group_enc[g]
#         theta = self.fc_loc(g_enc)
#         if self.layer_transforms and l > 0:
#             theta = self.layer_transforms[l - 1](theta)
#         else:
#             theta = theta[
#                 :, self.transform_params * l : self.transform_params * (l + 1)
#             ]
#         # print("Theta", theta)
#         if self.only_translation:
#             theta = torch.cat(
#                 [
#                     torch.eye(3 if self.include_channels else 2, device=theta.device)
#                     .unsqueeze(0)
#                     .repeat(theta.shape[0], 1, 1),
#                     theta.unsqueeze(2),
#                 ],
#                 dim=2,
#             )
#         elif self.prevent_translation:
#             shape = [theta.shape[0]] + [3, 3] if self.include_channels else [2, 2]
#             theta = torch.cat(
#                 [
#                     theta.view(*shape),
#                     torch.zeros(shape[0], shape[1], 1, device=theta.device),
#                 ],
#                 dim=2,
#             )
#
#         elif self.include_channels:
#             theta = theta.view(-1, 3, 4)
#         else:
#             theta = theta.view(-1, 2, 3)
#
#         ###########
#         # print(theta)
#         # print(F.cosine_similarity(theta[0:10].flatten(1),theta[10:20].flatten(1)))
#         # kernels = self.group_enc
#         # print(kernels)
#         # kernels = F.normalize(kernels, dim=1)
#         # similarity_matrix = torch.matmul(kernels, kernels.T)
#         # similarity_matrix = torch.triu(
#         #     similarity_matrix, diagonal=1
#         # )  # get only entries above diag
#         # G = similarity_matrix.shape[0]
#         # idx = torch.arange(G)
#         # idx = (idx * -1) % G
#         # similarity_matrix[torch.arange(G), idx] = 0
#         # normalization = ((G * (G - 1)) / 2) - G # upper triangle
#         # normalization -= (G-1) // 2  # removing -g
#         # reg = torch.sum(torch.abs(similarity_matrix)) / normalization
#         # print(reg)
#         ########
#
#         if len(x.shape) == 2 and l == self.num_layers and self.handle_output_layer:
#             x = x.unsqueeze(1).unsqueeze(3)
#         elif len(x.shape) == 2:
#             x = reshape_flat_to_chw(x)
#         squeeze_after = False
#         if self.include_channels and len(x.shape) < 5:
#             x = x.unsqueeze(1)
#             squeeze_after = True
#         grid = F.affine_grid(theta, x.size())
#         # print("Grid", grid)
#         x = F.grid_sample(x, grid)
#         if squeeze_after:
#             x = rearrange(x, "b o c h w -> b (o c) h w")  # squeeze this dim
#         return x, theta


class LearnedEquivarianceSTSimple(nn.Module):
    def __init__(
        self,
        num_layers,
        group_size,
        only_translation,
        handle_output_layer=True,
        include_channels=False,
        prevent_translation=False,
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
                    transform[:2, :2] = m[:, :2]
                    transform[:2, 3:] = m[:, 2:]
                    # transform[2, 3] = torch.empty(1).uniform_(-0.1, 0.1).item()
                    # x_angle = float(torch.empty(1).uniform_(0.0, 360.0).item())
                    # y_angle = float(torch.empty(1).uniform_(0.0, 360.0).item())
                    # x_rot = math.radians(x_angle)
                    # y_rot = math.radians(y_angle)
                    # transform = transform @ torch.tensor(
                    #     [
                    #         [1, 0, 0, 0],
                    #         [0, math.cos(x_rot), -math.sin(x_rot), 0],
                    #         [0, math.sin(x_rot), math.cos(x_rot), 0],
                    #         [0,0,0,1]
                    #     ]
                    # )
                    # transform = transform @ torch.tensor(
                    #     [
                    #         [math.cos(y_rot), 0, math.sin(y_rot), 0],
                    #         [0, 1, 0, 0],
                    #         [-math.sin(y_rot), 0, math.cos(y_rot), 0],
                    #         [0, 0, 0, 1]
                    #     ]
                    # )
                    transform = transform[:3]
                else:
                    transform = m
                init[l][g] = transform

        init = init.flatten(2)
        self.theta = nn.Parameter(init)
        # self.group_enc = nn.Parameter(torch.randn(group_size,self.transform_params))
        self.include_channels = include_channels
        self.handle_output_layer = handle_output_layer
        self.num_layers = num_layers

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
        if self.include_channels:
            padding = get_padding((x.shape[2], x.shape[3], x.shape[4]))
        else:
            padding = get_padding((x.shape[2], x.shape[3]))
        x_padded = F.pad(x, padding, mode="circular")
        grid = F.affine_grid(theta, x_padded.size())
        x_padded = F.grid_sample(x_padded, grid)
        if self.include_channels:
            x = x_padded[
                :,
                :,
                padding[4] : padding[4] + x.shape[2],
                padding[2] : padding[2] + x.shape[3],
                padding[0] : padding[0] + x.shape[4],
            ]
        else:
            x = x_padded[
                :,
                :,
                padding[2] : padding[2] + x.shape[2],
                padding[0] : padding[0] + x.shape[3],
            ]
        if squeeze_after:
            x = rearrange(x, "b o c h w -> b (o c) h w")  # squeeze this dim
        return x, theta


class LearnedEquivarianceST(nn.Module):
    def __init__(self, patch_size, num_layers, group_size):
        super().__init__()
        self.patching = Patching(patch_size, flatten=False)
        self.pos_enc = PositionalEncoding3D(patch_size, max_lens=(64, 28, 28))
        self.layer_enc = nn.Parameter(torch.randn(num_layers, patch_size))
        self.group_enc = nn.Parameter(torch.randn(group_size, patch_size))
        # patch_size += 2
        self.layer1 = TransformerEncoderLayer(
            patch_size, nhead=1, dim_feedforward=patch_size
        )
        self.layer2 = TransformerEncoderLayer(
            patch_size, nhead=1, dim_feedforward=patch_size
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(patch_size, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def localization(self, x, g, l):
        x, _ = self.patching(x)
        x = self.pos_enc(x)

        # add g and l as additional channel for each patch
        # g = g.view([g.shape[0]] + [1 for _ in range(len(x.shape) - 1)])
        # g = g.expand([-1, 1] + list(x.shape[2:])).float().to(x.device)
        # l = l * torch.ones([x.shape[0], 1] + list(x.shape[2:])).float().to(x.device)
        # x = torch.cat([x, g, l], dim=1)  # g and l as channel dimension

        x = x.flatten(2)
        x = x.permute(2, 0, 1)

        g_l_enc = self.layer_enc[l] + self.group_enc[g]
        x = x + g_l_enc
        x = self.layer1(x)
        x = x + g_l_enc
        x = self.layer2(x)
        x = x + g_l_enc

        x = x.permute(1, 2, 0)

        x, _ = torch.max(x, dim=2)
        theta = self.fc_loc(x)
        return theta

    # Spatial transformer network forward function
    def forward(self, x, g, l=0, *args, **kwargs):
        theta = self.localization(x, g, l)
        # print("Theta", theta)
        theta = theta.view(-1, 2, 3)
        if len(x.shape) == 2:
            x = reshape_flat_to_chw(x)
        grid = F.affine_grid(theta, x.size())
        # print("Grid", grid)
        x = F.grid_sample(x, grid, padding_mode="reflection")

        return x


class LearnedEquivariancePatchNet(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patching = Patching(patch_size, flatten=False)
        self.pos_enc = PositionalEncoding3D(patch_size, max_lens=(64, 28, 28))
        patch_size += 2
        self.layer1 = TransformerEncoderLayer(
            patch_size, nhead=1, dim_feedforward=patch_size
        )
        self.layer2 = TransformerEncoderLayer(
            patch_size, nhead=1, dim_feedforward=patch_size
        )

    def forward(self, x, g, l=0, *args, **kwargs):
        x, (p, s_p, orig_shape) = self.patching(x)
        x = self.pos_enc(x)

        # add g and l as additional channel for each patch
        g = g.view([g.shape[0]] + [1 for _ in range(len(x.shape) - 1)])
        g = g.expand([-1, 1] + list(x.shape[2:])).float().to(x.device)
        l = l * torch.ones([x.shape[0], 1] + list(x.shape[2:])).float().to(x.device)
        x = torch.cat([x, g, l], dim=1)  # g and l as channel dimension

        unflattened_shape = x.shape
        x = x.flatten(2)
        x = x.permute(2, 0, 1)

        x = self.layer1(x)
        x = self.layer2(x)

        x = x.permute(1, 2, 0)
        x = x.reshape(unflattened_shape)
        x = x[:, :-2]

        x = self.patching.inverse(x, p, s_p, orig_shape)

        return x


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


class LearnedEquivarianceFactorized(LearnedEquivariance):
    def __init__(
        self,
        kernel_size=5,
        depth_kernel_size=5,
        group_size=40,
        num_layers=0,
        output_size=10,
    ):
        super().__init__(
            kernel_size=kernel_size,
            group_size=group_size,
            num_layers=num_layers,
            output_size=output_size,
        )
        self.depth_kernels = torch.nn.Parameter(
            torch.randn((group_size, depth_kernel_size))
        )
        if num_layers:
            self.layer_transforms_depth = nn.ModuleList(
                [
                    nn.Linear(depth_kernel_size, depth_kernel_size, bias=True)
                    for _ in range(num_layers - 1)
                ]
            )
        else:
            self.layer_transforms_depth = None
        self.full_depth_padding = get_padding((depth_kernel_size,))
        self.output_size = output_size

    def forward(self, x, g=None, l=0, n=1):
        if g is not None:
            shape = x.shape
            x = self.reshape_input(x)
            if shape[1] >= self.depth_kernels.shape[1] and l < len(
                self.layer_transforms_depth
            ):
                g %= self.depth_kernels.shape[0]
                x = x.flatten(2).permute(
                    2, 0, 1
                )  # switch spatial with batch dimension to apply different kernel to each sample (based on g)
                # -> [w*h, batch_size, c]
                kernel = self.depth_kernels[g]
                padding = self.full_depth_padding
                if self.layer_transforms_depth is not None and l > 0:
                    kernel = self.layer_transforms_depth[l](kernel)
                kernel = kernel.unsqueeze(
                    1
                )  # [batch_size, 1, k] -> [out_channels, in_channels/groups, k]
                for i in range(n):
                    x_padded = F.pad(
                        x, padding, mode="circular"
                    )  # Troublesome if spatial dimension smaller than padding! (for cirular padding)
                    x = F.conv1d(
                        x_padded,
                        kernel,
                        groups=kernel.shape[0],
                    )
                x = x.permute(1, 2, 0).reshape(shape)

            x = super(LearnedEquivarianceFactorized, self).forward(
                x, g, l, n
            )  # spatial transform
            return x

        return 0


def equiv_builder(seed: int, config):
    if config.fully_connected:
        model = LearnedEquivarianceFC(
            group_size=config.group_size,
            layer_sizes=config.layer_sizes,
        )
    elif config.factorized:
        model = LearnedEquivarianceFactorized(
            kernel_size=config.kernel_size,
            group_size=config.group_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
        )
    elif config.unet:
        model = LearnedEquivarianceUNet(
            base_kernel_size=config.kernel_size,
            spatial_kernel_size=config.spatial_kernel_size,
            channel_kernel_size=config.channel_kernel_size,
            spatial_pooling=config.spatial_pooling,
            channel_pooling=config.channel_pooling,
            latent_size=config.latent_size,
            num_layers=config.num_layers,
        )
    elif config.spatial_transformer:
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
        )
    elif config.patch_net:
        model = LearnedEquivariancePatchNet(patch_size=config.patch_size)
    else:
        model = LearnedEquivariance(
            kernel_size=config.kernel_size,
            group_size=config.group_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
            vit_input=config.vit_input,
        )
    return model

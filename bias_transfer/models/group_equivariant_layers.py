import torch

"""Implementation of modules from the paper
'Group Equivariant Convolutional Networks'
by T. Cohen and M. Welling (2016).
Regular Z2 images are tensors with shape NxCxHxW.
Rotation P4 images are tensors with shape NxCx4xHxW.

Implementation from: https://github.com/claudio-unipv/groupcnn/blob/main/groupcnn.py
"""


class ConvZ2P4(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, stride=1, padding=1):
        super().__init__()
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(w)
        torch.nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

    def _rotated(self, w):
        ws = [torch.rot90(w, k, (2, 3)) for k in range(4)]
        return torch.cat(ws, 1).view(-1, w.size(1), w.size(2), w.size(3))

    def forward(self, x):
        w = self._rotated(self.weight)
        y = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=self.padding)
        y = y.view(y.size(0), -1, 4, y.size(2), y.size(3))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


class ConvP4(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, stride=1, padding=1):
        super().__init__()
        w = torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(w)
        torch.nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

    def _rotated(self, w):
        ws = [_grot90(w, k).view(w.size(0), -1, w.size(3), w.size(4)) for k in range(4)]
        return torch.cat(ws, 1).view(4 * w.size(0), 4 * w.size(1), w.size(3), w.size(4))

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        w = self._rotated(self.weight)
        y = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=self.padding)
        y = y.view(y.size(0), -1, 4, y.size(2), y.size(3))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


class MaxRotationPoolP4(torch.nn.Module):
    def forward(self, x):
        return x.max(2).values


class MaxSpatialPoolP4(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.inner = torch.nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        y = x.view(x.size(0), -1, x.size(3), x.size(4))
        y = self.inner(y)
        y = y.view(x.size(0), -1, 4, y.size(2), y.size(3))
        return y


class AvgRootPoolP4(torch.nn.Module):
    def forward(self, x):
        return x.mean(2)


def _grot90(x, k):
    return torch.rot90(x.roll(k, 2), k, (3, 4))


def _test():
    import matplotlib.pyplot as plt
    conv = ConvZ2P4(3, 10, 3)
    conv.weight[0, :, :, :].data.fill_(0)
    conv.weight[0, :, 0, :].data.fill_(1)
    conv.weight[0, :, -1, :].data.fill_(-1)
    x = torch.zeros(1, 3, 256, 256)
    x.data[0, 0, 32:224, 32:64] = 1
    x.data[0, 1, 32:64, 32:224] = 1
    print(sum(p.numel() for p in conv.parameters()))
    y = conv(x)
    print(x.size(), "->", y.size())
    k = 1
    y2 = _grot90(conv(torch.rot90(x, k, (2, 3))), -k)
    plt.imshow(y[0, 0, :, :, :].reshape(-1, y.size(-1)).detach())
    plt.figure()
    plt.imshow(y2[0, 0, :, :, :].reshape(-1, y2.size(-1)).detach())
    plt.show()


def _test():
    conv = ConvZ2P4(3, 10, 3)
    x = torch.rand(7, 3, 256, 256)
    y = conv(x)
    for k in range(1, 4):
        y2 = _grot90(conv(torch.rot90(x, k, (2, 3))), -k)
        print(k, (y - y2).abs().max().item())
    print()

    conv2 = ConvP4(3, 10, 3, padding=0, bias=False)
    x = torch.rand(7, 3, 4, 256, 256)
    y = conv2(x)
    for k in range(1, 4):
        y2 = _grot90(conv2(_grot90(x, k)), -k)
        print(k, (y - y2).abs().max().item())


if __name__ == "__main__":
    _test()
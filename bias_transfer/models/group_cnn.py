from .group_equivariant_layers import *
import torch


class GCNN(torch.nn.Module):
    def __init__(self, output_size=10):
        super(GCNN, self).__init__()
        self.conv1 = ConvZ2P4(1, 8, 5)
        self.pool1 = MaxSpatialPoolP4(2)
        self.conv2 = ConvP4(8, 32, 3)
        self.pool2 = MaxSpatialPoolP4(2)
        self.conv3 = ConvP4(32, 64, 3)
        self.pool3 = MaxSpatialPoolP4(2)
        self.conv4 = ConvP4(64, output_size, 3)
        self.pool4 = MaxRotationPoolP4()
        self.pool5 = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(self.pool1(x))
        x = self.conv2(x)
        x = torch.nn.functional.relu(self.pool2(x))
        x = self.conv3(x)
        x = torch.nn.functional.relu(self.pool3(x))
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.pool5(x)
        x = x.squeeze(-1).squeeze(-1)
        return x


def gcnn_builder(seed: int, config):
    model = GCNN(
        output_size=config.output_size,
    )
    return model

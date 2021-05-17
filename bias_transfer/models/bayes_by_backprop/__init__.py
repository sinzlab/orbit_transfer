import numpy as np
import torch

from bias_transfer.models.bayes_by_backprop.lenet_300_100 import LeNet300100
from bias_transfer.models.bayes_by_backprop.lenet_5 import LeNet5


def lenet_builder(seed: int, config):
    if "5" in config.type:
        lenet = LeNet5
    elif "300-100" in config.type:
        lenet = LeNet300100

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    model = lenet(
        num_classes=config.num_classes,
        input_height=config.input_height,
        input_width=config.input_width,
        input_channels=config.input_channels,
        dropout=config.dropout,
    )
    return model


def linear_bayes_builder(seed: int, config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    model = BayesLinear(config.input_size, config.num_classes, bias=False)
    return model
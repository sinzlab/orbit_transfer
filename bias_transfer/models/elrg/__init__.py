import numpy as np
import torch

from bias_transfer.models.elrg.lenet_300_100 import LeNet300100
from bias_transfer.models.elrg.lenet_5 import LeNet5
from bias_transfer.models.elrg.linear import ELRGLinear


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
        rank=config.rank,
        alpha=config.gamma,
        train_var=config.train_var,
        initial_var=config.initial_var,
    )
    return model


def linear_elrg_builder(seed: int, config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    model = ELRGLinear(
        config.input_size,
        config.num_classes,
        bias=False,
        rank=config.rank,
        alpha=config.alpha,
        train_var=config.train_var,
        initial_posterior_var=config.initial_var
    )
    return model
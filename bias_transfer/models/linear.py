import torch
import numpy as np
from torch import nn

from bias_transfer.models.layers.bayes_linear import BayesLinear
from bias_transfer.models.layers.elrg_linear import ELRGLinear
from nntransfer.models.utils import concatenate_flattened

def linear_builder(seed: int, config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    model = torch.nn.Linear(config.input_size, config.num_classes, bias=False)
    return model


def linear_bayes_builder(seed: int, config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    model = BayesLinear(config.input_size, config.num_classes, bias=False)
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

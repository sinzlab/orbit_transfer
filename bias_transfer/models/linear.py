import torch
import numpy as np


def linear_builder(seed: int, config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    model = torch.nn.Linear(config.input_size, config.num_classes, bias=False)
    return model



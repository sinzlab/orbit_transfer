import torch
import numpy as np
from torch.hub import load_state_dict_from_url

from nnfabrik.utility.nn_helpers import load_state_dict


from nntransfer.models.resnet import resnet_builder
from nntransfer.models.utils import get_model_parameters
from nntransfer.models.vgg import vgg_builder
from nntransfer.models.lenet import lenet_builder

# from nntransfer.models.mlp import MLP
from nntransfer.models.wrappers import *

from bias_transfer.configs.model import (
    ClassificationModel,
)
from bias_transfer.models.bayes_by_backprop import lenet_builder as bayes_builder, linear_bayes_builder
from .lenet_frcl import lenet_builder as frcl_builder
from bias_transfer.models.elrg import lenet_builder as elrg_builder
from bias_transfer.models.elrg import linear_elrg_builder
from .linear import linear_builder
from .mlp import mlp_builder


def classification_model_builder(data_loader, seed: int, **config):
    config = ClassificationModel.from_dict(config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if "vgg" in config.type:
        model = vgg_builder(seed, config)
        from torchvision.models.vgg import model_urls
    elif "resnet" in config.type:
        model = resnet_builder(seed, config)
        from torchvision.models.resnet import model_urls
    elif "lenet" in config.type:
        if "bayes" in config.type:
            model = bayes_builder(seed, config)
        elif "elrg" in config.type:
            model = elrg_builder(seed, config)
        elif "frcl" in config.type:
            model = frcl_builder(seed, config)
        else:
            model = lenet_builder(seed, config)
    elif "linear" in config.type:
        if "bayes" in config.type:
            model = linear_bayes_builder(seed, config)
        elif "elrg" in config.type:
            model = linear_elrg_builder(seed, config)
        else:
            model = linear_builder(seed, config)
    elif "mlp":
        model = mlp_builder(seed, config)
    else:
        raise Exception("Unknown type {}".format(config.type))

    if config.pretrained:
        print("Downloading pretrained model:", flush=True)
        url = (
            model_urls[config.type]
            if not config.pretrained_url
            else config.pretrained_url
        )
        state_dict = load_state_dict_from_url(url, progress=True)
        try:
            load_state_dict(model, state_dict)
        except:
            load_state_dict(model, state_dict["model_state_dict"])

    print("Model with {} parameters.".format(get_model_parameters(model)))
    if config.add_buffer:
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                for b in config.add_buffer:
                    if isinstance(b, str):
                        model.register_buffer(
                            f"{n}_{b}",
                            p.detach().clone().zero_(),
                        )
                    else:
                        k = b[1]
                        b = b[0]
                        model.register_buffer(
                            f"{n}_{b}",
                            torch.zeros(k, *p.data.shape),
                        )
    if config.add_custom_buffer:
        for key, size in config.add_custom_buffer.items():
            model.register_buffer(
                key,
                torch.zeros(size),
            )
    # Add wrappers
    if config.get_intermediate_rep:
        model = IntermediateLayerGetter(
            model, return_layers=config.get_intermediate_rep, keep_output=True
        )
    if config.noise_adv_regression or config.noise_adv_classification:
        assert not config.self_attention
        model = NoiseAdvWrapper(
            model,
            input_size=model.fc.in_features
            if "resnet" in config.type
            else model.n_features,
            hidden_size=model.fc.in_features if "resnet" in config.type else 4096,
            classification=config.noise_adv_classification,
            num_noise_readout_layers=config.num_noise_readout_layers,
            sigmoid_output=config.noise_sigmoid_output,
        )
    return model

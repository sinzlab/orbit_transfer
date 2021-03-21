import torch
import numpy as np
from torch.hub import load_state_dict_from_url

from nnfabrik.utility.nn_helpers import load_state_dict
from nnvision.models.models import se_core_gauss_readout, se_core_point_readout


from nntransfer.models.resnet import resnet_builder
from nntransfer.models.utils import get_model_parameters
from nntransfer.models.vgg import vgg_builder
from nntransfer.models.lenet import lenet_builder
# from nntransfer.models.mlp import MLP
from nntransfer.models.wrappers import *

from bias_transfer.configs.model import (
    ClassificationModel,
)
from .lenet_bayesian import lenet_builder as bayes_builder
from .lenet_frcl import lenet_builder as frcl_builder


def neural_cnn_builder(data_loaders, seed: int = 1000, **config):
    config.pop("comment", None)
    readout_type = config.pop("readout_type", None)
    if readout_type == "point":
        model = se_core_point_readout(dataloaders=data_loaders, seed=seed, **config)
    elif readout_type == "gauss":
        model = se_core_gauss_readout(dataloaders=data_loaders, seed=seed, **config)
    print("Model with {} parameters.".format(get_model_parameters(model)))
    return model

#
# def mtl_builder(data_loaders, seed: int = 1000, **config):
#     config = MTL.from_dict(config)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#
#     from .mtl_vgg import MTL_VGG
#
#     model = MTL_VGG(
#         data_loaders,
#         vgg_type=config.vgg_type,
#         classification=config.classification,
#         classification_readout_type=config.classification_readout_type,
#         input_size=config.input_size,
#         num_classes=config.num_classes,
#         pretrained=config.pretrained,
#         v1_model_layer=config.v1_model_layer,
#         neural_input_channels=config.neural_input_channels,
#         classification_input_channels=config.classification_input_channels,
#         v1_fine_tune=config.v1_fine_tune,
#         v1_init_mu_range=config.v1_init_mu_range,
#         v1_init_sigma_range=config.v1_init_sigma_range,
#         v1_readout_bias=config.v1_readout_bias,
#         v1_bias=config.v1_bias,
#         v1_gamma_readout=config.v1_gamma_readout,
#         v1_elu_offset=config.v1_elu_offset,
#         v1_final_batchnorm=config.v1_final_batchnorm,
#     )
#
#     print("Model with {} parameters.".format(get_model_parameters(model)))
#     return model


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
        elif "frcl" in config.type:
            model = frcl_builder(seed, config)
        else:
            model = lenet_builder(seed, config)
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
    print("Model with {} parameters.".format(get_model_parameters(model)))
    if config.add_buffer:
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                for b in config.add_buffer:
                    model.register_buffer(
                        f"{n}_{b}", p.detach().clone().zero_(),
                    )
    return model

#
# def regression_model_builder(data_loader, seed: int, **config):
#     config = RegressionModel.from_dict(config)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#
#     model = MLP(
#         input_size=config.input_size,
#         num_layers=config.num_layers,
#         layer_size=config.layer_size,
#         output_size=config.output_size,
#         activation=config.activation,
#         dropout=config.dropout,
#     )
#
#     # Add wrappers
#     if config.get_intermediate_rep:
#         model = IntermediateLayerGetter(
#             model, return_layers=config.get_intermediate_rep, keep_output=True
#         )
#
#     print("Model with {} parameters.".format(get_model_parameters(model)))
#     return model

import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, BasicBlock
from bias_transfer.models.resnet import ResNet


def reset_params(model, reset=None):
    model = model.module if isinstance(model, nn.DataParallel) else model
    if reset == "all":
        print(f"Resetting all parameters")
        model.apply(weight_reset)
    elif reset:
        print(f"Resetting {reset}")
        for layer in reset:
            if isinstance(layer, str):
                getattr(model, layer).apply(weight_reset)
            else:
                model[layer].apply(weight_reset)


def freeze_params(model, freeze=None, readout_name=""):
    if not freeze:
        return

    to_freeze = None
    not_to_freeze = None
    if "core" in freeze:
        not_to_freeze = (readout_name,)
    elif freeze == ("readout",):
        to_freeze = (readout_name,)
    else:
        to_freeze = freeze
    for name, param in model.named_parameters():
        if to_freeze:
            freeze = False
            for freeze_key in to_freeze:
                if freeze_key in name:
                    freeze = True
        elif not_to_freeze:
            freeze = True
            for un_freeze_key in not_to_freeze:
                if un_freeze_key in name:
                    freeze = False
        else:
            raise Exception(
                "Please provide either to_freeze or not_to_freeze arguments!"
            )
        if freeze and param.requires_grad:
            param.requires_grad = False


def weight_reset(m, advanced_init=False, zero_init_residual=False):
    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Conv2d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv3d)
        or isinstance(m, nn.ConvTranspose1d)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.ConvTranspose3d)
        or isinstance(m, nn.BatchNorm1d)
        or isinstance(m, nn.BatchNorm2d)
        or isinstance(m, nn.BatchNorm3d)
        or isinstance(m, nn.GroupNorm)
    ):
        m.reset_parameters()
        if advanced_init and isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif advanced_init and isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    if zero_init_residual and isinstance(m, Bottleneck):
        nn.init.constant_(m.bn3.weight, 0)
    elif zero_init_residual and isinstance(m, BasicBlock):
        nn.init.constant_(m.bn2.weight, 0)


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def set_bn_to_eval(m, train_mode=False):
    classname = m.__class__.__name__
    if "BatchNorm" in classname:
        m.train(train_mode)


def concatenate_flattened(tensor_list) -> torch.Tensor:
       """
       Given list of tensors, flattens each and concatenates their values.
       """
       return torch.cat([torch.reshape(t, (-1,)) for t in tensor_list])
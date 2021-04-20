import os
from pathlib import Path

import torch
import numpy as np

from torchvision.datasets import MNIST, FashionMNIST, EMNIST, KMNIST, QMNIST
from torchvision import transforms

from nnfabrik.utility.nn_helpers import set_random_seed
from .addition import apply_additon
from .expansion import apply_expansion
from .noise import apply_gaussian_noise
from .color import apply_color, get_color_codes
from .scale import apply_scale
from .subsample import Subsample
from .translation import apply_translation
from .rotation import apply_rotation
from .shuffle import apply_label_shuffle


def generate_dataset(data_loader, transform_fs=(), options=()):
    new_ds_source = []
    new_ds_target = []
    for source, target in data_loader:
        source = source.detach().numpy()
        target = target.detach().numpy()
        for t, transform_f in enumerate(transform_fs):
            if transform_f is None:
                continue
            source, target = transform_f(source, target, **options[t])
        if source is not None and target is not None:
            new_ds_source.append(source)
            new_ds_target.append(target)
    new_ds_source = np.concatenate(new_ds_source)
    new_ds_target = np.concatenate(new_ds_target)
    return new_ds_source, new_ds_target


bias_dict = {
    "color": (
        (apply_color,),
        (
            {
                "cfg_means": get_color_codes(),
                "cbg_means": get_color_codes(),
                "bg": False,
                "fg": True,
                "color_variance": 0.02,
            },
        ),
    ),
    "color_easy": (
        (apply_color,),
        (
            {
                "cfg_means": get_color_codes(),
                "cbg_means": get_color_codes(),
                "bg": False,
                "fg": True,
                "color_variance": 0.00,
            },
        ),
    ),
    "color_shuffle": (
        (apply_color,),
        (
            {
                "cfg_means": get_color_codes(),
                "cbg_means": get_color_codes(),
                "bg": False,
                "fg": True,
                "color_variance": 0.02,
                "shuffle": True,
            },
        ),
    ),
    "noise": ((apply_gaussian_noise,), ({"severity": -1},)),  # random
    "translation": ((apply_translation,), ({"std": 5},)),
    "rotation": ((apply_rotation,), ({},)),
    "rotation_regression": ((apply_rotation,), ({"regression": True},)),
    "addition_regression": ((apply_additon,), ({},)),
    "addition_regression_noise": ((apply_additon, apply_gaussian_noise), ({}, {"severity": -1})),
    "clean": ((None,), ({},)),
    "low_resource": ((Subsample(total_per_class=100),), ({},)),
    "clean_shuffle": ((apply_label_shuffle,), ({},)),
    "scale": ((apply_scale,), ({},)),
}


def generate_and_save(
    bias: str,
    base_path: str = "/work/data/image_classification/torchvision/",
    bias_options_: dict = None,
    dataset: str = "MNIST",
):
    set_random_seed(42)
    write_path = os.path.join(base_path, f"{dataset}-Transfer")
    Path(write_path).mkdir(parents=True, exist_ok=True)
    if (
        os.path.isfile(os.path.join(write_path, f"{bias}_train_source.npy"))
        and os.path.isfile(os.path.join(write_path, f"{bias}_train_target.npy"))
        and os.path.isfile(os.path.join(write_path, f"{bias}_test_source.npy"))
        and os.path.isfile(os.path.join(write_path, f"{bias}_test_target.npy"))
    ):
        return
    apply_bias, bias_options = bias_dict[bias]
    bias_options = bias_options_ if bias_options_ is not None else bias_options
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train = globals().get(dataset)(
        root=base_path,
        train=True,
        download=True,
        transform=transform,
    )
    test = globals().get(dataset)(
        root=base_path,
        train=False,
        download=True,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=64,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=False,
    )
    train_ds = generate_dataset(
        data_loader=train_loader,
        transform_fs=(apply_expansion, *apply_bias),
        options=({}, *bias_options),
    )
    Subsample.active = False  # in case we are using this, we need to deactivate for test set
    test_ds = generate_dataset(
        data_loader=test_loader,
        transform_fs=(apply_expansion, *apply_bias),
        options=({}, *bias_options),
    )
    np.save(os.path.join(write_path, f"{bias}_train_source.npy"), train_ds[0])
    np.save(os.path.join(write_path, f"{bias}_train_target.npy"), train_ds[1])
    np.save(os.path.join(write_path, f"{bias}_test_source.npy"), test_ds[0])
    np.save(os.path.join(write_path, f"{bias}_test_target.npy"), test_ds[1])

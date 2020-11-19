import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import ChainDataset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from bias_transfer.configs.dataset import ImageDatasetConfig
from .MNIST_IB import generate_and_save
from .dataset_classes.pkl_dataset import PklDataset
from .dataset_classes.npy_dataset import NpyDataset
from .utils import (
    get_dataset,
    create_ImageFolder_format,
)

DATASET_URLS = {
    "TinyImageNet": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "CIFAR10-Semisupervised": "1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi",
    "CIFAR10-C": "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar",
    "CIFAR100-C": "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar",
    "TinyImageNet-C": "https://zenodo.org/record/2536630/files/Tiny-ImageNet-C.tar",
    "TinyImageNet-ST": "https://informatikunihamburgde-my.sharepoint.com/:u:/g/personal/shahd_safarani_informatik_uni-hamburg_de/EZhUKKVXTvRHlqi2HXHaIjEBLmAv4tQP8olvdGNRoWrPqA?e=8kSrHI&download=1",
    "ImageNet": None,
    "ImageNet-C": {
        "blur": "https://zenodo.org/record/2235448/files/blur.tar",
        "digital": "https://zenodo.org/record/2235448/files/digital.tar",
        "extra": "https://zenodo.org/record/2235448/files/extra.tar",
        "noise": "https://zenodo.org/record/2235448/files/noise.tar",
        "weather": "https://zenodo.org/record/2235448/files/weather.tar",
    },
}


def img_dataset_loader(seed, **config):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    config = ImageDatasetConfig.from_dict(config)
    print("Loading dataset: {}".format(config.dataset_cls))
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform_test, transform_train, transform_val = get_transforms(config)

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (config.valid_size >= 0) and (config.valid_size <= 1), error_msg

    (
        train_dataset,
        valid_dataset,
        test_dataset,
        c_test_datasets,
        st_test_dataset,
    ) = get_datasets(config, transform_test, transform_train, transform_val)

    filters = [globals().get(f)(config, train_dataset) for f in config.filters]
    datasets_ = [train_dataset, valid_dataset, test_dataset]
    if config.add_corrupted_test:
        for c_ds in c_test_datasets.values():
            datasets_ += list(c_ds.values())
    for ds in datasets_:
        for filt in filters:
            filt.apply(ds)

    data_loaders = get_data_loaders(
        st_test_dataset,
        c_test_datasets,
        config,
        seed,
        test_dataset,
        train_dataset,
        valid_dataset,
    )

    return data_loaders


def get_transforms(config):
    if config.dataset_cls == "ImageNet":
        transform_train = [
            transforms.RandomResizedCrop(config.input_size)
            if config.apply_augmentation
            else None,
            transforms.RandomHorizontalFlip() if config.apply_augmentation else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_val = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_test = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
    else:
        transform_train = [
            transforms.ToPILImage()
            if config.dataset_cls == "CIFAR10-Semisupervised"
            or config.dataset_cls == "MNIST-IB"
            else None,
            transforms.RandomCrop(config.input_size, padding=4)
            if config.apply_augmentation
            else None,
            transforms.RandomHorizontalFlip() if config.apply_augmentation else None,
            transforms.RandomRotation(15)
            if config.apply_augmentation and not "MNIST" in config.dataset_cls
            else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            if config.convert_to_rgb
            else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_val = [
            transforms.ToPILImage()
            if config.dataset_cls == "CIFAR10-Semisupervised"
            or config.dataset_cls == "MNIST-IB"
            else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            if config.convert_to_rgb
            else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_test = [
            transforms.ToPILImage() if config.dataset_cls == "MNIST-IB" else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            if config.convert_to_rgb
            else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
    transform_test = transforms.Compose(
        list(filter(lambda x: x is not None, transform_test))
    )
    transform_val = transforms.Compose(
        list(filter(lambda x: x is not None, transform_val))
    )
    transform_train = transforms.Compose(
        list(filter(lambda x: x is not None, transform_train))
    )
    return transform_test, transform_train, transform_val


def get_datasets(config, transform_test, transform_train, transform_val):
    if (
        config.dataset_cls in list(torchvision.datasets.__dict__.keys())
        and config.dataset_cls != "ImageNet"
    ):
        dataset_cls = eval("torchvision.datasets." + config.dataset_cls)
        kwargs = {
            "root": config.data_dir,
            "download": config.download,
            "transform": transform_train,
        }

        if config.dataset_cls == "SVHN":
            kwargs["split"] = "train"
        else:
            kwargs["train"] = True
        train_dataset = dataset_cls(**kwargs)

        kwargs["transform"] = transform_val
        valid_dataset = dataset_cls(**kwargs)

        kwargs["transform"] = transform_test
        if config.dataset_cls == "SVHN":
            kwargs["split"] = "test"
        else:
            kwargs["train"] = False
        test_dataset = dataset_cls(**kwargs)
    elif config.dataset_cls == "MNIST-IB":
        dataset_dir = os.path.join(config.data_dir, config.dataset_sub_cls + "-IB")
        generate_and_save(
            config.bias, base_path=config.data_dir, dataset=config.dataset_sub_cls
        )
        train_dataset = NpyDataset(
            f"{config.bias}_train_source.npy",
            f"{config.bias}_train_target.npy",
            root=dataset_dir,
            transform=transform_train,
            target_type=torch.float32 if "regression" in config.bias else torch.long
        )
        valid_dataset = NpyDataset(
            f"{config.bias}_train_source.npy",
            f"{config.bias}_train_target.npy",
            root=dataset_dir,
            transform=transform_val,
            target_type=torch.float32 if "regression" in config.bias else torch.long
        )
        test_dataset = NpyDataset(
            f"{config.bias}_test_source.npy",
            f"{config.bias}_test_target.npy",
            root=dataset_dir,
            transform=transform_test,
            target_type=torch.float32 if "regression" in config.bias else torch.long
        )
    else:
        dataset_dir = get_dataset(
            DATASET_URLS[config.dataset_cls],
            config.data_dir,
            dataset_cls=config.dataset_cls,
            download=config.dowload,
        )

        train_dir = os.path.join(dataset_dir, "train")
        if config.dataset_cls == "CIFAR10-Semisupervised":
            train_dataset = PklDataset(
                train_dir, transform=transform_train, root=config.data_dir
            )
            valid_dataset = PklDataset(
                train_dir, transform=transform_val, root=config.data_dir
            )
            dataset_cls = torchvision.datasets.CIFAR10
            test_dataset = dataset_cls(
                root=config.data_dir,
                train=False,
                download=config.download,
                transform=transform_test,
            )
        else:
            if config.dataset_cls != "ImageNet":
                create_ImageFolder_format(dataset_dir)
            val_dir = os.path.join(dataset_dir, "val", "images")
            train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
            valid_dataset = datasets.ImageFolder(train_dir, transform=transform_val)
            test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)

    st_test_dataset = None
    if config.add_stylized_test:
        st_dataset_dir = get_dataset(
            DATASET_URLS[config.dataset_cls + "-ST"],
            config.data_dir,
            dataset_cls=config.dataset_cls + "-ST",
            download=config.dowload,
        )
        st_test_dataset = datasets.ImageFolder(st_dataset_dir, transform=transform_test)

    c_test_datasets = None
    if config.add_corrupted_test:
        urls = DATASET_URLS[config.dataset_cls + "-C"]
        if not isinstance(urls, dict):
            urls = {"default": urls}
        for key, url in urls.items():
            dataset_dir = get_dataset(
                url,
                config.data_dir,
                dataset_cls=config.dataset_cls + "-C",
                download=config.download,
            )

            c_test_datasets = {}
            for c_category in os.listdir(dataset_dir):
                if config.dataset_cls in ("CIFAR10", "CIFAR100"):
                    if c_category == "labels.npy" or not c_category.endswith(".npy"):
                        continue
                    c_test_datasets[c_category[:-4]] = {}
                    for c_level in range(1, 6):
                        start = (c_level - 1) * 10000
                        end = c_level * 10000
                        c_test_datasets[c_category[:-4]][c_level] = NpyDataset(
                            sample_file=c_category,
                            target_file="labels.npy",
                            root=dataset_dir,
                            start=start,
                            end=end,
                            transform=transform_test,
                        )
                else:
                    if not os.path.isdir(os.path.join(dataset_dir, c_category)):
                        continue
                    c_test_datasets[c_category] = {}
                    for c_level in os.listdir(os.path.join(dataset_dir, c_category)):
                        c_test_datasets[c_category][
                            int(c_level)
                        ] = datasets.ImageFolder(
                            os.path.join(dataset_dir, c_category, c_level),
                            transform=transform_test,
                        )
    return train_dataset, valid_dataset, test_dataset, c_test_datasets, st_test_dataset


def get_data_loaders(
    st_test_dataset,
    c_test_datasets,
    config,
    seed,
    test_dataset,
    train_dataset,
    valid_dataset,
):
    num_train = len(train_dataset)
    indices = list(range(num_train))
    if config.use_c_test_as_val:  # Use valid_size of the c_test set for validation
        train_sampler = SubsetRandomSampler(indices)
        datasets = []
        val_indices = []
        start_idx = 0
        for c_category in c_test_datasets.keys():
            if c_category not in (
                "speckle_noise",
                "gaussian_blur",
                "spatter",
                "saturate",
            ):
                continue
            for dataset in c_test_datasets[c_category].values():
                num_val = len(dataset)
                indices = list(range(start_idx, start_idx + num_val))
                split = int(np.floor(config.valid_size * num_val))
                if config.shuffle:
                    np.random.shuffle(indices)
                val_indices += indices[:split]
                datasets.append(dataset)
                start_idx += num_val
        valid_dataset = ConcatDataset(datasets)
        valid_sampler = SubsetRandomSampler(val_indices)
    else:  # Use valid_size of the train set for validation
        split = int(np.floor(config.valid_size * num_train))
        if config.shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        if config.train_subset:
            subset_split = int(np.floor(config.train_subset * len(train_idx)))
            train_idx = train_idx[:subset_split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        sampler=valid_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True,
    )
    task_key = "regression" if "regression" in config.bias else "img_classification"
    data_loaders = {
        "train": {task_key: train_loader},
        "validation": {task_key: valid_loader},
        "test": {task_key: test_loader},
    }

    if config.add_stylized_test:
        st_test_loader = torch.utils.data.DataLoader(
            st_test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            shuffle=False,
        )
        data_loaders["st_test"] = st_test_loader

    if config.add_corrupted_test:
        c_test_loaders = {}
        for c_category in c_test_datasets.keys():
            c_test_loaders[c_category] = {}
            for c_level, dataset in c_test_datasets[c_category].items():
                c_test_loaders[c_category][c_level] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory,
                    shuffle=True,
                )
        data_loaders["c_test"] = {"img_classification": c_test_loaders}
    return data_loaders

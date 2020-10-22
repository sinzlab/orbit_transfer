from typing import Dict

from .base import BaseConfig, baseline
from nnfabrik.main import *


class DatasetConfig(BaseConfig):
    config_name = "dataset"
    table = None
    fn = None

    @baseline
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = kwargs.pop("batch_size", 128)
        self.update(**kwargs)


class ImageDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.img_dataset_loader"

    @baseline
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_cls = kwargs.pop("dataset_cls", "CIFAR10")
        self.apply_augmentation = kwargs.pop("apply_data_augmentation", True)
        self.apply_normalization = kwargs.pop("apply_data_normalization", True)
        self.apply_grayscale = kwargs.pop("apply_grayscale", False)
        self.apply_noise = kwargs.pop("apply_noise", {})
        self.convert_to_rgb = kwargs.pop("convert_to_rgb", False)
        self.input_size = kwargs.pop("input_size", 32)
        self.pin_memory = kwargs.pop("pin_memory", True)
        if (
            "CIFAR" in self.dataset_cls
            or "SVHN" in self.dataset_cls
            or "MNIST" in self.dataset_cls
        ):
            if self.dataset_cls == "CIFAR100":
                self.train_data_mean = (
                    0.5070751592371323,
                    0.48654887331495095,
                    0.4409178433670343,
                )
                self.train_data_std = (
                    0.2673342858792401,
                    0.2564384629170883,
                    0.27615047132568404,
                )
            elif "CIFAR10" in self.dataset_cls:  # also covers semi-supervised version
                self.train_data_mean = (0.49139968, 0.48215841, 0.44653091)
                self.train_data_std = (0.24703223, 0.24348513, 0.26158784)
            elif self.dataset_cls == "SVHN":
                self.train_data_mean = (0.4377, 0.4438, 0.4728)
                self.train_data_std = (0.1980, 0.2010, 0.1970)
            elif self.dataset_cls == "MNIST":
                self.train_data_mean = (0.1307,)
                self.train_data_std = (0.3081,)
                self.input_size = 28
            elif self.dataset_cls == "MNIST-IB":
                self.bias = kwargs.pop("bias", "translation")
                self.dataset_sub_cls = kwargs.pop(
                    "dataset_sub_cls", "MNIST"
                )  # could be e.g. FashionMNIST
                self.input_size = 40 if self.bias != "addition" else 80
                if self.bias == "color":
                    self.train_data_mean = (
                        (0.03685451, 0.0367535, 0.03952756)
                        if self.dataset_sub_cls == "MNIST"
                        else (0.08239705, 0.09176614, 0.0904255,)
                    )
                    self.train_data_std = (
                        (0.17386045, 0.16883257, 0.1768625)
                        if self.dataset_sub_cls == "MNIST"
                        else (0.25112887, 0.26145387, 0.26009334,)
                    )
                elif self.bias == "noise":
                    self.train_data_mean = (
                        (0.13405791,)
                        if self.dataset_sub_cls == "MNIST"
                        else (0.19938468,)
                    )
                    self.train_data_std = (
                        (0.23784825,)
                        if self.dataset_sub_cls == "MNIST"
                        else (0.28845804,)
                    )
                elif self.bias == "rotation":
                    self.train_data_mean = (
                        (0.0640235,)
                        if self.dataset_sub_cls == "MNIST"
                        else (0.14016011,)
                    )
                    self.train_data_std = (
                        (0.22387815,)
                        if self.dataset_sub_cls == "MNIST"
                        else (0.28369352,)
                    )
                elif (
                    self.bias == "addition"
                    or self.bias == "translation"
                    or self.bias == "expansion"
                ):
                    self.train_data_mean = (
                        (0.06402363,)
                        if self.dataset_sub_cls == "MNIST"
                        else (0.1401599,)
                    )
                    self.train_data_std = (
                        (0.22534915,)
                        if self.dataset_sub_cls == "MNIST"
                        else (0.28550556,)
                    )
            self.data_dir = kwargs.pop(
                "data_dir", "./data/image_classification/torchvision/"
            )
            self.num_workers = 1
            self.valid_size = kwargs.pop("valid_size", 0.1)
        elif self.dataset_cls == "TinyImageNet":
            if self.apply_grayscale:
                self.train_data_mean = (0.4519,)
                self.train_data_std = (0.2221,)
            else:
                self.train_data_mean = (
                    0.4802,
                    0.4481,
                    0.3975,
                )
                self.train_data_std = (
                    0.2302,
                    0.2265,
                    0.2262,
                )
            self.data_dir = kwargs.pop("data_dir", "./data/image_classification/")
            self.input_size = 64
            self.num_workers = kwargs.pop("num_workers", 2)
            self.valid_size = kwargs.pop("valid_size", 0.1)
        elif self.dataset_cls == "ImageNet":
            self.train_data_mean = (0.485, 0.456, 0.406)
            self.train_data_std = (0.229, 0.224, 0.225)
            self.data_dir = kwargs.pop("data_dir", "./data/image_classification/")
            self.input_size = 224
            self.num_workers = kwargs.pop("num_workers", 8)
            self.valid_size = kwargs.pop(
                "valid_size", 0.0416
            )  # To get ~50K (test set size)
        else:
            raise NameError()
        self.add_corrupted_test = kwargs.pop("add_corrupted_test", True)
        self.add_stylized_test = kwargs.pop("add_stylized_test", False)
        self.use_c_test_as_val = kwargs.pop("use_c_test_as_val", False)
        self.shuffle = kwargs.pop("shuffle", True)
        self.show_sample = kwargs.pop("show_sample", False)
        self.filter_classes = kwargs.pop("filter_classes", None)  # (start,end)
        self.download = kwargs.pop(
            "download", False
        )  # For safety (e.g. to not download ImageNet by accident)

    @property
    def filters(self):
        filters = []
        if self.filter_classes:
            filters.append("ClassesFilter")
        return filters


class NeuralDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.neural_dataset_loader"

    @baseline
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_frac = kwargs.pop("train_frac", 0.8)
        self.dataset = kwargs.pop("dataset", "CSRF19_V1")
        self.data_dir = "./data/monkey/toliaslab/{}".format(self.dataset)
        self.seed = kwargs.pop("seed", 1000)
        self.subsample = kwargs.pop("subsample", 1)
        self.crop = kwargs.pop("crop", 70)
        self.time_bins_sum = kwargs.pop("time_bins_sum", 12)


class MTLDatasetsConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.mtl_datasets_loader"

    @baseline
    def __init__(self, sub_configs, **kwargs):
        super().__init__(**kwargs)
        self.sub_configs = sub_configs

        # super().__init__(**kwargs)
        # self.neural_dataset_dict = kwargs.pop("neural_dataset_dict", {})
        # self.neural_dataset_config = NeuralDatasetConfig(
        #     **self.neural_dataset_dict
        # ).to_dict()
        # self.img_dataset_dict = kwargs.pop("img_dataset_dict", {})
        # self.img_dataset_config = ImageDatasetConfig(**self.img_dataset_dict).to_dict()
        #
        # self.update(**kwargs)

    def items(self):
        return self.sub_configs.items()

    def values(self):
        return self.sub_configs.values()

    def keys(self):
        return self.sub_configs.keys()

    def __getitem__(self, item):
        return self.sub_configs[item]

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MTLDatasetsConfig":
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
        Returns:
            :class:`MTLDatasetConfig`: An instance of a configuration object
        """
        sub_configs = {}
        for name, conf in config_dict.items():
            dataset_cls = next(iter(conf.keys()))
            sub_configs[name] = globals()[dataset_cls].from_dict(conf[dataset_cls])
        return cls(sub_configs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = {}
        for name, conf in self.sub_configs.items():
            output[name] = {conf.__class__.__name__: conf.to_dict()}
        return output


class RegressionDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.regression_dataset_loader"

    @baseline
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_cls = kwargs.pop("dataset_cls", "co2")
        self.apply_normalization = kwargs.pop("apply_data_normalization", False)
        self.apply_noise = kwargs.pop("apply_noise", False)
        self.input_size = kwargs.pop("input_size", 32)
        self.pin_memory = kwargs.pop("pin_memory", True)
        self.num_workers = kwargs.pop("num_workers", 0)
        self.shuffle = kwargs.pop("shuffle", True)
        self.valid_size = kwargs.pop("valid_size", 0.1)
        self.train_range = kwargs.pop("train_range", 10)

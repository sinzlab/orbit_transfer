from typing import Dict, Tuple

from bias_transfer.configs.dataset.base import DatasetConfig
from bias_transfer.tables.nnfabrik import Dataset


class ImageDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.img_dataset_loader"

    data_mean_defaults = {
        "CIFAR100": (0.5070751592371323, 0.48654887331495095, 0.4409178433670343,),
        "CIFAR10": (0.49139968, 0.48215841, 0.44653091),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "TinyImageNet_bw": (0.4519,),
        "TinyImageNet": (0.4802, 0.4481, 0.3975,),
        "ImageNet": (0.485, 0.456, 0.406),
        "MNIST": (0.1307,),
        "MNIST_color": (0.03685451, 0.0367535, 0.03952756),
        "MNIST_color_easy": (0.03685451, 0.0367535, 0.03952756),
        "MNIST_noise": (0.13405791,),
        "MNIST_rotation": (0.0640235,),
        "MNIST_translation": (0.06402363,),
        "MNIST_addition": (0.06402363,),
        "MNIST_clean": (0.06402363,),
        "MNIST_clean_shuffle": (0.06402363,),
        "FashionMNIST_color": (0.08239705, 0.09176614, 0.0904255,),
        "FashionMNIST_color_easy": (0.08239705, 0.09176614, 0.0904255,),
        "FashionMNIST_noise": (0.19938468,),
        "FashionMNIST_rotation": (0.14016011,),
        "FashionMNIST_translation": (0.1401599,),
        "FashionMNIST_addition": (0.1401599,),
        "FashionMNIST_clean": (0.1401599,),
        "FashionMNIST_clean_shuffle": (0.1401599,),
    }
    data_std_defaults = {
        "CIFAR100": (0.2673342858792401, 0.2564384629170883, 0.27615047132568404,),
        "CIFAR10": (0.24703223, 0.24348513, 0.26158784),
        "SVHN": (0.1980, 0.2010, 0.1970),
        "TinyImageNet_bw": (0.2221,),
        "TinyImageNet": (0.2302, 0.2265, 0.2262,),
        "ImageNet": (0.229, 0.224, 0.225),
        "MNIST": (0.3081,),
        "MNIST_color": (0.17386045, 0.16883257, 0.1768625),
        "MNIST_color_easy": (0.17386045, 0.16883257, 0.1768625),
        "MNIST_noise": (0.22387815,),
        "MNIST_rotation": (0.0640235,),
        "MNIST_translation": (0.22534915,),
        "MNIST_addition": (0.22534915,),
        "MNIST_clean": (0.22534915,),
        "MNIST_clean_shuffle": (0.22534915,),
        "FashionMNIST_color": (0.25112887, 0.26145387, 0.26009334,),
        "FashionMNIST_color_easy": (0.25112887, 0.26145387, 0.26009334,),
        "FashionMNIST_noise": (0.28845804,),
        "FashionMNIST_rotation": (0.28369352,),
        "FashionMNIST_translation": (0.28550556,),
        "FashionMNIST_addition": (0.28550556,),
        "FashionMNIST_clean": (0.28550556,),
        "FashionMNIST_clean_shuffle": (0.28550556,),
    }

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.dataset_cls: str = "CIFAR10"
        self.apply_augmentation: bool = True
        self.apply_normalization: bool = True
        self.apply_grayscale: bool = False
        self.apply_noise: Dict = {}
        self.convert_to_rgb: bool = False
        self.input_size: int = 32
        self.add_corrupted_test: bool = False
        self.add_stylized_test: bool = False
        self.use_c_test_as_val: bool = False
        self.show_sample: bool = False
        self.filter_classes: Tuple = ()  # (start,end)
        self.data_dir: str = "./data/image_classification/torchvision/"
        self.num_workers: int = 1
        dataset_id = (
            f"{self.dataset_sub_cls}_{self.bias}" if self.bias else self.dataset_cls
        )
        self.train_data_mean: Tuple[float] = self.data_mean_defaults[dataset_id]
        self.train_data_std: Tuple[float] = self.data_std_defaults[dataset_id]

        super().__init__(**kwargs)

    @property
    def filters(self):
        filters = []
        if self.filter_classes:
            filters.append("ClassesFilter")
        return filters
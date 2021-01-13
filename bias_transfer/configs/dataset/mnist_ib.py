from bias_transfer.configs.dataset.image import ImageDatasetConfig


class MNIST_IB(ImageDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls = "MNIST-IB"
        self.input_size: int = 40 if self.bias != "addition" else 80
        self.convert_to_rgb: bool = False
        self.bias: str = "clean"
        self.dataset_sub_cls: str = "FashionMNIST"  # could also be MNIST
        self.apply_data_normalization: bool = False
        self.apply_data_augmentation: bool = False
        self.add_corrupted_test: bool = False
        super().__init__(**kwargs)

from nntransfer.configs.dataset.image import ImageDatasetConfig


class MNIST(ImageDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls: str = "MNIST"
        self.input_width: int = 28
        self.input_height: int = 28
        self.apply_augmentation: bool = False
        self.apply_normalization: bool = False
        super().__init__(**kwargs)

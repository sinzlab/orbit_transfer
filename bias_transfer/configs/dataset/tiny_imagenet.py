from bias_transfer.configs.dataset.image import ImageDatasetConfig


class TinyImageNet(ImageDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls: str = "TinyImageNet"
        self.data_dir: str = "./data/image_classification/"
        self.input_size: int = 64
        self.num_workers: int = 2
        self.valid_size: int = 0.1
        super().__init__(**kwargs)

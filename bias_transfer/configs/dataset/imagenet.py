from bias_transfer.configs.dataset.image import ImageDatasetConfig


class ImageNet(ImageDatasetConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls: str = "ImageNet"
        self.data_dir: str = "./data/image_classification/"
        self.input_size: int = 224
        self.num_workers: int = 8
        self.valid_size: float = 0.0416  # To get ~50K (test set size)

        super().__init__(**kwargs)

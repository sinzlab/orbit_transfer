from nntransfer.configs.dataset.image import ImageDatasetConfig


class TinyImageNet(ImageDatasetConfig):
    fn = "bias_transfer.dataset.url_dataset_loader"

    data_mean_defaults = {
        "TinyImageNet_bw": (0.4519,),
        "TinyImageNet": (0.4802, 0.4481, 0.3975,),
    }
    data_std_defaults = {
        "TinyImageNet_bw": (0.2221,),
        "TinyImageNet": (0.2302, 0.2265, 0.2262,),
    }
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls: str = "TinyImageNet"
        self.data_dir: str = "./data/image_classification/"
        self.input_width: int = 64
        self.input_height: int = 64
        self.num_workers: int = 2
        self.valid_size: int = 0.1
        super().__init__(**kwargs)

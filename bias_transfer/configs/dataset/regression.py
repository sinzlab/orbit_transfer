from bias_transfer.configs.dataset.base import DatasetConfig
from bias_transfer.tables.nnfabrik import Dataset


class Regression(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.regression_dataset_loader"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls: str = "co2"
        self.apply_normalization: bool = False
        self.apply_noise: bool = False
        self.input_size: int = 32
        self.num_workers: int = 0
        self.train_range: int = 10
        super().__init__(**kwargs)

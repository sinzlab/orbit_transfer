from bias_transfer.configs.base import BaseConfig
from bias_transfer.tables.nnfabrik import Dataset


class Generated(BaseConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.transferred_dataset_loader"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.train_on_reduced_data: bool = False
        self.train_on_coreset: bool = False
        self.load_coreset: bool = False
        super().__init__(**kwargs)

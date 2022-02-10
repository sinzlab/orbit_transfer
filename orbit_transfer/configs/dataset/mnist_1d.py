import math

from nntransfer.configs import DatasetConfig
from nntransfer.tables.nnfabrik import Dataset


class MNIST1DDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "orbit_transfer.dataset.mnist_1d.dataset_fn"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.train_shift = 30
        self.original = False
        self.dataset_cls = "1DMNIST"
        super().__init__(**kwargs)


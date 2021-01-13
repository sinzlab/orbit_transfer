from bias_transfer.configs.dataset.base import DatasetConfig
from nnfabrik.main import Dataset


class NeuralDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.neural_dataset_loader"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.train_frac = 0.8
        self.dataset = "CSRF19_V1"
        self.data_dir = "./data/monkey/toliaslab/{}".format(self.dataset)
        self.seed = 1000
        self.subsample = 1
        self.crop = 70
        self.time_bins_sum = 12
        super().__init__(**kwargs)

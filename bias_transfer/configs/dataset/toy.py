from nntransfer.configs import DatasetConfig
from nntransfer.tables.nnfabrik import Dataset

class ToyDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.toy_dataset_loader"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.size: int = 100
        self.batch_size: int = 100
        self.num_workers: int = 0
        self.pin_memory: bool = False
        self.shuffle: bool = True
        self.cov_a = [[1, 0], [0, 1]]
        self.cov_b = [[1, 0], [0, 1]]
        self.mu_a = [-1, -1]
        self.mu_b = [1, 1]
        super().__init__(**kwargs)
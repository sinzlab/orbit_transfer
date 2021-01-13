from bias_transfer.configs.base import BaseConfig


class DatasetConfig(BaseConfig):
    config_name = "dataset"
    table = None
    fn = None

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.batch_size: int = 128
        self.pin_memory: bool = True
        self.shuffle: bool = True
        self.valid_size: float = 0.1
        super().__init__(**kwargs)



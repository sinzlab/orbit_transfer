from typing import Dict

from bias_transfer.configs.base import BaseConfig
from bias_transfer.tables.nnfabrik import Model


class ModelConfig(BaseConfig):
    config_name = "model"
    table = Model()
    fn = None

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dropout: float = 0.0
        self.get_intermediate_rep: Dict = {}
        super().__init__(**kwargs)



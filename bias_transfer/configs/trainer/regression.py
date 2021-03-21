from typing import Dict

from nntransfer.configs.trainer.base import TrainerConfig
from nntransfer.tables.nnfabrik import Trainer


class Regression(TrainerConfig):
    config_name = "trainer"
    table = Trainer()
    fn = "bias_transfer.trainer.regression"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.loss_functions: Dict = {"regression": "MSELoss"}
        self.maximize: bool = False
        self.noise_test: Dict = {}
        self.apply_noise_to_validation: bool = False
        self.show_epoch_progress: bool = False

        super().__init__(**kwargs)

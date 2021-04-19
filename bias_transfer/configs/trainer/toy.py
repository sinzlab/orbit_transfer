from typing import Dict, Tuple

from nntransfer.configs import TrainerConfig
from nntransfer.tables.nnfabrik import Trainer


class ToyTrainerConfig(TrainerConfig):
    config_name = "trainer"
    fn = "bias_transfer.trainer.toy"
    table = Trainer()

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.chkpt_options: Dict = {
            "save_every_n": 1000,
            "keep_best_n": 1,
            "keep_last_n": 1,
            "keep_selection": (),
        }
        self.optimizer_options: Dict = {
            "amsgrad": False,
            "lr": 0.001,
        }
        self.lr_milestones: Tuple = ()
        self.loss_functions: Dict = {"img_classification": "BCEWithLogitsLoss"}
        self.max_iter: int = 2000  # maximum number of iterations (epochs)
        self.patience: int = 100000
        self.maximize: bool = True  # if stop_function maximized or minimized
        self.show_epoch_progress: bool = True
        self.force_cpu: bool = True
        self.restore_best: bool = (
            False  # in case of loading best model at the end of training
        )
        super().__init__(**kwargs)

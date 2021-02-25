from typing import Dict, Tuple

from bias_transfer.configs.base import BaseConfig
from bias_transfer.tables.nnfabrik import *


class TrainerConfig(BaseConfig):
    config_name = "trainer"
    table = Trainer()

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.force_cpu: bool = False
        self.optimizer: str = "Adam"
        self.optimizer_options: Dict = {
            "amsgrad": False,
            "lr": 0.0003,
        }
        self.lr_decay: float = 0.8
        self.lr_warmup: int = 0
        self.epoch: int = 0
        self.scheduler: str = "manual"  # or "adaptive"
        self.chkpt_options: Dict = {
            "save_every_n": 1,
            "keep_best_n": 1,
            "keep_last_n": 1,
            "keep_selection": (),
        }
        self.keep_checkpoints: bool = False
        self.patience: int = 10
        self.threshold: float = 0.0001
        self.verbose: bool = False
        self.lr_milestones: Tuple = (30, 60)
        self.min_lr: float = 0.00001  # lr scheduler min learning rate
        self.threshold_mode: str = "rel"
        self.train_cycler: str = "LongCycler"
        self.train_cycler_args: Dict = {}
        self.loss_functions: Dict = {"img_classification": "CrossEntropyLoss"}
        self.loss_weighing: bool = False
        self.loss_accum_batch_n: int = (
            None  # for gradient accumulation how often to call opt.step
        )
        self.interval: int = (
            1  # interval at which objective evaluated for early stopping
        )
        self.max_iter: int = 100  # maximum number of iterations (epochs)
        self.restore_best: bool = (
            True  # in case of loading best model at the end of training
        )
        self.lr_decay_steps: int = 1  # Number of times the learning rate should be reduced before stopping the training.
        self.show_epoch_progress: bool = False

        self.main_loop_modules: list = []

        self.mtl: bool = False

        self.data_transfer: bool = False

        super().__init__(**kwargs)

    def conditional_assignment(self):
        if "ModelWrapper" in self.main_loop_modules:
            self.main_loop_modules.remove("ModelWrapper")
        self.main_loop_modules.append("ModelWrapper")  # we need this to be added last
        super().conditional_assignment()

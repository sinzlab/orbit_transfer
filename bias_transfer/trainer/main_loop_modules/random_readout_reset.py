import numpy as np
import torch
from torch import nn
from functools import partial

from .main_loop_module import MainLoopModule


class RandomReadoutReset(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.batch_progress = 0
        self.epoch_progress = 0

    def pre_epoch(self, model, mode, **options):
        super(RandomReadoutReset, self).pre_epoch(model, mode, **options)
        if self.train_mode and self.config.reset_linear_frequency.get("epoch"):
            if self.epoch_progress % self.config.reset_linear_frequency["epoch"] == 0:
                model.module.linear_readout.reset_parameters()
            self.epoch_progress += 1

    def pre_forward(self, model, inputs, task_key, shared_memory):
        super().pre_forward(model, inputs, task_key, shared_memory)
        if self.train_mode and self.config.reset_linear_frequency.get("batch"):
            if self.batch_progress % self.config.reset_linear_frequency["batch"] == 0:
                model.module.linear_readout.reset_parameters()
            self.batch_progress += 1
        return model, inputs

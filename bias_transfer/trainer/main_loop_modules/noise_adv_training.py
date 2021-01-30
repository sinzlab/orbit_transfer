import numpy as np
import torch
from torch import nn
from functools import partial

from neuralpredictors.training import LongCycler
from .main_loop_module import MainLoopModule


class NoiseAdvTraining(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.progress = 0.0
        if isinstance(self.train_loader, LongCycler):
            train_loader = self.train_loader.loaders
        self.step_size = 1 / (
            self.config.max_iter * len(self.train_loader["img_classification"])
        )
        if self.config.noise_adv_regression:
            self.criterion = nn.MSELoss()
        else:  # config.noise_adv_classification
            self.criterion = nn.BCELoss()
        objectives = {
            "Training": {"NoiseAdvTraining": {"loss": 0, "normalization": 0}},
            "Validation": {"NoiseAdvTraining": {"loss": 0, "normalization": 0}},
            "Test": {"NoiseAdvTraining": {"loss": 0, "normalization": 0}},
        }
        self.tracker.add_objectives(objectives)

    def pre_forward(self, model, inputs, task_key, shared_memory):
        super().pre_forward(model, inputs, task_key, shared_memory)
        noise_adv_lambda = (
            2.0 / (1.0 + np.exp(-self.config.noise_adv_gamma * self.progress)) - 1
        )
        if self.train_mode:
            self.progress += self.step_size
        return partial(model, noise_lambda=noise_adv_lambda), inputs

    def post_forward(self, outputs, loss, targets, **shared_memory):
        if not self.options.get("noise_adv",True):
            return outputs, loss, targets
        applied_std = shared_memory["applied_std"]
        num_inputs = (applied_std != 0).sum().item()
        extra_outputs = outputs[0]
        if applied_std is None:
            applied_std = torch.zeros_like(
                extra_outputs["noise_pred"], device=self.device
            )
        if self.config.noise_adv_classification:
            applied_std = (
                (applied_std > 0.0).type(torch.FloatTensor).to(device=self.device)
            )
        noise_loss = self.criterion(extra_outputs["noise_pred"], applied_std)
        self.tracker.log_objective(
            noise_loss.item() * num_inputs, (self.mode, "NoiseAdvTraining", "loss")
        )
        self.tracker.log_objective(
            num_inputs, (self.mode, "NoiseAdvTraining", "normalization"),
        )
        loss += self.config.noise_adv_loss_factor * noise_loss
        return outputs, loss, targets

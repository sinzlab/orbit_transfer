import os
import torch
import numpy as np

from bias_transfer.trainer.main_loop_modules.main_loop_module import MainLoopModule


class Mixup(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)

    def mixup_data(self, x):
        """
        Returns mixed inputs, and saves index and lambdas
        Adapted from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
        """
        alpha = self.config.regularization.get("alpha",1.0)
        if alpha > 0:
            self.lam = np.random.beta(alpha, alpha)
        else:
            self.lam = 1

        batch_size = x.size()[0]
        self.index = torch.randperm(batch_size).to(self.device)
        mixed_x = self.lam * x + (1 - self.lam) * x[self.index, :]
        return mixed_x

    def pre_forward(self, model, inputs, task_key, shared_memory):
        model, inputs = super().pre_forward(
            model, inputs, task_key, shared_memory
        )
        if self.train_mode:
            inputs = self.mixup_data(inputs)
        else:
            self.lam = 1.0
            self.index = torch.arange(inputs.size()[0])
        return model, inputs

    def post_forward(self, outputs, loss, targets, **shared_memory):
        if self.train_mode:
            loss += (1 - self.lam) * self.trainer.criterion["img_classification"](
                outputs, targets[self.index]
            )
        loss += self.lam * self.trainer.criterion["img_classification"](
            outputs, targets
        )
        _, predicted = outputs.max(1)
        correct = 100 * (
            self.lam * predicted.eq(targets).sum().item()
            + (1 - self.lam) * predicted.eq(targets[self.index]).sum().item()
        )
        self.tracker.log_objective(
            correct, keys=(self.mode, self.task_key, "accuracy"),
        )
        return outputs, loss, targets

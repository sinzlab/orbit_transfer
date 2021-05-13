from functools import partial

from nntransfer.trainer.utils.checkpointing import (
    RemoteCheckpointing,
    LocalCheckpointing,
    TemporaryCheckpointing,
    NoCheckpointing,
)
from nntransfer.trainer.trainer import Trainer
from nntransfer.trainer.utils import get_subdict, stringify
from nntransfer.trainer.utils.loss import *
from .img_classification_trainer import ImgClassificationTrainer
from .main_loop_modules import *
from nntransfer.trainer.main_loop_modules import *
from neuralpredictors.training.tracking import AdvancedTracker

from torch import nn, optim

from .regression_trainer import RegressionTrainer
from .transfer import DataGenerator


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = ToyTrainer(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()


def regression_trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = ToyRegressionTrainer(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()


class ToyTrainer(ImgClassificationTrainer):
    checkpointing_cls = NoCheckpointing

    def compute_loss(
        self,
        mode,
        task_key,
        loss,
        outputs,
        targets,
    ):
        if task_key != "transfer" and task_key in self.config.loss_functions:
            if not (
                self.config.regularization
                and self.config.regularization.get("regularizer") == "Mixup"
            ):  # otherwise this is done in the mainloop-module
                loss += self.criterion[task_key](outputs, targets)
                predicted = outputs > 0
                self.tracker.log_objective(
                    100 * predicted.eq(targets).sum().item(),
                    key=(mode, task_key, "accuracy"),
                )
            batch_size = targets.size(0)
            self.tracker.log_objective(
                batch_size,
                key=(mode, task_key, "normalization"),
            )
            self.tracker.log_objective(
                loss.item() * batch_size,
                key=(mode, task_key, "loss"),
            )
        return loss


class ToyRegressionTrainer(RegressionTrainer):
    checkpointing_cls = NoCheckpointing

    def move_data(self, batch_data):
        inputs, targets, data_key, _ = super().move_data(batch_data)
        if torch.is_tensor(inputs):
            inputs = inputs.reshape(-1, 1)
            # print(inputs.shape)
        if torch.is_tensor(targets):
            targets = targets.reshape(-1)
            # print(targets.shape)
        # elif isinstance(targets, dict):
        #     targets = {k: t.reshape(-1) for k, t in targets.items()}
            # for k,t in targets.items():
                # print(k, t.shape)
        return inputs, targets, data_key, None


class TransferDataGeneratorToyRegression(ToyRegressionTrainer, DataGenerator):
    pass


def regression_transfer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = TransferDataGeneratorToyRegression(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()

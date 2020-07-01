import os
from copy import copy
from functools import partial

import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn as cudnn
from tqdm import tqdm

from bias_transfer.models.utils import freeze_params, set_bn_to_eval
from bias_transfer.trainer.utils import (
    get_subdict,
    StopClosureWrapper,
    SchedulerWrapper,
)
import nnfabrik as nnf
from bias_transfer.configs.trainer import TrainerConfig
from bias_transfer.trainer.transfer import transfer_model
from bias_transfer.utils.io import load_checkpoint
from nnfabrik.utility.nn_helpers import load_state_dict
from .utils import early_stopping

# from .utils import save_best_state
from .main_loop_modules import *
from .utils import MTL_Cycler
from mlutils.training import LongCycler, ShortCycler, copy_state


class Trainer:
    def __init__(self, dataloaders, model, seed, uid, **kwargs):
        self.config = TrainerConfig.from_dict(kwargs)
        self.uid = nnf.utility.dj_helpers.make_hash(uid)
        self.model, self.device = nnf.utility.nn_helpers.move_to_device(model)
        nnf.utility.nn_helpers.set_random_seed(seed)
        self.seed = seed

        self.data_loaders = dataloaders
        self.task_keys = dataloaders["validation"].keys()
        self.tracker = self.get_tracker()
        self.main_loop_modules = [
            globals().get(k)(trainer=self) for k in self.config.main_loop_modules
        ]
        self.optimizer, self.stop_closure, self.criterion = self.get_training_controls()

        self.lr_scheduler = self.prepare_lr_schedule()

        start_epoch, self.best_eval, self.best_epoch = self.try_load_model()

        # Potentially freeze parts of the model
        freeze_params(self.model, self.config.freeze, self.config.readout_name)

        # Prepare iterator for training
        print("==> Starting model {}".format(self.config.comment), flush=True)
        self.train_stats = []
        self.epoch_iterator = early_stopping(
            self.model,
            self.stop_closure,
            self.config,
            interval=self.config.interval,
            patience=self.config.patience,
            start=start_epoch,
            max_iter=self.config.max_iter,
            maximize=self.config.maximize,
            tolerance=self.config.threshold,
            restore_best=self.config.restore_best,
            tracker=self.tracker,
            scheduler=self.lr_scheduler,
            lr_decay_steps=self.config.lr_decay_steps,
        )

    def get_tracker(self):
        raise NotImplementedError

    def try_load_model(self):
        best_epoch = 0
        best_eval = {k: {"eval": -100000, "loss": 100000} for k in self.task_keys}
        start_epoch = self.config.epoch
        path = "./checkpoint/ckpt.{}.pth".format(self.uid)
        # ... from checkpoint
        if os.path.isfile(path):
            model, best_eval, start_epoch = load_checkpoint(
                path, self.model, self.optimizer
            )
            best_epoch = start_epoch
        # ... or from transfer
        elif self.config.transfer_from_path and not self.config.transfer_after_train:
            self.data_loaders["train"] = transfer_model(
                self.model,
                self.config,
                criterion=self.criterion,
                device=self.device,
                data_loader=self.data_loaders["train"],
                restriction=self.config.transfer_restriction,
            )
        return start_epoch, best_eval, best_epoch

    def prepare_lr_schedule(self):
        lr_scheduler = None
        if self.config.scheduler is not None:
            if self.config.scheduler == "adaptive":
                if self.config.scheduler_options["mtl"]:
                    pass
                    # self.lr_scheduler = optim.lr_scheduler.StepLR(
                    #     self.optimizer, step_size=1, gamma=self.config.lr_decay
                    # )
                elif not self.config.scheduler_options["mtl"]:
                    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        factor=self.config.lr_decay,
                        patience=self.config.patience,
                        threshold=self.config.threshold,
                        verbose=self.config.verbose,
                        min_lr=self.config.min_lr,
                        mode="max" if self.config.maximize else "min",
                        threshold_mode=self.config.threshold_mode,
                    )
            elif self.config.scheduler == "manual":
                lr_scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=self.config.scheduler_options["milestones"],
                    gamma=self.config.lr_decay,
                )
        if self.config.lr_warmup:
            import pytorch_warmup as warmup

            warmup_scheduler = warmup.LinearWarmup(
                self.optimizer, warmup_period=self.config.lr_warmup,
            )
            lr_scheduler = SchedulerWrapper(lr_scheduler, warmup_scheduler)
        return lr_scheduler

    def main_loop(
        self,
        data_loader,
        mode="Training",
        epoch: int = 0,
        cycler="LongCycler",
        cycler_args={},
        module_options=None,
    ):
        reset_state_dict = {}
        if mode == "Training":
            train_mode = True
            batch_norm_train_mode = not self.config.freeze_bn
            return_outputs = False
        elif "BN" in mode:
            train_mode = False
            batch_norm_train_mode = True
            return_outputs = False
            reset_state_dict = copy_state(self.model)
        else:
            train_mode = False
            batch_norm_train_mode = False
            return_outputs = False
        module_options = {} if module_options is None else module_options
        self.model.train() if train_mode else self.model.eval()
        self.model.apply(partial(set_bn_to_eval, train_mode=batch_norm_train_mode))
        collected_outputs = []
        if hasattr(
            tqdm, "_instances"
        ):  # To have tqdm output without line-breaks between steps
            tqdm._instances.clear()
        data_cycler = globals().get(cycler)(data_loader, **cycler_args)

        with tqdm(
            iterable=enumerate(data_cycler),
            total=len(data_cycler),
            desc="{} Epoch {}".format(mode, epoch),
        ) as t, torch.enable_grad() if train_mode else torch.no_grad():
            for module in self.main_loop_modules:
                module.pre_epoch(self.model, mode, **module_options)
            if train_mode:
                self.optimizer.zero_grad()
            # Iterate over batches
            for batch_idx, batch_data in t:

                # Pre-Forward
                loss = torch.zeros(1, device=self.device)
                inputs, targets, task_key, batch_dict = self.move_data(batch_data)
                shared_memory = {}  # e.g. to remember where which noise was applied
                model_ = self.model
                for module in self.main_loop_modules:
                    model_, inputs = module.pre_forward(
                        model_, inputs, task_key, shared_memory
                    )
                # Forward
                outputs = model_(inputs)

                # Post-Forward and Book-keeping
                if return_outputs:
                    collected_outputs.append(outputs[0])
                for module in self.main_loop_modules:
                    outputs, loss, targets = module.post_forward(
                        outputs, loss, targets, **shared_memory
                    )

                loss = self.compute_loss(mode, task_key, loss, outputs, targets)
                self.tracker.display_log(tqdm_iterator=t, keys=(mode,))
                if train_mode:
                    # Backward
                    loss.backward()
                    for module in self.main_loop_modules:
                        module.post_backward(self.model)
                    if (
                        not self.config.optim_step_count
                        or (batch_idx + 1) % self.config.optim_step_count == 0
                    ):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

        if reset_state_dict:
            load_state_dict(
                self.model,
                reset_state_dict,
                ignore_missing=True,
                ignore_dim_mismatch=True,  # intermediate output is included here and may change in dim
            )

        if len(data_loader) == 1:
            objective = self.tracker.get_current_objective(
                mode, next(iter(data_loader.keys())), "accuracy"
            )
        else:
            objective = self.tracker.get_current_main_objective(mode)
        if return_outputs:
            return (collected_outputs, objective)
        else:
            return objective

    def train(self, cb):
        # train over epochs
        epoch = 0
        self.tracker.start_epoch()
        for epoch, dev_eval in self.epoch_iterator:
            if cb:
                cb()
            self.tracker.start_epoch()
            self.tracker.log_objective(self.optimizer.param_groups[0]["lr"], ("LR",))
            self.main_loop(
                data_loader=self.data_loaders["train"], mode="Training", epoch=epoch,
            )

        if self.config.lottery_ticket or epoch == 0:
            for module in self.main_loop_modules:
                module.pre_epoch(self.model, "Training")
        if self.config.transfer_after_train and self.config.transfer_from_path:
            transfer_model(
                self.model,
                self.config,
                criterion=self.criterion,
                device=self.device,
                data_loader=self.data_loaders["train"],
                restriction=self.config.transfer_restriction,
            )

        test_result = self.test_final_model(epoch)
        return (
            test_result,
            self.tracker.to_dict(),
            self.model.state_dict(),
        )

    def move_data(self, batch_data):
        raise NotImplementedError

    def get_training_controls(self):
        raise NotImplementedError

    def compute_loss(
        self, mode, data_key, loss, outputs, targets,
    ):
        raise NotImplementedError

    def test_final_model(self, epoch):
        raise NotImplementedError

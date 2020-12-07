import sys
from functools import partial

from tqdm import tqdm
import torch
from torch import optim, nn
import nnfabrik as nnf
from mlutils.training import copy_state

from bias_transfer.models.utils import (
    freeze_params,
    set_bn_to_eval,
    weight_reset,
    reset_params,
)
from bias_transfer.trainer.utils import SchedulerWrapper
from bias_transfer.configs.trainer import TrainerConfig
from nnfabrik.utility.nn_helpers import load_state_dict
from bias_transfer.trainer.utils.checkpointing import LocalCheckpointing
from bias_transfer.trainer.main_loop_modules import *
from bias_transfer.trainer.utils import LongCycler, MTL_Cycler
from bias_transfer.trainer.utils.early_stopping import early_stopping


class Trainer:
    checkpointing_cls = (
        LocalCheckpointing  # Open to chose between local and remote checkpointing
    )

    def __init__(self, dataloaders, model, seed, uid, cb, **kwargs):
        self.config = TrainerConfig.from_dict(kwargs)
        self.uid = uid
        self.model, self.device = nnf.utility.nn_helpers.move_to_device(model)
        nnf.utility.nn_helpers.set_random_seed(seed)
        self.seed = seed

        self.data_loaders = dataloaders
        self.task_keys = dataloaders["train"].keys()
        self.main_loop_modules = [
            globals().get(k)(trainer=self) for k in self.config.main_loop_modules
        ]
        self.optimizer, self.stop_closure, self.criterion = self.get_training_controls()
        self.lr_scheduler = self.prepare_lr_schedule()

        # Potentially reset parts of the model (after loading pretrained parameters)
        reset_params(self.model, self.config.reset)
        # Potentially freeze parts of the model
        freeze_params(self.model, self.config.freeze, self.config.readout_name)

        # Prepare iterator for training
        print("==> Starting model {}".format(self.config.comment), flush=True)
        self.train_stats = []
        checkpointing = self.checkpointing_cls(
            self.model,
            self.lr_scheduler,
            self.tracker,
            self.config.chkpt_options,
            self.config.maximize,
            partial(cb, uid=uid),
        )
        self.epoch_iterator = early_stopping(
            self.model,
            self.stop_closure,
            self.config,
            self.optimizer,
            checkpointing=checkpointing,
            interval=self.config.interval,
            patience=self.config.patience,
            max_iter=self.config.max_iter,
            maximize=self.config.maximize,
            tolerance=self.config.threshold,
            restore_best=self.config.restore_best,
            tracker=self.tracker,
            scheduler=self.lr_scheduler,
            lr_decay_steps=self.config.lr_decay_steps,
        )

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
        return_outputs=False,
    ):
        reset_state_dict = {}
        if mode == "Training":
            train_mode = True
            batch_norm_train_mode = not self.config.freeze_bn
            record_grad = True
        elif "BN" in mode:
            train_mode = False
            batch_norm_train_mode = True
            reset_state_dict = copy_state(self.model)
            record_grad = False
        elif mode == "Generation":
            train_mode = False
            batch_norm_train_mode = False
            record_grad = True
        else:
            train_mode = False
            batch_norm_train_mode = False
            record_grad = False
        module_options = {} if module_options is None else module_options
        self.model.train() if train_mode else self.model.eval()
        self.model.apply(partial(set_bn_to_eval, train_mode=batch_norm_train_mode))
        collected_outputs = []
        data_cycler = globals().get(cycler)(data_loader, **cycler_args)

        with tqdm(
            iterable=enumerate(data_cycler),
            total=len(data_cycler),
            desc="{} Epoch {}".format(mode, epoch),
            disable=self.config.show_epoch_progress,
            file=sys.stdout,
        ) as t, torch.enable_grad() if train_mode or record_grad else torch.no_grad():
            for module in self.main_loop_modules:
                module.pre_epoch(self.model, mode, **module_options)
            if train_mode:
                self.optimizer.zero_grad()
            # Iterate over batches
            for batch_idx, batch_data in t:

                # Pre-Forward
                loss = torch.zeros(1, device=self.device)
                inputs, targets, task_key, _ = self.move_data(batch_data)
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
                    collected_outputs.append(
                        {k: v.detach().to("cpu") for k, v in outputs[0].items()}
                    )
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
                        for module in self.main_loop_modules:
                            module.post_optimizer(self.model)
                        self.optimizer.zero_grad()

            for module in self.main_loop_modules:
                module.post_epoch(self.model)

        if reset_state_dict:
            load_state_dict(
                self.model,
                reset_state_dict,
                ignore_missing=True,
                ignore_dim_mismatch=True,  # intermediate output is included here and may change in dim
            )

        # if len(data_loader) == 1:
        #     objective = self.tracker.get_current_objective(
        #         mode, next(iter(data_loader.keys())), "accuracy"
        #     )
        # else:
        objective = self.tracker.get_current_main_objective(mode)
        if return_outputs:
            return objective, collected_outputs
        else:
            return objective

    def train(self):
        # train over epochs
        epoch = 0
        self.tracker.start_epoch()
        if hasattr(
            tqdm, "_instances"
        ):  # To have tqdm output without line-breaks between steps
            tqdm._instances.clear()
        for epoch, dev_eval in tqdm(
            iterable=self.epoch_iterator,
            total=self.config.max_iter,
            disable=(not self.config.show_epoch_progress),
        ):
            self.tracker.log_objective(self.optimizer.param_groups[0]["lr"], ("LR",))
            self.main_loop(
                data_loader=self.data_loaders["train"], mode="Training", epoch=epoch,
            )
            self.tracker.start_epoch()

        if self.config.lottery_ticket or epoch == 0:
            for module in self.main_loop_modules:
                module.pre_epoch(self.model, "Training")
        if self.config.transfer_after_train and self.config.transfer_from_path:
            self.transfer_model()

        test_result = self.test_final_model(epoch)
        return (
            test_result,
            self.tracker.state_dict(),
            self.model.state_dict(),
        )

    @property
    def tracker(self):
        raise NotImplementedError

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

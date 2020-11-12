from functools import partial

import torch

from bias_transfer.trainer.utils.checkpointing import RemoteCheckpointing, LocalCheckpointing
from bias_transfer.trainer.trainer import Trainer
from bias_transfer.trainer.utils import get_subdict
from bias_transfer.utils import stringify
from mlutils.tracking import AdvancedMultipleObjectiveTracker

from torch import nn, optim


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = ImgClassificationTrainer(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()


class ImgClassificationTrainer(Trainer):
    checkpointing_cls = RemoteCheckpointing

    def get_tracker(self):
        objectives = {
            "LR": 0,
            "Training": {
                "img_classification": {"loss": 0, "accuracy": 0, "normalization": 0}
            },
            "Validation": {
                "img_classification": {"loss": 0, "accuracy": 0, "normalization": 0},
                "patience": 0,
            },
        }
        tracker = AdvancedMultipleObjectiveTracker(
            main_objective=("img_classification", "accuracy"), **objectives
        )
        return tracker

    def get_training_controls(self):
        criterion, stop_closure = {}, {}
        for k in self.task_keys:
            if self.config.loss_weighing:
                pass
                # self.criterion[k] = XEntropyLossWrapper(
                #     getattr(nn, self.config.loss_functions[k])()
                # ).to(self.device)
            else:
                criterion[k] = getattr(nn, self.config.loss_functions[k])()
            stop_closure[k] = partial(
                self.main_loop,
                data_loader=get_subdict(self.data_loaders["validation"], [k]),
                mode="Validation",
                epoch=0,
                cycler_args={},
                cycler="LongCycler",
            )
        optimizer = getattr(optim, self.config.optimizer)(
            self.model.parameters(), **self.config.optimizer_options
        )
        return optimizer, stop_closure, criterion

    def move_data(self, batch_data):
        batch_dict = None
        data_key, inputs = batch_data[0], batch_data[1][0]

        if len(batch_data[1]) > 2:
            targets = [b.to(self.device) for b in batch_data[1][1:]]
        else:
            targets = batch_data[1][1].to(self.device)
        inputs = inputs.to(self.device, dtype=torch.float)
        return inputs, targets, data_key, batch_dict

    def compute_loss(
        self, mode, task_key, loss, outputs, targets,
    ):
        if not self.config.mixup:  # otherwise this is done in the mainloop-module
            loss += self.criterion["img_classification"](outputs, targets)
            _, predicted = outputs.max(1)
            self.tracker.log_objective(
                100 * predicted.eq(targets).sum().item(), keys=(mode, task_key, "accuracy"),
            )
        batch_size = targets.size(0)
        self.tracker.log_objective(
            batch_size, keys=(mode, task_key, "normalization"),
        )
        self.tracker.log_objective(
            loss.item() * batch_size, keys=(mode, task_key, "loss"),
        )
        return loss

    def test_final_model(self, epoch, bn_train=""):
        if not bn_train and self.config.eval_with_bn_train:
            self.test_final_model(epoch, bn_train=" BN=Train")
        # test the final model with noise on the dev-set
        # test the final model on the test set
        for k in self.task_keys:
            if "rep_matching" not in k and self.config.noise_test:
                for n_type, n_vals in self.config.noise_test.items():
                    for val in n_vals:
                        val_str = stringify(val)
                        mode = "Noise {} {}".format(n_type, val_str) + bn_train
                        objectives = {
                            mode: {k: {"accuracy": 0, "loss": 0, "normalization": 0,}}
                        }
                        self.tracker.add_objectives(objectives, init_epoch=True)
                        module_options = {
                            "noise_snr": None,
                            "noise_std": None,
                            "rep_matching": False,
                            "noise_adv": False,
                        }
                        module_options[n_type] = val
                        self.main_loop(
                            epoch=epoch,
                            data_loader=get_subdict(
                                self.data_loaders["validation"], [k]
                            ),
                            mode=mode,
                            cycler_args={},
                            cycler="LongCycler",
                            module_options=module_options,
                        )

            objectives = {
                "Test" + bn_train: {k: {"accuracy": 0, "loss": 0, "normalization": 0,}}
            }
            self.tracker.add_objectives(objectives, init_epoch=True)
            test_result = self.main_loop(
                epoch=epoch,
                data_loader=get_subdict(self.data_loaders["test"], [k]),
                mode="Test" + bn_train,
                cycler_args={},
                cycler="LongCycler",
                module_options={
                    "noise_snr": None,
                    "noise_std": None,
                    "rep_matching": False,
                    "noise_adv": False,
                },
            )
        if "c_test" in self.data_loaders:
            for k in self.task_keys:
                if "rep_matching" not in k:
                    for c_category in list(self.data_loaders["c_test"][k].keys()):
                        for c_level, data_loader in self.data_loaders["c_test"][k][
                            c_category
                        ].items():

                            objectives = {
                                c_category
                                + bn_train: {
                                    str(c_level): {
                                        "accuracy": 0,
                                        "loss": 0,
                                        "normalization": 0,
                                    }
                                }
                            }
                            self.tracker.add_objectives(objectives, init_epoch=True)
                            self.main_loop(
                                epoch=epoch,
                                data_loader={str(c_level): data_loader},
                                mode=c_category + bn_train,
                                cycler_args={},
                                cycler="LongCycler",
                                module_options={
                                    "noise_snr": None,
                                    "noise_std": None,
                                    "rep_matching": False,
                                    "noise_adv": False,
                                },
                            )
        if "st_test" in self.data_loaders:
            self.main_loop(
                epoch=epoch,
                data_loader={"img_classification": self.data_loaders["st_test"]},
                mode="Test-ST" + bn_train,
                cycler_args={},
                cycler="LongCycler",
                module_options={
                    "noise_snr": None,
                    "noise_std": None,
                    "rep_matching": False,
                    "noise_adv": False,
                },
            )
        return test_result

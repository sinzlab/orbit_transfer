from functools import partial
from itertools import chain

from nntransfer.trainer.utils.checkpointing import (
    RemoteCheckpointing,
    LocalCheckpointing,
)
from nntransfer.trainer.trainer import Trainer
from nntransfer.trainer.utils import get_subdict, stringify
from nntransfer.trainer.utils.loss import *
from .main_loop_modules import *
from nntransfer.trainer.main_loop_modules import *
from neuralpredictors.training.tracking import AdvancedTracker
from .losses import MSELikelihood, CELikelihood

from torch import nn, optim


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = ImgClassificationTrainer(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()


class ImgClassificationTrainer(Trainer):
    checkpointing_cls = LocalCheckpointing
    task = "img_classification"

    @property
    def main_loop_modules(self):
        try:
            return self._main_loop_modules
        except AttributeError:
            self._main_loop_modules = [
                globals().get(k)(trainer=self) for k in self.config.main_loop_modules
            ]
            return self._main_loop_modules

    @property
    def tracker(self):
        try:
            return self._tracker
        except AttributeError:
            objectives = {
                "Training": {
                    "LR": 0,
                    "img_classification": {
                        "loss": 0,
                        "accuracy": 0,
                        "normalization": 0,
                        "std": 0.0,
                    }
                },
                "Validation": {
                    "img_classification": {
                        "loss": 0,
                        "accuracy": 0,
                        "std": 0,
                        "normalization": 0,
                    },
                    "patience": 0,
                },
            }
            self._tracker = AdvancedTracker(
                main_objective=("img_classification", "loss")
                if self.config.main_objective == "loss"
                else ("img_classification", "accuracy"),
                **objectives
            )
            return self._tracker

    def get_training_controls(self):
        criterion, stop_closure = {}, {}
        for k in self.task_keys:
            if k == "transfer" or k not in self.config.loss_functions:
                continue  # no validation on this data and training is handled in mainloop modules
            opts = (
                self.config.loss_function_options[k]
                if self.config.loss_function_options
                else {}
            )
            if opts:
                opts["model"] = self.model
            criterion[k] = (
                globals().get(self.config.loss_functions[k])
                or getattr(nn, self.config.loss_functions[k])
            )(**opts)
            criterion[k] = criterion[k].to(self.device)

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
        data_key, inputs, targets = batch_data[0], batch_data[1][0], batch_data[1][1]

        # targets
        if isinstance(targets, dict):
            targets = {k: t.to(self.device) for k, t in targets.items()}
            if len(targets) == 1 and data_key != "transfer":
                targets = next(iter(targets.values()))
        else:
            targets = targets.to(self.device)

        # inputs
        if (
            isinstance(inputs, dict) and len(inputs) == 1
        ):  # TODO add support for multiple inputs
            inputs = next(iter(inputs.values()))
        inputs = inputs.to(self.device, dtype=torch.float)

        return inputs, targets, data_key, None

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
                gamma = 0.0
                for module in self.main_loop_modules:
                    if hasattr(module, "gamma"):
                        gamma = module.gamma
                        break
                loss += self.criterion[task_key](outputs, targets) * (1 - gamma)
                _, predicted = outputs.max(1)
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
            if hasattr(self.criterion[task_key], "log_var"):
                self.tracker.log_objective(
                    (torch.exp(self.criterion[task_key].log_var) ** 0.5).item()
                    * batch_size,
                    (
                        mode,
                        task_key,
                        "std",
                    ),
                )
        return loss

    def test_final_model(self, epoch, bn_train=""):
        deactivate_options = {
            "noise_snr": None,
            "noise_std": None,
            "rep_matching": False,
            "rep_monitoring": False,
            "noise_adv": False,
        }
        if not bn_train and self.config.eval_with_bn_train:
            self.test_final_model(epoch, bn_train=" BN=Train")
        # test the final model with noise on the dev-set
        # test the final model on the test set
        for k in self.task_keys:
            if k == "transfer":
                continue
            if "rep_matching" not in k and self.config.noise_test:
                for n_type, n_vals in self.config.noise_test.items():
                    for val in n_vals:
                        val_str = stringify(val)
                        mode = "Noise {} {}".format(n_type, val_str) + bn_train
                        objectives = {
                            mode: {
                                k: {
                                    "accuracy": 0,
                                    "loss": 0,
                                    "normalization": 0,
                                    "std": 0,
                                }
                            }
                        }
                        self.tracker.add_objectives(objectives, init_epoch=True)

                        module_options = deactivate_options.copy()
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
                "Test"
                + bn_train: {
                    k: {
                        "accuracy": 0,
                        "loss": 0,
                        "normalization": 0,
                        "std": 0,
                    }
                }
            }
            self.tracker.add_objectives(objectives, init_epoch=True)
            test_result = self.main_loop(
                epoch=epoch,
                data_loader=get_subdict(self.data_loaders["test"], [k]),
                mode="Test" + bn_train,
                cycler_args={},
                cycler="LongCycler",
                module_options=deactivate_options,
            )
        if "c_test" in self.data_loaders:
            for k in self.task_keys:
                if "rep_matching" not in k:
                    for c_category in list(self.data_loaders["c_test"][k].keys()):
                        for c_level, data_loader in self.data_loaders["c_test"][k][
                            c_category
                        ].items():
                            # TODO make work again for multiple c_level!

                            objectives = {
                                c_category
                                + bn_train: {
                                    "img_classification": {
                                        "accuracy": 0,
                                        "loss": 0,
                                        "normalization": 0,
                                        "std": 0,
                                    }
                                }
                            }
                            self.tracker.add_objectives(objectives, init_epoch=True)
                            self.main_loop(
                                epoch=epoch,
                                data_loader={"img_classification": data_loader},
                                mode=c_category + bn_train,
                                cycler_args={},
                                cycler="LongCycler",
                                module_options=deactivate_options,
                            )
        if "st_test" in self.data_loaders:
            self.main_loop(
                epoch=epoch,
                data_loader={"img_classification": self.data_loaders["st_test"]},
                mode="Test-ST" + bn_train,
                cycler_args={},
                cycler="LongCycler",
                module_options=deactivate_options,
            )
        if "rot_test" in self.data_loaders:
            objectives = {
                "Rotation Test": {
                    "img_classification": {
                        "accuracy": 0,
                        "loss": 0,
                        "normalization": 0,
                        "std": 0,
                    }
                }
            }
            self.tracker.add_objectives(objectives, init_epoch=True)
            self.main_loop(
                epoch=epoch,
                data_loader={"img_classification": self.data_loaders["rot_test"]},
                mode="Rotation Test" + bn_train,
                cycler_args={},
                cycler="LongCycler",
                module_options=deactivate_options,
            )
        return test_result

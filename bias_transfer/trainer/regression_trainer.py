import torch

from nntransfer.trainer.utils.checkpointing import RemoteCheckpointing
from bias_transfer.trainer.img_classification_trainer import ImgClassificationTrainer
from nntransfer.trainer.utils import get_subdict, arctanh
from neuralpredictors.training.tracking import AdvancedTracker


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = RegressionTrainer(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()


class RegressionTrainer(ImgClassificationTrainer):
    checkpointing_cls = RemoteCheckpointing
    task = "regression"

    @property
    def tracker(self):
        try:
            return self._tracker
        except AttributeError:
            objectives = {
                "LR": 0,
                "Training": {"regression": {"loss": 0, "normalization": 0,
                                            "std": 0,
                                            },
                             },
                "Validation": {
                    "regression": {"loss": 0, "normalization": 0,
                                   "std": 0,
                                   },
                    "patience": 0,
                },
            }
            self._tracker = AdvancedTracker(
                main_objective=("regression", "loss"), **objectives
            )
            return self._tracker

    def compute_loss(self, mode, task_key, loss, outputs, targets):
        if task_key != "transfer" and task_key in self.config.loss_functions:
            reg_loss = self.criterion["regression"](outputs.reshape((-1,)), targets)
            if self.config.scale_loss_with_arctanh:
                reg_loss = arctanh(reg_loss)

            loss += reg_loss
            _, predicted = outputs.max(1)
            batch_size = targets.size(0)
            self.tracker.log_objective(
                batch_size,
                key=(mode, task_key, "normalization"),
            )
            self.tracker.log_objective(
                loss.item() * batch_size,
                key=(mode, task_key, "loss"),
            )
            if hasattr(self.criterion[task_key],"log_var") :
                self.tracker.log_objective(
                    (torch.exp(self.criterion[task_key].log_var)**0.5).item()*batch_size, (mode, task_key, "std",)
                )
        return loss

    def test_final_model(self, epoch, bn_train=""):
        if not bn_train and self.config.eval_with_bn_train:
            self.test_final_model(epoch, bn_train=" BN=Train")
        # test the final model on the test set
        for k in self.task_keys:
            if k == "transfer":
                continue
            objectives = {
                "Test"
                + bn_train: {
                    k: {
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
                module_options={},
            )
        return test_result

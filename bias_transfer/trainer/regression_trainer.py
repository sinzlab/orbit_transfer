from bias_transfer.trainer.utils.checkpointing import RemoteCheckpointing
from bias_transfer.trainer.img_classification_trainer import ImgClassificationTrainer
from bias_transfer.trainer.utils import get_subdict, arctanh
from neuralpredictors.tracking import AdvancedMultipleObjectiveTracker


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = RegressionTrainer(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()


class RegressionTrainer(ImgClassificationTrainer):
    checkpointing_cls = RemoteCheckpointing

    @property
    def tracker(self):
        try:
            return self._tracker
        except AttributeError:
            objectives = {
                "LR": 0,
                "Training": {"regression": {"loss": 0, "normalization": 0}},
                "Validation": {
                    "regression": {"loss": 0, "normalization": 0},
                    "patience": 0,
                },
            }
            self._tracker = AdvancedMultipleObjectiveTracker(
                main_objective=("regression", "loss"), **objectives
            )
            return self._tracker

    def compute_loss(
        self, mode, task_key, loss, outputs, targets,
    ):
        reg_loss = self.criterion["regression"](outputs.reshape((-1,)), targets)
        if self.config.scale_loss_with_arctanh:
            reg_loss = arctanh(reg_loss)

        loss += reg_loss
        _, predicted = outputs.max(1)
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
        # test the final model on the test set
        for k in self.task_keys:
            objectives = {"Test" + bn_train: {k: {"loss": 0, "normalization": 0,}}}
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

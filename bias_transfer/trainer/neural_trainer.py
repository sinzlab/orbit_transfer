from functools import partial

import numpy as np
from torch import nn

from bias_transfer.trainer.trainer import Trainer
from bias_transfer.trainer.utils import NBLossWrapper, get_subdict
from neuralpredictors import measures as mlmeasures
from nnvision.utility import measures
from nnvision.utility.measures import get_poisson_loss


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = NeuralTrainer(dataloaders, model, seed, uid, **kwargs)
    return t.train(cb)


class NeuralTrainer(Trainer):
    def get_tracker(self):
        if self.config.track_training:
            tracker_dict = dict(
                correlation=partial(
                    get_correlations(),
                    self.model,
                    self.dataloaders["validation"],
                    device=self.device,
                    per_neuron=False,
                ),
                poisson_loss=partial(
                    get_poisson_loss(),
                    self.model,
                    self.dataloaders["validation"],
                    device=self.device,
                    per_neuron=False,
                    avg=False,
                ),
            )
            if hasattr(self.model, "tracked_values"):
                tracker_dict.update(self.model.tracked_values)
            tracker = MultipleObjectiveTracker(**tracker_dict)
        else:
            tracker = None
        return tracker

    def get_training_controls(self):
        self.criterion, self.stop_closure = {}, {}
        for k in self.val_keys:
            if "img_classification" not in k:
                pass
                if self.config.loss_weighing:
                    self.criterion[k] = NBLossWrapper().to(self.device)
                else:
                    self.criterion[k] = getattr(
                        mlmeasures, self.config.loss_functions[k]
                    )(avg=self.config.avg_loss)
                self.stop_closure[k] = {}
                self.stop_closure[k]["eval"] = partial(
                    getattr(measures, "get_correlations"),
                    dataloaders=dataloaders["validation"][k],
                    device=self.device,
                    per_neuron=False,
                    avg=True,
                )
                self.stop_closure[k]["loss"] = partial(
                    get_poisson_loss,
                    dataloaders=dataloaders["validation"][k],
                    device=self.device,
                    per_neuron=False,
                    avg=False,
                )

        params = list(self.model.parameters())
        if self.config.loss_weighing:
            for _, loss_object in self.criterion.items():
                params += list(loss_object.parameters())
        self.optimizer = getattr(optim, self.config.optimizer)(
            params, **self.config.optimizer_options
        )

    def compute_loss(
        self,
        average_loss,
        correct,
        data_key,
        loss,
        outputs,
        targets,
        task_dict,
        total,
        total_loss,
        total_loss_weight,
    ):
        if "img_classification" not in data_key:
            loss += neural_full_objective(
                self.model,
                outputs,
                data_loader,
                self.criterion["neural"],
                self.scale_loss,
                data_key,
                inputs,
                targets,
            )
            total["neural"] += get_correlations(
                self.model,
                batch_dict,
                device=self.device,
                as_dict=False,
                per_neuron=False,
            )
            task_dict["neural"]["eval"] = average_loss(total["neural"])
            total_loss["neural"] += loss.item()
            task_dict["neural"]["epoch_loss"] = average_loss(total_loss["neural"])
            if self.config.loss_weighing:
                total_loss_weight["neural"] += np.exp(
                    self.criterion["neural"].log_w.item()
                )
                task_dict["neural"]["loss_weight"] = average_loss(
                    total_loss_weight["neural"]
                )
        return loss

    def test_neural_model(model, data_loader, device, epoch, eval_type="Validation"):
        loss = get_poisson_loss(
            model, data_loader, device, as_dict=False, per_neuron=False
        )
        eval = get_correlations(
            model, data_loader, device=device, as_dict=False, per_neuron=False
        )
        results = {"neural": {"eval": eval, "loss": loss}}
        print(
            "Neural {} Epoch {}: eval={}, loss={}".format(
                eval_type, epoch, results["neural"]["eval"], results["neural"]["loss"]
            )
        )
        return results

    def test_final_model(
        self,
        best_epoch,
        best_eval,
        config,
        criterion,
        dataloaders,
        device,
        epoch,
        model,
        seed,
        test_n_iterations,
        val_keys,
        val_n_iterations,
    ):
        # test the final model with noise on the dev-set
        # test the final model on the test set
        test_results_dict, dev_final_results_dict = {}, {}
        for k in self.val_keys:
            if "img_classification" not in k:
                dev_final_results = test_neural_model(
                    model,
                    data_loader=dataloaders["validation"][k],
                    device=device,
                    epoch=epoch,
                    eval_type="Validation",
                )
                test_results = test_neural_model(
                    model,
                    data_loader=dataloaders["test"][k],
                    device=device,
                    epoch=epoch,
                    eval_type="Test",
                )
                dev_final_results_dict.update(dev_final_results)
                test_results_dict.update(test_results)
        final_results = {
            "test_results": test_results_dict,
            "dev_eval": best_eval,
            "epoch": best_epoch,
            "dev_final_results": dev_final_results_dict,
        }
        return final_results, test_results_dict


def neural_full_objective(
    model, outputs, dataloader, criterion, scale_loss, data_key, inputs, targets
):

    loss = criterion(outputs, targets)
    loss_scale = (
        np.sqrt(len(dataloader[data_key].dataset) / inputs.shape[0])
        if scale_loss
        else 1.0
    )
    loss *= loss_scale
    if scale_loss:
        loss += model.regularizer(data_key)
    return loss

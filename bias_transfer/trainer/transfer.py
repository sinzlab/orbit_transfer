from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from bias_transfer.dataset.dataset_classes.npy_dataset import NpyDataset
from bias_transfer.trainer.img_classification_trainer import ImgClassificationTrainer
from bias_transfer.trainer.main_loop_modules.fisher_estimation import FisherEstimation
from bias_transfer.trainer.regression_trainer import RegressionTrainer
from bias_transfer.trainer.trainer import Trainer


class PseudoTrainer(Trainer):
    def __init__(self, dataloaders, model, seed, uid, cb, **kwargs):
        super().__init__(dataloaders, model, seed, uid, cb, **kwargs)
        self.main_task = list(self.task_keys)[0]

    def train(self):
        self.tracker.start_epoch()
        if hasattr(tqdm, "_instances"):
            tqdm._instances.clear()

        if self.config.save_representation:
            train = self.generate_rep_dataset(data="train")
        elif self.config.extract_coreset:
            train = self.extract_coreset(
                data="train",
                method=self.config.extract_coreset.get("method"),
                size=self.config.extract_coreset.get("size"),
            )
            if f"{self.main_task}_cs" in self.data_loaders["train"]:  # update coreset
                cs = self.data_loaders["train"][f"{self.main_task}_cs"].dataset
                train["source_cs"] = np.concatenate([train["source_cs"], cs.samples])
                train["target_cs"] = np.concatenate([train["target_cs"], cs.targets])
        else:
            train = None

        if self.config.compute_fisher:
            self.estimate_fisher(data="train")
        elif self.config.compute_si_omega:
            self.compute_omega()

        if self.config.reset_for_new_task:
            self.model.reset_for_new_task()
        return 0.0, {}, self.model.state_dict(), train

    def generate_rep_dataset(self, data):
        _, collected_outputs = self.main_loop(
            data_loader=self.data_loaders[data],
            epoch=0,
            mode="Validation",
            return_outputs=True,
        )
        outputs = {}
        for rep_name in collected_outputs[0].keys():
            outputs[rep_name] = torch.cat(
                [batch_output[rep_name] for batch_output in collected_outputs]
            ).numpy()
        if self.config.save_input:
            collected_inputs = []
            data_loader = next(iter(self.data_loaders[data].values()))
            for src, _ in data_loader:
                collected_inputs.append(src)
            outputs["source"] = torch.cat(collected_inputs).numpy()
        return outputs

    def extract_coreset(self, data, method, size):
        print(method)
        indices = list(range(len(self.data_loaders[data][self.main_task].dataset)))
        if method == "random":
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            coreset_idx, remain_idx = indices[:size], indices[size:]
        elif method == "k-center":
            try:
                dataset = self.data_loaders[data][self.main_task].dataset.data.numpy()
            except:
                dataset = self.data_loaders[data][
                    self.main_task
                ].dataset.dataset.samples.numpy()

            def update_distance(dists, x_train, current_id):
                for i in range(x_train.shape[0]):
                    current_dist = np.linalg.norm(
                        x_train[i, :] - x_train[current_id, :]
                    )
                    dists[i] = np.minimum(current_dist, dists[i])
                return dists

            dists = np.full(dataset.shape[0], np.inf)
            current_id = 0
            coreset_idx = []
            remain_idx = indices
            for _ in range(size):
                dists = update_distance(dists, dataset, current_id)
                coreset_idx.append(current_id)
                remain_idx.remove(current_id)
                current_id = np.argmax(dists)
        collected_inputs = []
        collected_labels = []
        data_loader = next(iter(self.data_loaders[data].values()))
        for src, trg in data_loader:
            collected_inputs.append(src)
            collected_labels.append(trg)
        inputs = torch.cat(collected_inputs).numpy()
        labels = torch.cat(collected_labels).numpy()
        outputs = {
            "source": inputs[remain_idx],
            "source_cs": inputs[coreset_idx],
            "target": labels[remain_idx],
            "target_cs": labels[coreset_idx],
        }
        return outputs

    def estimate_fisher(self, data):
        task_key = next(iter(self.data_loaders[data].keys()))
        data_loader = self.data_loaders[data][task_key]
        indices = list(range(len(data_loader.dataset)))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        indices = indices[: self.config.compute_fisher.get("num_samples", 128)]
        sampler = SubsetRandomSampler(indices)
        data_loader = torch.utils.data.DataLoader(
            data_loader.dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            shuffle=False,
        )
        objectives = {
            "Generation": {task_key: {"loss": 0, "accuracy": 0, "normalization": 0}},
        }
        self.tracker.add_objectives(objectives, init_epoch=True)
        self.main_loop_modules.append(FisherEstimation(trainer=self))
        self.main_loop(
            data_loader={task_key: data_loader},
            epoch=0,
            mode="Generation",
            return_outputs=False,
        )

    def compute_omega(self):
        print("Compute Synaptic Intelligence Omega")
        damping_factor = self.config.compute_si_omega.get("damping_factor", 0.0001)
        # Loop over all parameters
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(
                    self.model, f"{n}_SI_prev_task"
                )  # initial param values
                omega = getattr(self.model, f"{n}_SI_omega")
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_new = omega / (p_change ** 2 + damping_factor)

                # Store these new values in the model
                self.model.register_buffer(f"{n}_importance", omega_new)
                delattr(self.model, f"{n}_SI_omega")
                delattr(self.model, f"{n}_SI_prev_task")


class TransferPseudoTrainerClassificiation(ImgClassificationTrainer, PseudoTrainer):
    pass


class TransferPseudoTrainerRegression(RegressionTrainer, PseudoTrainer):
    pass


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = TransferPseudoTrainerClassificiation(
        dataloaders, model, seed, uid, cb, **kwargs
    )
    return t.train()


def regression_trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = TransferPseudoTrainerRegression(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()

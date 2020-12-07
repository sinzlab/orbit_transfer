from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from bias_transfer.configs.trainer import TransferTrainerConfig
from bias_transfer.dataset.dataset_classes.npy_dataset import NpyDataset
from bias_transfer.trainer.img_classification_trainer import ImgClassificationTrainer
from bias_transfer.trainer.main_loop_modules.fisher_estimation import FisherEstimation
from bias_transfer.trainer.regression_trainer import RegressionTrainer
from bias_transfer.trainer.trainer import Trainer


class PseudoTrainer(Trainer):
    def __init__(self, dataloaders, model, seed, uid, cb, **kwargs):
        super().__init__(dataloaders, model, seed, uid, cb, **kwargs)
        self.config = TransferTrainerConfig.from_dict(kwargs)

    def train(self):
        self.tracker.start_epoch()
        if hasattr(tqdm, "_instances"):
            tqdm._instances.clear()
        if self.config.save_representation:
            train = self.generate_rep_dataset(data="train")
        else:
            train = None
        if self.config.compute_fisher:
            self.estimate_fisher(data="train")
        elif self.config.compute_si_omega:
            self.compute_omega()
        return train, self.model.state_dict()

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
            for src, _ in self.data_loaders[data]["img_classification"]:
                collected_inputs.append(src)
            outputs["source"] = torch.cat(collected_inputs).numpy()
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

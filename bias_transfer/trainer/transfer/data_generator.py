import math
from copy import copy

from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from bias_transfer.dataset.transferred_loader import load_npy
from bias_transfer.trainer.transfer.kpca import kpca
from nntransfer.models.utils import copy_ensemble_buffer_to_param
from nntransfer.models.wrappers import IntermediateLayerGetter
from nntransfer.trainer.trainer import Trainer
from nntransfer.dataset.dataset_classes.npy_dataset import NpyDataset
from bias_transfer.trainer.img_classification_trainer import ImgClassificationTrainer
from bias_transfer.trainer.main_loop_modules.fisher_estimation import FisherEstimation
from bias_transfer.trainer.main_loop_modules.function_regularization.fromp import FROMP
from bias_transfer.trainer.regression_trainer import RegressionTrainer
from bias_transfer.trainer.transfer.coreset_extraction import extract_coreset


class DataGenerator(Trainer):
    def __init__(self, dataloaders, model, seed, uid, cb, **kwargs):
        super().__init__(dataloaders, model, seed, uid, cb, **kwargs)
        self.main_task = list(self.task_keys)[0]
        print(
            self.main_loop_modules
        )  # to initialize this, we need to access this property once

    def train(self):
        self.tracker.start_epoch()
        if hasattr(tqdm, "_instances"):
            tqdm._instances.clear()

        if self.config.save_representation:
            print("save representation")
            train = self.generate_rep_dataset(data="train")
        elif self.config.extract_coreset:
            save_in_model = self.config.extract_coreset.pop("save_in_model", False)
            train = extract_coreset(
                data_loader=self.data_loaders["train"][self.main_task],
                model=self.model,
                seed=self.seed,
                device=self.device,
                **self.config.extract_coreset,
            )
            if f"{self.main_task}_cs" in self.data_loaders["train"]:  # update coreset
                cs = self.data_loaders["train"][f"{self.main_task}_cs"].dataset
                train["source_cs"] = np.concatenate([train["source_cs"], cs.samples])
                train["target_cs"] = np.concatenate([train["target_cs"], cs.targets])
            if save_in_model:
                self.model.coreset = torch.tensor(train["source_cs"]).to(self.device)
        else:
            train = {}

        if self.config.compute_fisher:
            self.estimate_fisher(data="train")
        elif self.config.compute_si_omega:
            self.compute_omega()
        elif self.config.bayesian_to_deterministic:
            self.bayesian_to_deterministic()
        elif self.config.init_fromp:
            self.init_fromp(train)

        if self.config.reset_for_new_task:
            self.model.reset_for_new_task()

        if isinstance(self.model, IntermediateLayerGetter):
            self.model = self.model._model
        return 0.0, {"transfer_data": train}, self.model.state_dict()

    def generate_rep_dataset(self, data):
        n_samples = self.config.compute_covariance.get("n_samples", 1)
        outputs = {rep_name: [] for rep_name in self.model.return_layers.keys()}
        print(self.model)
        for s in range(n_samples):
            if self.config.compute_covariance:
                task_key = next(iter(self.data_loaders[data].keys()))
                objectives = {
                    "MC-Dropout": {
                        task_key: {"loss": 0, "accuracy": 0, "normalization": 0}
                    },
                }
                self.tracker.add_objectives(objectives, init_epoch=True)
            _, collected_outputs = self.main_loop(
                data_loader=self.data_loaders[data],
                epoch=0,
                mode="MC-Dropout" if self.config.compute_covariance else "Validation",
                return_outputs=True,
            )
            for rep_name in collected_outputs[0].keys():
                out_tensor = torch.cat(
                    [batch_output[rep_name] for batch_output in collected_outputs]
                )
                if self.config.apply_softmax:
                    T = self.config.softmax_temp if self.config.softmax_temp else 1.0
                    out_tensor = F.softmax(out_tensor / T, dim=1)
                outputs[rep_name].append(out_tensor)
            if self.config.compute_covariance.get("ensembling") and s < n_samples -1:
                copy_ensemble_buffer_to_param(self.model, ensemble_iteration=s)

        for rep_name, reps in outputs.items():
            reps = torch.stack(reps)
            reps = reps.transpose(0, 1).transpose(1, 2)  # train_samples x rep_dim x ensemble_members
            outputs[rep_name] = reps.numpy()
        if self.config.save_input:
            collected_inputs = []
            data_loader = next(iter(self.data_loaders[data].values()))
            for src, _ in data_loader:
                collected_inputs.append(src)
            outputs["source"] = torch.cat(collected_inputs).numpy()
        return outputs

    def compute_rep_covariance(self, reps, return_dict, rep_name):
        if self.config.compute_covariance.get("type", "diagonal") == "diagonal":
            return_dict[rep_name + "_var"] = torch.var(reps, dim=0)
        else:
            # ensemble_members, train_samples, rep_dim = reps.shape
            reps = reps.transpose(0, 1).transpose(1, 2)  # train_samples x rep_dim x ensemble_members
            # mean = return_dict[f"{rep_name}"].unsqueeze(-1)
            return_dict[f"{rep_name}_cov_V"] = reps #(reps - mean) / math.sqrt(ensemble_members - 1)
            # reps = reps.reshape(
            #     (ensemble_members, -1)
            # ).T  # (train_samples * rep_dim) x ensemble_members
            # eps_inv = 1 / self.config.compute_covariance.get("eps", 1e-10)
            # if self.config.compute_covariance.get("precision", "float") == "double":
            #     reps = reps.type(torch.DoubleTensor)
            # rank = self.config.compute_covariance.get("n_components", reps.shape[1])
            # vs, lambdas = kpca(reps, n_components=rank)
            # print(vs, lambdas)
            # C = vs @ torch.diag(lambdas) @ vs.T + eps -> NxN
            # A = eps_inv * vs  # -> NxM
            # B = (
            #     torch.inverse(torch.diag(1 / lambdas) + (vs.T * eps_inv) @ vs)
            #     @ (vs.T * eps_inv)
            # ).T  # -> NxM
            # return_dict[rep_name + "_cov_A"] = A.reshape(
            #     (train_samples, rep_dim, ensemble_members)
            # )
            # return_dict[rep_name + "_cov_B"] = B.reshape(
            #     (train_samples, rep_dim, ensemble_members)
            # )
            # C_inv = eps_inv - A @ B.T
            # return_dict[f"{rep_name}_cov_V"] = reps.reshape(
            #     (train_samples, rep_dim, ensemble_members)
            # )
            # rep_name_ = rep_name.replace(".", "__")
            # self.model._model.register_buffer(f"{rep_name_}_cov_lambdas", lambdas)

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
        self._main_loop_modules.append(FisherEstimation(trainer=self))
        self.main_loop(
            data_loader={task_key: data_loader},
            epoch=0,
            mode="Generation",
            return_outputs=False,
        )

    def compute_covariance(self, data, batch_size=32):
        task_key = next(iter(self.data_loaders[data].keys()))
        data_loader = self.data_loaders[data][task_key]
        np.random.seed(self.seed)
        data_loader = torch.utils.data.DataLoader(
            data_loader.dataset,
            batch_size=batch_size,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            shuffle=False,
        )
        self.model.eval()
        covariance = 0
        # self.state['fisher'] = torch.zeros_like(self.state['mu'])
        for data, label in tqdm(data_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            covariance += FROMP.compute_covariance(data, self.model).detach().cpu()
        return covariance

    def init_fromp(self, train):
        train["covariance"] = self.compute_covariance(
            data="train",
            batch_size=self.config.compute_covariance.get("batch_size", 32),
        )
        main_task = next(iter(self.data_loaders["train"].keys()))
        load_npy(
            "_cs",
            f"{main_task}_cs",
            train,
            self.data_loaders,
            self.data_loaders["train"][main_task],
        )
        self.data_loaders.update(train)
        print(self.data_loaders.keys())
        fromp = FROMP(self)
        fromp.init_task(self.config.regularization.get("eps", 1e-5))
        train["preds_prev_mem_prev_model"] = (
            fromp.preds_prev_mem_prev_model.detach().cpu().numpy()
        )
        train["kernel_inv_prev_mem_prev_model"] = [
            k.detach().cpu().numpy() for k in fromp.kernel_inv_prev_mem_prev_model
        ]

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
                delattr(self.model, f"{n}_SI_omega")
                delattr(self.model, f"{n}_SI_prev_task")
                if isinstance(self.model, IntermediateLayerGetter):
                    n = n[len("_model__") :]
                    self.model._model.register_buffer(f"{n}_importance", omega_new)
                else:
                    self.model.register_buffer(f"{n}_importance", omega_new)

    def bayesian_to_deterministic(self):
        print("Transform covariance into importance")
        # Loop over all parameters
        params = dict(self.model.named_parameters())
        for n, p in params.items():
            if "_posterior_log_var" in n:
                n_new = n[: -len("_posterior_log_var")]
                if n_new[-1] == "w":
                    n_new += "eight"
                elif n_new[-1] == "b":
                    n_new += "ias"
                n_new = n_new.replace(".", "__")
                importance = 1 / torch.exp(p)
                self.model.register_buffer(f"{n_new}_importance", importance)
            elif "_posterior_v" in n:
                n_new = n[: -len("_posterior_v")]
                if n_new[-1] == "w":
                    n_new += "eight"
                elif n_new[-1] == "b":
                    n_new += "ias"
                n_new = n_new.replace(".", "__")
                self.model.register_buffer(f"{n_new}_importance_v", p)


class TransferDataGeneratorClassificiation(ImgClassificationTrainer, DataGenerator):
    pass


class TransferDataGeneratorRegression(RegressionTrainer, DataGenerator):
    pass


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = TransferDataGeneratorClassificiation(
        dataloaders, model, seed, uid, cb, **kwargs
    )
    return t.train()


def regression_trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = TransferDataGeneratorRegression(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()

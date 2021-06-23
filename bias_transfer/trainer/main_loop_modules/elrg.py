import copy
import os
from collections import OrderedDict
from functools import partial

import torch

from nntransfer.trainer.main_loop_modules.main_loop_module import MainLoopModule


class ELRG(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.eps = self.config.regularization.get("eps", 1e-8)
        self.prior_var = torch.tensor(self.config.regularization.get("prior_var", 1e-5))
        self.num_samples = self.config.regularization.get("num_samples", 10)
        self.train_len = len(
            self.trainer.data_loaders["train"]["img_classification"].dataset
        )

    def pre_forward(self, model, inputs, task_key, shared_memory):
        super().pre_forward(model, inputs, task_key, shared_memory)
        model_ = partial(model, num_samples=self.num_samples)
        return model_, inputs

    def post_forward(self, outputs, loss, targets, **shared_memory):
        loss += self._calculate_kl_term() / self.train_len  # usually this will be the main loss
        targets = torch.cat(self.num_samples * [targets])
        return outputs, loss, targets

    def _calculate_kl_term(self):
        """
        Calculates and returns the KL divergence of the new posterior and the previous
        iteration's posterior.
        """
        model = self.trainer.model
        means = model.get_parameters("posterior_mean")
        vs = model.get_parameters("posterior_v", keep_first_dim=True)
        log_vars = model.get_parameters("posterior_log_var")
        vars = torch.exp(log_vars)
        alpha = model.alpha
        D = means.shape[0]

        # Calculate KL for individual normal distributions over parameters
        kl = (vars / self.prior_var - log_vars).sum()
        kl += (torch.norm(vs, 2, dim=1) ** 2).sum() * alpha / self.prior_var
        kl -= torch.logdet(
            torch.eye(model.rank, device=self.device) + alpha * ((vs / vars) @ vs.T)
        )
        kl += torch.norm(means, 2) ** 2 / self.prior_var
        kl += D * (torch.log(self.prior_var) - 1.0)
        return 0.5 * kl

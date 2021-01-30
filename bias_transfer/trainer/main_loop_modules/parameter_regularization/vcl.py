import copy
import os
from collections import OrderedDict
from functools import partial

import torch

from bias_transfer.trainer.main_loop_modules.main_loop_module import MainLoopModule


class VCL(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.eps = self.config.regularization.get("eps", 1e-8)
        self.num_samples = self.config.regularization.get("num_samples", 10)
        self.train_len = len(
            self.trainer.data_loaders["train"]["img_classification"].dataset
        )

    def pre_forward(self, model, inputs, task_key, shared_memory):
        super().pre_forward(model, inputs, task_key, shared_memory)
        model_ = partial(model, num_samples=self.num_samples)
        return model_, inputs

    def post_forward(self, outputs, loss, targets, **shared_memory):
        loss += self._calculate_kl_term() / self.train_len
        targets = targets.repeat(self.num_samples).view(-1)
        return outputs, loss, targets

    def _calculate_kl_term(self):
        """
        Calculates and returns the KL divergence of the new posterior and the previous
        iteration's posterior. See equation L3, slide 14.
        """
        model = self.trainer.model
        # Prior
        prior_means = model.get_parameters("prior_mean")
        prior_log_vars = model.get_parameters("prior_log_var")
        prior_vars = torch.exp(prior_log_vars)

        # Posterior
        posterior_means = model.get_parameters("posterior_mean")
        posterior_log_vars = model.get_parameters("posterior_log_var")
        posterior_vars = torch.exp(posterior_log_vars)

        # Calculate KL for individual normal distributions over parameters
        kl_elementwise = (
            posterior_vars / (prior_vars + self.eps)
            + torch.pow(prior_means - posterior_means, 2) / (prior_vars + self.eps)
            - 1
            + (prior_log_vars - posterior_log_vars)
        )

        # Sum KL over all parameters
        return 0.5 * kl_elementwise.sum()

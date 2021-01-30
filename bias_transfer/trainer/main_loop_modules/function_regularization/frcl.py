from functools import partial

import torch

from bias_transfer.trainer.main_loop_modules.main_loop_module import MainLoopModule


class FRCL(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.eps = self.config.regularization.get("eps", 1e-8)
        self.num_samples = self.config.regularization.get("num_samples", 9)
        self.train_len = len(
            self.trainer.data_loaders["train"]["img_classification"].dataset
        )

    def pre_forward(self, model, inputs, task_key, shared_memory):
        super().pre_forward(model, inputs, task_key, shared_memory)
        model_ = partial(model, num_samples=self.num_samples)
        return model_, inputs

    def post_forward(self, outputs, loss, targets, **shared_memory):
        if self.train_mode:
            loss += self._calculate_kl_term() / self.train_len
        targets = targets.repeat(self.num_samples).view(-1)
        return outputs, loss, targets

    @staticmethod
    def kl(m1, S1, m2, S2):
        S2 = S2 + torch.eye(S2.shape[0]).to(S2) * 1e-3
        S1 = S1 + torch.eye(S1.shape[0]).to(S1) * 1e-3
        S2_ = torch.inverse(S2)
        return 0.5 * (
            torch.trace(S2_ @ S1)
            + (m2 - m1).T @ S2_ @ (m2 - m1)
            - S1.shape[0]
            + torch.logdet(S2)
            - torch.logdet(S1)
        )

    def _calculate_kl_term(self):
        model = self.trainer.model
        kls = 0
        for i in range(model.num_classes):
            # kls -= kl_divergence(self.w_distr[i], self.w_prior)
            kls -= self.kl(
                model.mu[i],
                model.L[i] @ model.L[i].T,
                model.w_prior.mean,
                model.w_prior.covariance_matrix,
            )
            # curr_task_kls = -kls.item()

        if model.prev:
            out_dim = model.num_classes  # model.prev_num_classes
            phi_i = model.core_forward(model.coreset_prev)
            cov_i = phi_i @ phi_i.T + torch.eye(phi_i.shape[0]).to(self.device) * 1e-6
            # p_u = MultivariateNormal(torch.zeros(cov_i.shape[0]).to(self.device),
            #                          covariance_matrix=cov_i * self.sigma_prior)
            # kls -= sum([kl_divergence(self.prev_tasks_distr[i][j], p_u) for j in range(self.out_dim)])
            prev_kls = sum(
                [
                    self.kl(
                        model._buffers[f"mu_prev_{j}"],
                        model._buffers[f"cov_prev_{j}"],
                        torch.zeros(cov_i.shape[0]).to(self.device),
                        cov_i * model.sigma_prior,
                    )
                    for j in range(out_dim)
                ]
            )
            # if state is not None:
            #     state.kls.append(prev_kls.item())
            kls -= prev_kls

        # if state is not None:
        #     state.kls.append(curr_task_kls)
        #     state.kls_div_nk = kls.item() / N_k
        # Sum KL over all parameters
        return -kls

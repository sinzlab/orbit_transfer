import copy
import os
from collections import OrderedDict

import torch

from nntransfer.trainer.main_loop_modules.main_loop_module import MainLoopModule


class ParamDistance(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.sp_state_dict = OrderedDict()
        state_dict = self.trainer.model.state_dict()
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                self.sp_state_dict[k] = v.clone()
            else:
                self.sp_state_dict[k] = copy.deepcopy(v)
        self.warned = False
        self.gamma = self.config.regularization.get("gamma", 1.0)
        self.ignore_layers = self.config.regularization.get("ignore_layers", ())
        self.use_full_importance = self.config.regularization.get(
            "use_full_importance", False
        )
        self.use_elrg_importance = self.config.regularization.get(
            "use_elrg_importance", False
        )
        if self.use_elrg_importance:
            self.elrg_alpha = self.config.regularization.get("elrg_alpha", 1.0)
            self.pre_compute_elrg()
        custom_importance = self.config.regularization.get("custom_importance", {})
        if custom_importance:
            for n, param in self.trainer.model.named_parameters():
                n_ = n.replace(".", "__")
                importance = torch.from_numpy(custom_importance[n_]).to(self.device)
                self.trainer.model.register_buffer(f"{n}_importance", importance)

        objectives = {  # TODO: make adaptable to other tasks!
            "Training": {"img_classification": {"P-Dist": 0}},
            "Validation": {"img_classification": {"P-Dist": 0}},
            "Test": {"img_classification": {"P-Dist": 0}},
        }
        self.tracker.add_objectives(objectives, init_epoch=True)

    def pre_compute_elrg(self):
        model = self.trainer.model
        self.gammas = {}
        self.deltas = {}
        for n, param in model.named_parameters():
            n_ = n.replace(".", "__")
            importance = getattr(model, f"{n_}_importance")
            v = getattr(model, f"{n_}_importance_v")
            k = v.shape[0]
            importance = importance.reshape(-1)
            v = v.reshape(k, -1)
            self.gammas[n] = (importance * v).t()
            self.deltas[n] = torch.inverse(self.elrg_alpha * torch.eye(k, device=v.device) + (v * importance) @ v.t())

    def post_forward(self, outputs, loss, targets, **shared_memory):
        model = self.trainer.model
        if self.train_mode:
            reg_loss = torch.zeros(1, dtype=torch.float32, device=self.trainer.device)
            for n, param in model.named_parameters():
                if n not in self.sp_state_dict:
                    if not self.warned:
                        print(f"skipping {n}")
                        self.warned = True
                    continue
                for l in self.ignore_layers:
                    if l in n:
                        continue
                n_ = n.replace(".", "__")
                importance = getattr(model, f"{n_}_importance", torch.tensor(1.0))
                if self.use_full_importance:
                    param = param.flatten()
                    starting_point = self.sp_state_dict[n].flatten()
                    distance = (
                        (param - starting_point)
                        @ importance
                        @ (param - starting_point).t()
                    ).squeeze()
                elif self.use_elrg_importance:
                    distance = (importance * (param - self.sp_state_dict[n]) ** 2).sum()
                    param = param.flatten()
                    starting_point = self.sp_state_dict[n].flatten()
                    d = (param - starting_point).t() @ self.gammas[n]
                    distance -= d @ self.deltas[n] @ d.t()
                else:
                    distance = (importance * (param - self.sp_state_dict[n]) ** 2).sum()
                reg_loss = reg_loss + distance
            loss += self.gamma * reg_loss
            self.tracker.log_objective(
                loss.item(), (self.mode, self.task_key, "P-Dist")
            )
            return outputs, loss, targets
        else:
            return outputs, loss, targets

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
        self.alpha = self.config.regularization.get("alpha", 1.0)
        self.ignore_layers = self.config.regularization.get("ignore_layers", ())
        objectives = {  # TODO: make adaptable to other tasks!
            "Training": {"img_classification": {"P-Dist": 0}},
            "Validation": {"img_classification": {"P-Dist": 0}},
            "Test": {"img_classification": {"P-Dist": 0}},
        }
        self.tracker.add_objectives(objectives, init_epoch=True)

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
                importance = getattr(model, f"{n_}_importance", 1.0)
                distance = (importance * (param - self.sp_state_dict[n]) ** 2).sum()
                reg_loss = reg_loss + distance
            loss += self.alpha * reg_loss
            self.tracker.log_objective(
                loss.item(), (self.mode, self.task_key, "P-Dist")
            )
            return outputs, loss, targets
        else:
            return outputs, loss, targets

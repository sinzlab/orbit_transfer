import copy
import os
from collections import OrderedDict

from torch import nn
import torch
from torch.autograd import Variable

from bias_transfer.trainer.main_loop_modules.main_loop_module import MainLoopModule
from mlutils.training import copy_state


class L2SP(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        objectives = {  # TODO: make adaptable to other tasks!
            "Training": {"img_classification": {"L2SP": 0}},
            "Validation": {"img_classification": {"L2SP": 0}},
            "Test": {"img_classification": {"L2SP": 0}},
        }
        self.tracker.add_objectives(objectives, init_epoch=True)
        self.sp_state_dict = OrderedDict()
        state_dict = self.trainer.model.state_dict()
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                self.sp_state_dict[k] = v.clone()
            else:
                self.sp_state_dict[k] = copy.deepcopy(v)
        self.distance = torch.nn.MSELoss(reduction="sum")
        self.warned = False
        self.alpha = self.config.regularization.get("alpha", 1.0)
        self.ignore_layers = self.config.regularization.get("ignore_layers", ())

    def post_forward(self, outputs, loss, targets, **shared_memory):
        if self.train_mode:
            l2sp_loss = Variable(
                torch.zeros(1).type(torch.FloatTensor).to(self.trainer.device),
                requires_grad=True,
            )
            for p_name, param in self.trainer.model.named_parameters():
                if p_name not in self.sp_state_dict:
                    if not self.warned:
                        print(f"skipping {p_name}")
                        self.warned = True
                    continue
                for l in self.ignore_layers:
                    if l in p_name:
                        continue
                l2sp_loss = l2sp_loss + self.distance(param, self.sp_state_dict[p_name])
            loss += self.alpha * l2sp_loss
            self.tracker.log_objective(loss.item(), (self.mode, self.task_key, "L2SP"))
            return outputs, loss, targets
        else:
            return outputs, loss, targets

import os

from torch import nn
import torch
from torch.autograd import Variable

from .main_loop_module import MainLoopModule


class L2SP(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        objectives = {  # TODO: make adaptable to other tasks!
            "Training": {"img_classification": {"L2SP": 0}},
            "Validation": {"img_classification": {"L2SP": 0}},
            "Test": {"img_classification": {"L2SP": 0}},
        }
        self.tracker.add_objectives(objectives, init_epoch=True)
        assert os.path.isfile(self.config.transfer_from_path), "Error: no model file found!"
        self.sp_state_dict = torch.load(self.config.transfer_from_path, map_location=self.trainer.device)
        self.distance = torch.nn.MSELoss(reduction="sum")

    def post_forward(self, outputs, loss, targets, **shared_memory):
        if self.train_mode:
            l2sp_loss = Variable(torch.zeros(1).type(torch.FloatTensor).to(self.trainer.device), requires_grad=True)
            for p_name, param in self.trainer.model.named_parameters():
                if p_name not in self.sp_state_dict:
                    print(f"skipping {p_name}")
                    continue
                l2sp_loss = l2sp_loss + self.distance(param, self.sp_state_dict[p_name])
            loss += self.config.l2sp * l2sp_loss
            self.tracker.log_objective(
                loss.item(), (self.mode, self.task_key, "L2SP")
            )
            return outputs, loss, targets
        else:
            return outputs, loss, targets

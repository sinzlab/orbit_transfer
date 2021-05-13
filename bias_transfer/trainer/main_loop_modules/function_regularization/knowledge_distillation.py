from torch import nn
import torch
import torch.nn.functional as F


from .representation import RepresentationRegularization


class KnowledgeDistillation(RepresentationRegularization):
    def __init__(self, trainer):
        super().__init__(trainer, name="KD")
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.T = self.config.regularization.get("softmax_temp", 1.0)

    def rep_distance(self, output, target, var=None):
        kd_loss = self.criterion(
            F.log_softmax(output / self.T, dim=1), F.softmax(target / self.T, dim=1)
        )
        if var is not None:
            print(kd_loss.shape)
            print(var.shape)
            print(var)
            kd_loss *= var
            prin()
        return kd_loss * self.T * self.T


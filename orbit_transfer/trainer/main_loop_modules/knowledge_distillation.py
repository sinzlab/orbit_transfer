from torch import nn
import torch
import torch.nn.functional as F


from .representation import RepresentationRegularization


class KnowledgeDistillation(RepresentationRegularization):
    def __init__(self, trainer):
        super().__init__(trainer, name="KD")
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.T = self.config.regularization.get("softmax_temp", 1.0)

    def rep_distance(self, output, target, *args, **kwargs):
        kd_loss = self.criterion(
            F.log_softmax(output / self.T, dim=1), F.softmax(target.squeeze() / self.T, dim=1)
        )
        return kd_loss * self.T * self.T


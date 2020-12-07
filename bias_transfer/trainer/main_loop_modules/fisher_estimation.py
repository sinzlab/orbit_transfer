from .main_loop_module import MainLoopModule
import torch.nn.functional as F


class FisherEstimation(MainLoopModule):
    """
    Implementation adapted from https://github.com/GMvandeVen/continual-learning/blob/master/continual_learner.py
    """

    def __init__(self, trainer):
        super().__init__(trainer)
        self.num_samples = self.config.compute_fisher.get("num_samples", 128)
        self.empirical = self.config.compute_fisher.get("empirical", False)
        self.est_fisher_info = {}

    def pre_epoch(self, model, mode, **options):
        super().pre_epoch(model, mode, **options)
        # Prepare <dict> to store estimated Fisher Information matrix
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                self.est_fisher_info[n] = p.detach().clone().zero_()

    def post_forward(self, outputs, loss, targets, **shared_memory):
        model = self.trainer.model
        if self.empirical:
            # use provided label to calculate loglikelihood --> "empirical Fisher":
            label = targets
        else:
            # use predicted label to calculate loglikelihood:
            label = outputs.max(1)[1]
        # calculate negative log-likelihood
        loss = self.trainer.criterion[self.task_key](outputs, label)
        # loss = F.nll_loss(F.log_softmax(outputs, dim=1), label)

        # Calculate gradient of negative loglikelihood
        model.zero_grad()
        loss.backward()

        # Square gradients and keep running sum
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                if p.grad is not None:
                    self.est_fisher_info[n] += p.grad.detach() ** 2

        return outputs, loss, targets

    def post_epoch(self, model):
        # Normalize by sample size used for estimation
        est_fisher_info = {
            n: p / self.num_samples for n, p in self.est_fisher_info.items()
        }

        # Store new values in the network
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                # precision (approximated by diagonal Fisher Information matrix)
                model.register_buffer(
                    f"{n}_importance", est_fisher_info[n],
                )

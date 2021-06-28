import torch

from nntransfer.trainer.main_loop_modules.main_loop_module import MainLoopModule


class RepresentationRegularization(MainLoopModule):
    def __init__(self, trainer, name="RDL"):
        super().__init__(trainer)
        self.task = self.trainer.task
        objectives = {
            "Training": {name: {"distance": 0, "normalization": 0}},
            "Validation": {name: {"distance": 0, "normalization": 0}},
            "Test": {name: {"distance": 0, "normalization": 0}},
        }
        self.tracker.add_objectives(objectives, init_epoch=True)
        self.name = name
        self.alpha_0 = self.config.regularization.get("alpha", 1.0)
        self.alpha = 0.0

    def pre_epoch(self, model, mode, **options):
        super().pre_epoch(model, mode, **options)
        if self.config.regularization.get("decay_alpha"):
            self.alpha = self.alpha_0 * (1 - (self.epoch / self.config.max_iter))
        else:
            self.alpha = self.alpha_0

    def rep_distance(self, output, target, *args, **kwargs):
        raise NotImplementedError()

    def post_forward(self, outputs, loss, targets, **shared_memory):
        extra_outputs = outputs[0]
        if self.train_mode and (
            self.task_key == "transfer" or self.config.single_input_stream
        ):
            pred_loss = torch.zeros(1, device=self.device)
            rep_batch_size = 0
            for key in targets.keys():
                if key == "class" or "var" in key or "cov" in key:
                    continue
                pred_loss += self.rep_distance(extra_outputs[key], targets[key])
                if not rep_batch_size:
                    rep_batch_size = targets[key].shape[0]
            loss += self.alpha * pred_loss
            self.tracker.log_objective(
                pred_loss.item() * rep_batch_size, (self.mode, self.name, "distance")
            )
            self.tracker.log_objective(
                rep_batch_size, (self.mode, self.name, "normalization")
            )
            return outputs, loss, targets.get("class", next(iter(targets.values())))
        else:
            return outputs, loss, targets

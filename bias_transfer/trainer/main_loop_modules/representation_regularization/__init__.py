import torch

from nntransfer.trainer.main_loop_modules.main_loop_module import MainLoopModule


class RepresentationRegularization(MainLoopModule):
    def __init__(self, trainer, name="RDL"):
        super().__init__(trainer)
        objectives = {  # TODO: make adaptable to other tasks!
            "Training": {"img_classification": {name: 0}},
            "Validation": {"img_classification": {name: 0}},
            "Test": {"img_classification": {name: 0}},
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

    def rep_distance(self, output, target):
        raise NotImplementedError()

    def post_forward(self, outputs, loss, targets, **shared_memory):
        extra_outputs = outputs[0]
        if self.train_mode and (
            self.task_key == "transfer" or self.config.single_input_stream
        ):
            pred_loss = torch.zeros(1, device=self.device)
            for key in targets.keys():
                if key == "class":
                    continue
                pred_loss += self.rep_distance(extra_outputs[key], targets[key])
            loss += self.alpha * pred_loss
            self.tracker.log_objective(
                pred_loss.item(), (self.mode, "img_classification", self.name)
            )
            return outputs, loss, targets.get("class", next(iter(targets.values())))
        else:
            return outputs, loss, targets

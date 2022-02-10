import torch

from nntransfer.trainer.main_loop_modules.main_loop_module import MainLoopModule


class RepresentationRegularization(MainLoopModule):
    def __init__(self, trainer, name="RDL"):
        super().__init__(trainer)
        self.task = self.trainer.task
        self.teacher = self.trainer.teacher_model
        objectives = {
            "Training": {name: {"distance": 0, "normalization": 0}},
            "Validation": {name: {"distance": 0, "normalization": 0}},
            "Test": {name: {"distance": 0, "normalization": 0}},
        }
        self.tracker.add_objectives(objectives, init_epoch=True)
        self.name = name
        self.gamma_0 = self.config.regularization.get("gamma", 0.5)
        self.gamma = 0.0

    def pre_epoch(self, model, mode, **options):
        super().pre_epoch(model, mode, **options)
        if self.config.regularization.get("decay_gamma"):
            self.gamma = self.gamma_0 * (1 - (self.epoch / self.config.max_iter))
        else:
            self.gamma = self.gamma_0

    def pre_forward(self, model, inputs, task_key, shared_memory):
        model, inputs = super().pre_forward(model, inputs, task_key, shared_memory)
        if self.teacher is not None:
            shared_memory["teacher_output"] = self.teacher(inputs)
        return model, inputs

    def rep_distance(self, output, target, *args, **kwargs):
        raise NotImplementedError()

    def post_forward(self, outputs, loss, targets, **shared_memory):
        extra_outputs = outputs[0]

        if self.train_mode and (
            self.task_key == "transfer" or self.config.single_input_stream
        ):
            if self.teacher:
                if not isinstance(targets, dict):
                    targets = {"class": targets}
                targets.update(shared_memory["teacher_output"][0])
            pred_loss = torch.zeros(1, device=self.device)
            rep_batch_size = 0
            for key in extra_outputs.keys():
                if key == "class" or "var" in key or "cov" in key:
                    continue
                pred_loss += self.rep_distance(extra_outputs[key], targets[key])
                if not rep_batch_size:
                    rep_batch_size = targets[key].shape[0]
            loss += self.gamma * pred_loss
            self.tracker.log_objective(
                pred_loss.item() * rep_batch_size, (self.mode, self.name, "distance")
            )
            self.tracker.log_objective(
                rep_batch_size, (self.mode, self.name, "normalization")
            )
            return outputs, loss, targets.get("class", next(iter(targets.values())))
        else:
            return outputs, loss, targets

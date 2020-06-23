from functools import partial
from .main_loop_module import MainLoopModule


class ModelWrapper(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)

    def pre_forward(self, model, inputs, task_key, shared_memory):
        super().pre_forward(model, inputs, task_key, shared_memory)
        if self.config.mtl:
            if "img_classification" in task_key:
                model_ = partial(model, data_key=task_key, classification=True)
            else:
                model_ = partial(model, data_key=task_key)
        elif "img_classification" in task_key:
            model_ = model
        else:
            model_ = partial(model, data_key=task_key)
        return model_, inputs

    def post_forward(self, outputs, loss, targets, **shared_memory):
        if isinstance(outputs, tuple):
            return outputs[1], loss, targets
        return outputs, loss, targets

from functools import partial
from .main_loop_module import MainLoopModule


class ModelWrapper(MainLoopModule):
    def __init__(self, model, config, device, data_loader, seed):
        super().__init__(model, config, device, data_loader, seed)

    def post_forward(self, outputs, loss, targets, extra_losses, train_mode, **kwargs):
        if isinstance(outputs, tuple):
            return outputs[1], loss, targets
        return outputs, loss, targets

    def pre_forward(self, model, inputs, shared_memory, train_mode, **kwargs):
        data_key = kwargs.pop("data_key", None)
        if self.mtl:
            if "img_classification" in data_key:
                model_ = partial(model, data_key=data_key, classification=True)
            else:
                model_ = partial(model, data_key=data_key)
        elif "img_classification" in data_key:
            model_ = model
        else:
            model_ = partial(model, data_key=data_key)
        return model_, inputs

from .main_loop_module import MainLoopModule


class SynapticIntelligence(MainLoopModule):
    """
    Implementation adapted from https://github.com/GMvandeVen/continual-learning/blob/master/continual_learner.py
    """

    def __init__(self, trainer):
        super().__init__(trainer)
        # Register starting param-values
        model = trainer.model
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                model.register_buffer(f"{n}_SI_prev_task", p.data.clone())
        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.w = {}
        self.old_params = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                self.w[n] = p.data.clone().zero_()
                self.old_params[n] = p.data.clone()

    def pre_forward(self, model, inputs, task_key, shared_memory):
        super().pre_forward(model, inputs, task_key, shared_memory)
        # Save current parameters
        for n, p in self.params.items():
            n = n.replace(".", "__")
            self.old_params[n] = p.clone().detach()
        return model, inputs

    def post_optimizer(self, model):
        # Accumulate the w
        for n, p in self.params.items():
            n = n.replace(".", "__")
            delta = p.detach() - self.old_params[n]
            if (
                p.grad is not None
            ):  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= p.grad * delta  # w[n] is >=0

    def post_epoch(self, model):
        # Store to be used in final steps
        for n, w in self.w.items():
            model.register_buffer(f"{n}_SI_omega", w)

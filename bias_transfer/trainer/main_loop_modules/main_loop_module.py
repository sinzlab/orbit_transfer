class MainLoopModule(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.train_loader = trainer.data_loaders["train"]
        self.config = trainer.config
        self.device = trainer.device
        self.seed = trainer.seed
        self.tracker = self.trainer.tracker
        self.criterion = None
        self.mode = None
        self.train_mode = False
        self.task_key = ""
        self.epoch = -1
        self.options = {}

    def pre_epoch(self, model, mode, **options):
        self.mode = mode
        self.train_mode = mode == "Training"
        self.epoch = self.tracker.epoch
        self.options = options

    def pre_forward(self, model, inputs, task_key, shared_memory):
        self.task_key = task_key
        return model, inputs

    def post_forward(self, outputs, loss, targets, **shared_memory):
        return outputs, loss, targets

    def post_backward(self, model):
        pass

    def post_optimizer(self, model):
        pass

    def post_epoch(self, model):
        pass

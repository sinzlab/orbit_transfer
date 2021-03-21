from nntransfer.configs.trainer.base import TrainerConfig


class Classification(TrainerConfig):
    fn = "bias_transfer.trainer.img_classification"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.maximize: bool = True  # if stop_function maximized or minimized
        self.eval_with_bn_train: bool = False

        super(Classification, self).__init__(**kwargs)



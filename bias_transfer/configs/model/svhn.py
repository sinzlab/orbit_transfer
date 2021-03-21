from bias_transfer.configs.model.classification import ClassificationModel


class SVHNModel(ClassificationModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.input_size: int = 32
        self.num_classes: int = 10
        self.input_channels: int = 3
        super().__init__(**kwargs)

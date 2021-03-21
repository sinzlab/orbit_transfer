from bias_transfer.configs.model.classification import ClassificationModel


class CIFAR100Model(ClassificationModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.input_channels: int = 3
        self.input_size: int = 32
        self.num_classes: int = 100
        super().__init__(**kwargs)


class CIFAR10Model(ClassificationModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.input_channels: int = 3
        self.input_size: int = 32
        self.num_classes: int = 10
        super().__init__(**kwargs)

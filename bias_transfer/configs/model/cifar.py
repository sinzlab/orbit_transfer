from bias_transfer.configs.model.classification import Classification


class CIFAR100(Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.input_size: int = 32
        self.num_classes: int = 100
        super().__init__(**kwargs)


class CIFAR10(Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.input_size: int = 32
        self.num_classes: int = 10
        super().__init__(**kwargs)

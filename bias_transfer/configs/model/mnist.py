from bias_transfer.configs.model.classification import ClassificationModel


class MNISTModel(ClassificationModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "lenet5"
        self.num_classes: int = 10
        self.input_size: int = 28
        self.input_channels: int = 1
        self.comment = f"MNIST {self.type}"
        super().__init__(**kwargs)


class MNISTIBModel(MNISTModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.bias: str = "clean"
        self.type: str = "lenet300-100" if self.bias == "translation" else "lenet5"
        self.num_classes: int = 1 if "regression" in self.bias else 10
        self.input_size: int = 80 if self.bias == "addition" else 40
        self.input_channels: int = 3 if "color" in self.bias else 1
        self.comment = f"MNIST-IB {self.bias} {self.type}"
        super().__init__(**kwargs)

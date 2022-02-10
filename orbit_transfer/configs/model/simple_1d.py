from .classification import ClassificationModel


class MNIST1DModelConfig(ClassificationModel):
    fn = "orbit_transfer.models.mnist_1d.model_fn"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "fc_single"
        self.num_classes: int = 10
        self.input_size: int = 40
        self.hidden_dim: int = 400
        self.channels = 15
        self.stride = 1
        self.padding = 2
        self.kernel_size = 5
        self.group_size = 40
        super().__init__(**kwargs)

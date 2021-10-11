from bias_transfer.configs.model.classification import ClassificationModel


class MNISTModel(ClassificationModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "lenet5"
        self.num_classes: int = 10
        self.input_width: int = 28
        self.input_height: int = 28
        self.input_channels: int = 1
        self.comment = f"MNIST {self.type}"
        super().__init__(**kwargs)


class MNISTTransferModel(MNISTModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.bias: str = "clean"
        self.type: str = "lenet300-100" if self.bias == "translation" else "lenet5"
        self.num_classes: int = 1 if "regression" in self.bias else 10
        self.input_width: int = 40
        self.input_height: int = 40
        self.input_channels: int = 3 if "color" in self.bias else 1
        self.comment = f"MNIST-Transfer {self.bias} {self.type}"
        super().__init__(**kwargs)


class CNNModel(ClassificationModel):
    """
    vcnn_mnist7 = {
                   'fc_dim': [],
                   'depth': 3,
                   'proj_depth': 0,
                   'filters': [128, 64, 64],
                   'maxout_size': [128, 64, 16],
                   'kernel_size': [(3,3), (3,3), (3,3)],
                   'pool_size': [(1,1), (2,2), (2,2)],
                   'hidden_dropout_rate': 0.3,
                   'input_dropout_rate': 0.0}
    """

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "cnn"
        self.output_size: int = 10
        self.num_layers: int = 4
        self.channels: list = [1, 128, 64, 64]
        self.kernel_size: list = [3, 3, 3]
        self.pool_size: list = [1, 2, 2]
        self.max_out: list = [1, 1, 4]
        self.activation: str = "relu"
        self.dropout: float = 0.3
        self.batch_norm: bool = True
        self.comment = f"MNIST {self.type}"
        super().__init__(**kwargs)


class MLPModel(ClassificationModel):
    """
    ff_mnist4 = {'hidden_dim': [512, 128, 32],
                       'depth': 3,
                       'hidden_dropout_rate': 0.1,
                       'input_dropout_rate': 0.0}
    """

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "mlp"
        self.output_size: int = 10
        self.num_layers: int = 4
        self.layer_size: list = [28 * 28, 512, 128, 32]
        self.activation: str = "relu"
        self.dropout: float = 0.1
        self.batch_norm: bool = True
        self.comment = f"MNIST {self.type}"
        super().__init__(**kwargs)

class TransferModel(ClassificationModel):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "equiv_transfer"
        self.output_size: int = 10
        self.num_layers: int = 4
        self.kernel_size: int = 5
        self.group_size: int = 5
        self.comment = f"MNIST {self.type}"
        super().__init__(**kwargs)

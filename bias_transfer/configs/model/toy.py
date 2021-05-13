from nntransfer.configs import ModelConfig
from . import ClassificationModel


class ToyModel(ClassificationModel):
    fn = "bias_transfer.models.classification_model_builder"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "linear"
        self.num_classes: int = 1
        self.input_size: int = 2
        super().__init__(**kwargs)

class ToySineModel(ModelConfig):
    fn = "bias_transfer.models.classification_model_builder"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "mlp"
        self.input_size: int = 1
        self.output_size: int = 1
        self.layer_size: int = 40
        self.num_layers: int = 4
        self.activation: str = "relu"
        super().__init__(**kwargs)

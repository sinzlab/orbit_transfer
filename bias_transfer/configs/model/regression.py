from .base import ModelConfig


class Regression(ModelConfig):
    fn = "bias_transfer.models.regression_model_builder"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "fc"
        self.input_size: int = 1
        self.output_size: int = 1
        self.layer_size: int = 100
        self.num_layers: int = 4
        self.activation: str = "sigmoid"
        super().__init__(**kwargs)

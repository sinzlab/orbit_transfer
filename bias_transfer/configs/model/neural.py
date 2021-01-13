from .base import ModelConfig


class Neural(ModelConfig):
    fn = "bias_transfer.models.neural_cnn_builder"

    @baseline
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.readout_type = kwargs.pop("readout_type", "point")
        if self.readout_type == "point":
            self.hidden_dilation = kwargs.pop("hidden_dilation", 2)
            self.se_reduction = kwargs.pop("se_reduction", 16)
        self.input_kern = kwargs.pop("input_kern", 24)
        self.hidden_kern = kwargs.pop("hidden_kern", 9)
        self.depth_separable = kwargs.pop("depth_separable", True)
        self.stack = kwargs.pop("stack", -1)
        self.n_se_blocks = kwargs.pop("n_se_blocks", 2)
        self.gamma_readout = kwargs.pop("gamma_readout", 0.5)
        self.gamma_input = kwargs.pop("gamma_input", 10)
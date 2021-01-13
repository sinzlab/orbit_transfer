from .base import ModelConfig


class MTL(ModelConfig):
    fn = "bias_transfer.models.mtl_builder"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.vgg_type = kwargs.pop("vgg_type", "vgg19_bn")
        self.classification = kwargs.pop("classification", False)
        self.classification_readout_type = kwargs.pop(
            "classification_readout_type", None
        )
        self.input_size = kwargs.pop("input_size", None)
        self.num_classes = kwargs.pop("num_classes", 200)
        self.pretrained = kwargs.pop("pretrained", True)

        self.v1_model_layer = kwargs.pop("v1_model_layer", 17)
        self.neural_input_channels = kwargs.pop("neural_input_channels", 1)
        self.v1_fine_tune = kwargs.pop("v1_fine_tune", False)
        self.v1_init_mu_range = kwargs.pop("v1_init_mu_range", 0.3)
        self.v1_init_sigma_range = kwargs.pop("v1_init_sigma_range", 0.6)
        self.v1_readout_bias = kwargs.pop("v1_readout_bias", True)
        self.v1_bias = kwargs.pop("v1_bias", True)
        self.v1_final_batchnorm = kwargs.pop("v1_final_batchnorm", False)
        self.v1_gamma_readout = kwargs.pop("v1_gamma_readout", 0.5)
        self.v1_elu_offset = kwargs.pop("v1_elu_offset", -1)
        self.classification_input_channels = kwargs.pop(
            "classification_input_channels", 1
        )
        super().__init__(**kwargs)

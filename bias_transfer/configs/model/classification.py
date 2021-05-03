from typing import Dict, Tuple

from nntransfer.configs.model.base import ModelConfig


class ClassificationModel(ModelConfig):
    fn = "bias_transfer.models.classification_model_builder"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "resnet50"
        self.core_type: str = "conv"
        self.conv_stem_kernel_size: int = 3
        self.conv_stem_padding: int = 1
        self.conv_stem_stride: int = 1
        self.core_stride: int = 1
        self.max_pool_after_stem: bool = False
        self.advanced_init: bool = False
        self.zero_init_residual: bool = False
        self.adaptive_pooling: bool = False
        self.avg_pool: bool = False

        # resnet specific
        self.noise_adv_classification: bool = False
        self.noise_adv_regression: bool = False
        self.num_noise_readout_layers: int = 1
        self.noise_sigmoid_output: bool = self.noise_adv_classification
        # vgg specific
        self.pretrained: bool = False
        self.pretrained_path: str = ""
        self.readout_type: str = "dense"
        self.add_buffer: Tuple = ()
        super().__init__(**kwargs)

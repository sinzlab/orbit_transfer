from bias_transfer.configs.model.classification import Classification


class ImageNet(Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.num_classes: int = 1000
        self.input_size: int = 224
        self.input_channels: int = 3
        self.conv_stem_kernel_size: int = 7
        self.conv_stem_padding: int = 3
        self.conv_stem_stride: int = 2
        self.max_pool_after_stem: bool = True
        self.advanced_init: bool = True
        self.zero_init_residual: bool = True
        self.adaptive_pooling: bool = True
        self.avg_pool: bool = True
        super().__init__(**kwargs)


class TinyImageNet(Classification):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.num_classes = 200
        self.input_size = 64
        self.input_channels: int = 3
        self.core_stride = 2
        self.conv_stem_kernel_size = 5
        super().__init__(**kwargs)

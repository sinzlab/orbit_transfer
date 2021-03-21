from nntransfer.configs.dataset.image import ImageDatasetConfig


class MNIST_IB(ImageDatasetConfig):
    fn = "bias_transfer.dataset.mnist_ib_dataset_loader"

    data_mean_defaults = {
        "MNIST_color": (0.03685451, 0.0367535, 0.03952756),
        "MNIST_color_easy": (0.03685451, 0.0367535, 0.03952756),
        "MNIST_noise": (0.13405791,),
        "MNIST_rotation": (0.0640235,),
        "MNIST_translation": (0.06402363,),
        "MNIST_addition": (0.06402363,),
        "MNIST_clean": (0.06402363,),
        "MNIST_clean_shuffle": (0.06402363,),
        "FashionMNIST_color": (0.08239705, 0.09176614, 0.0904255,),
        "FashionMNIST_color_shuffle": (0.08239705, 0.09176614, 0.0904255,),
        "FashionMNIST_color_easy": (0.08239705, 0.09176614, 0.0904255,),
        "FashionMNIST_noise": (0.19938468,),
        "FashionMNIST_rotation": (0.14016011,),
        "FashionMNIST_rotation_regression": (0.14016011,),
        "FashionMNIST_translation": (0.1401599,),
        "FashionMNIST_addition": (0.1401599,),
        "FashionMNIST_clean": (0.1401599,),
        "FashionMNIST_clean_shuffle": (0.1401599,),
    }
    data_std_defaults = {
        "MNIST_color": (0.17386045, 0.16883257, 0.1768625),
        "MNIST_color_easy": (0.17386045, 0.16883257, 0.1768625),
        "MNIST_noise": (0.22387815,),
        "MNIST_rotation": (0.0640235,),
        "MNIST_translation": (0.22534915,),
        "MNIST_addition": (0.22534915,),
        "MNIST_clean": (0.22534915,),
        "MNIST_clean_shuffle": (0.22534915,),
        "FashionMNIST_color": (0.25112887, 0.26145387, 0.26009334,),
        "FashionMNIST_color_shuffle": (0.25112887, 0.26145387, 0.26009334,),
        "FashionMNIST_color_easy": (0.25112887, 0.26145387, 0.26009334,),
        "FashionMNIST_noise": (0.28845804,),
        "FashionMNIST_rotation": (0.28369352,),
        "FashionMNIST_rotation_regression": (0.28369352,),
        "FashionMNIST_translation": (0.28550556,),
        "FashionMNIST_addition": (0.28550556,),
        "FashionMNIST_clean": (0.28550556,),
        "FashionMNIST_clean_shuffle": (0.28550556,),
    }
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls = "MNIST-IB"
        self.input_size: int = 40 if self.bias != "addition" else 80
        self.convert_to_rgb: bool = False
        self.bias: str = "clean"
        self.dataset_sub_cls: str = "FashionMNIST"  # could also be MNIST
        self.apply_data_normalization: bool = False
        self.apply_data_augmentation: bool = False
        self.add_corrupted_test: bool = False
        super().__init__(**kwargs)

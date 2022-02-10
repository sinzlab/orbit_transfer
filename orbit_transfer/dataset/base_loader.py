import torchvision.transforms as transforms
from nntransfer.dataset.img_dataset_loader import (
    ImageDatasetLoader as BaseImageLoader,
)


class ImageDatasetLoader(BaseImageLoader):
    def get_transforms(self, config):
        transform_train = [
            transforms.RandomCrop((config.input_height, config.input_width), padding=4)
            if config.apply_augmentation
            else None,
            transforms.RandomHorizontalFlip() if config.apply_augmentation else None,
            transforms.RandomRotation(15)
            if config.apply_augmentation and not "MNIST" in config.dataset_cls
            else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            if config.convert_to_rgb
            else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_val = [
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            if config.convert_to_rgb
            else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_test = [
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            if config.convert_to_rgb
            else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_test = transforms.Compose(
            list(filter(lambda x: x is not None, transform_test))
        )
        transform_val = transforms.Compose(
            list(filter(lambda x: x is not None, transform_val))
        )
        transform_train = transforms.Compose(
            list(filter(lambda x: x is not None, transform_train))
        )
        return transform_test, transform_train, transform_val

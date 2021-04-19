import torchvision
import torchvision.transforms as transforms

from bias_transfer.dataset.base_loader import ImageDatasetLoader
from nntransfer.dataset.utils import get_dataset


class TorchvisionDatasetLoader(ImageDatasetLoader):
    def get_transforms(self, config):
        transform_train = [
            transforms.ToPILImage()
            if config.dataset_cls == "CIFAR10-Semisupervised"
            or config.dataset_cls == "MNIST-Transfer"
            else None,
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
            transforms.ToPILImage()
            if config.dataset_cls == "CIFAR10-Semisupervised"
            or config.dataset_cls == "MNIST-Transfer"
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
        transform_test = [
            transforms.ToPILImage() if config.dataset_cls == "MNIST-Transfer" else None,
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

    def get_datasets(self, config, transform_test, transform_train, transform_val):
        if (
            config.dataset_cls in list(torchvision.datasets.__dict__.keys())
            and config.dataset_cls != "ImageNet"
        ):
            if config.dataset_cls == "MNIST":
                #download MNIST
                get_dataset(
                    "http://www.di.ens.fr/~lelarge/MNIST.tar.gz",
                    config.data_dir,
                    dataset_cls=config.dataset_cls,
                )

            dataset_cls = eval("torchvision.datasets." + config.dataset_cls)
            kwargs = {
                "root": config.data_dir,
                "transform": transform_train,
                "download": True,
            }

            if config.dataset_cls == "SVHN":
                kwargs["split"] = "train"
            else:
                kwargs["train"] = True
            train_dataset = dataset_cls(**kwargs)

            kwargs["transform"] = transform_val
            valid_dataset = dataset_cls(**kwargs)

            kwargs["transform"] = transform_test
            if config.dataset_cls == "SVHN":
                kwargs["split"] = "test"
            else:
                kwargs["train"] = False
            test_dataset = dataset_cls(**kwargs)
        else:
            raise KeyError()

        st_test_dataset = self.add_stylized_test(config, transform_test)
        c_test_datasets = self.add_corrupted_test(config, transform_test)

        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            c_test_datasets,
            st_test_dataset,
        )


torchvision_dataset_loader = TorchvisionDatasetLoader()

import os
import numpy as np
from . import generate_and_save


def main(dataset="FashionMNIST"):
    for bias in [
        # "clean",
        # "color",
        # "color_shuffle",
        # "translation",
        # "rotation",
        # "rotation_regression",
        # "noise",
        # "addition_regression",
        # "scale",
        # "addition_regression_noise",
        "translation_negative",
        "translation_positive",
    ]:
        generate_and_save(
            bias, base_path="./data/image_classification/torchvision/", dataset=dataset
        )
        train_tensor = np.load(
            os.path.join(
                f"./data/image_classification/torchvision/{dataset}-Transfer",
                f"{bias}_train_source.npy",
            )
        )
        mean = np.mean(train_tensor, axis=(0, 2, 3))
        std = np.std(train_tensor, axis=(0, 2, 3))
        print(f"Saved {dataset}-{bias} with mean {mean} and std {std}")


if __name__ == "__main__":
    main()

import os
import numpy as np
from . import generate_and_save


def main():
    for bias in ["expansion", "color", "translation", "noise", "addition"]:
        generate_and_save(
            bias, base_path="/work/data/image_classification/torchvision/",
        )
        train_tensor = np.load(
            os.path.join(
                "/work/data/image_classification/torchvision/MNIST-IB",
                f"{bias}_train_source.npy",
            )
        )
        mean = np.mean(train_tensor, axis=(0, 2, 3))
        std = np.std(train_tensor, axis=(0, 2, 3))
        print(f"Saved MNIST-{bias} with mean {mean} and std {std}")


if __name__ == "__main__":
    main()

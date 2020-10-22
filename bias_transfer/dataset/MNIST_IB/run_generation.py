import os
import numpy as np
from . import generate_and_save


def main(dataset="MNIST"):
    for bias in ["expansion", "color", "translation", "rotation", "noise", "addition"]:
        generate_and_save(
            bias, base_path="/work/data/image_classification/torchvision/",dataset=dataset
        )
        train_tensor = np.load(
            os.path.join(
                f"/work/data/image_classification/torchvision/{dataset}-IB",
                f"{bias}_train_source.npy",
            )
        )
        mean = np.mean(train_tensor, axis=(0, 2, 3))
        std = np.std(train_tensor, axis=(0, 2, 3))
        print(f"Saved {dataset}-{bias} with mean {mean} and std {std}")


if __name__ == "__main__":
    main()

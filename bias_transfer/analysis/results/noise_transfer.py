import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

from .base import Analyzer


class NoiseAnalyzer(Analyzer):
    def plot_training_progress(self, dataset="Validation", **kwargs):
        self.plot_progress_line(
            to_plot=((dataset, "img_classification", "accuracy")), **kwargs
        )

    def plot_noise_eval(self, std=True, bn_train=False, **kwargs):
        bn_train = " BN=Train" if bn_train else ""
        if std:
            to_plot = [
                (
                    f"Noise noise_std {p}_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                )
                for p in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
            ]
        else:
            raise NotImplementedError()

        def rename(name):
            number_idx = re.search(r"\d", name)
            name = name[number_idx.start() :]
            underscore_idx = name.find("_")
            name = name[:underscore_idx]
            return float(name)

        self.plot_comparison_line(to_plot, rename=rename, **kwargs)

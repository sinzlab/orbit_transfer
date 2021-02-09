import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

from .base import Analyzer
from ..utils import plot_preparation


class CTestAnalyzer(Analyzer):
    corruption_map = {
        "shot_noise": "Shot Noise",
        "impulse_noise": "Impulse Noise",
        "speckle_noise": "Speckle Noise",
        "gaussian_noise": "Gaussian Noise",
        "defocus_blur": "Defocus Blur",
        "gaussian_blur": "Gauss Blur",
        "motion_blur": "Motion Blur",
        "glass_blur": "Glass Blur",
        "zoom_blur": "Zoom Blur",
        "brightness": "Brightness",
        "fog": "Fog",
        "frost": "Frost",
        "snow": "Snow",
        "contrast": "Contrast",
        "elastic_transform": "Elastic Transform",
        "pixelate": "Pixelate",
        "jpeg_compression": "JPEG Compression",
        "saturate": "Saturate",
        "spatter": "Spatter",
    }

    Res_Alex_Net_mean = dict()
    Res_Alex_Net_mean["Gaussian Noise"] = 0.886
    Res_Alex_Net_mean["Shot Noise"] = 0.894
    Res_Alex_Net_mean["Impulse Noise"] = 0.923
    Res_Alex_Net_mean["Defocus Blur"] = 0.820
    Res_Alex_Net_mean["Gauss Blur"] = 0.826
    Res_Alex_Net_mean["Glass Blur"] = 0.826
    Res_Alex_Net_mean["Motion Blur"] = 0.786
    Res_Alex_Net_mean["Zoom Blur"] = 0.798
    Res_Alex_Net_mean["Snow"] = 0.867
    Res_Alex_Net_mean["Frost"] = 0.827
    Res_Alex_Net_mean["Fog"] = 0.819
    Res_Alex_Net_mean["Brightness"] = 0.565
    Res_Alex_Net_mean["Contrast"] = 0.853
    Res_Alex_Net_mean["Elastic Transform"] = 0.646
    Res_Alex_Net_mean["Pixelate"] = 0.718
    Res_Alex_Net_mean["JPEG Compression"] = 0.607
    Res_Alex_Net_mean["Speckle Noise"] = 0.845
    Res_Alex_Net_mean["Spatter"] = 0.718
    Res_Alex_Net_mean["Saturate"] = 0.658

    def extract_c_test_results(self):
        corruptions = (
            "shot_noise",
            "impulse_noise",
            # "speckle_noise",
            "gaussian_noise",
            "defocus_blur",
            # "gaussian_blur",
            "motion_blur",
            "glass_blur",
            "zoom_blur",
            "brightness",
            "fog",
            "frost",
            "snow",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            # "saturate",
            # "spatter",
        )
        data_to_plot = pd.DataFrame()
        for corruption in corruptions:
            row_list = []
            for desc, tracker in self.data.items():
                row = {
                    severity: tracker.get_current_objective(
                        corruption, str(severity), "accuracy"
                    )
                    for severity in range(1, 6)
                }
                row[0] = tracker.get_current_objective(
                    "Test", "img_classification", "accuracy"
                )
                row["name"] = desc.name
                row_list.append(row)
            df = pd.DataFrame(row_list)
            df = df.groupby("name").mean()
            df["Corruption"] = corruption
            data_to_plot = pd.concat([data_to_plot, df], axis=0, sort=True)
        return data_to_plot

    def calculate_c_scores(self):
        c_data = self.extract_c_test_results()
        df = c_data[c_data.columns[0:6]].apply(lambda x: 100 - x)
        df_mean = df[df.columns[0:6]].mean(axis=1)
        c_data = pd.concat([c_data, df_mean], axis=1)
        c_data.columns = [1, 2, 3, 4, 5, 0, "Corruption", "Mean"]

        def normalize_alexnet(row):
            mean_error = row["Mean"]
            corruption = row["Corruption"]
            ce = mean_error / self.Res_Alex_Net_mean[self.corruption_map[corruption]]
            return pd.concat([row, pd.Series({"mCE": ce})])

        c_data = c_data.apply(normalize_alexnet, axis=1)
        c_data = c_data.groupby("name").mean()
        return c_data

    def plot_grid(self, style, **kwargs):
        fig, ax = plot_preparation(style)
        data_to_plot = self.extract_c_test_results()
        g = sns.FacetGrid(
            data=data_to_plot,
            col="Corruption",
            col_wrap=4,
            sharey=True,
            sharex=True,
            # height=4
        )

        def draw_heatmap(data, *args, **kwargs):
            del data["Corruption"]
            # print(data)
            sns.heatmap(data, annot=True, cbar=False)

        g.map_dataframe(draw_heatmap)
        fig = g.fig
        sns.despine(offset=10, trim=True)
        # remove ticks again (see: https://stackoverflow.com/questions/37860163/seaborn-despine-brings-back-the-ytick-labels)
        # loop over the non-left axes:
        for i, ax in enumerate(g.axes.flat):
            if i % 4 != 0:
                # get the yticklabels from the axis and set visibility to False
                for label in ax.get_yticklabels():
                    label.set_visible(False)
                ax.yaxis.offsetText.set_visible(False)
            if i < len(g.axes) - 4:
                # get the xticklabels from the axis and set visibility to False
                for label in ax.get_xticklabels():
                    label.set_visible(False)
                ax.xaxis.offsetText.set_visible(False)
        self._post_plot_operations(style, **kwargs)


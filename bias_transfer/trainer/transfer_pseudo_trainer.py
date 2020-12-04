from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from bias_transfer.configs.trainer import TransferTrainerConfig
from bias_transfer.dataset.dataset_classes.npy_dataset import NpyDataset
from bias_transfer.trainer.img_classification_trainer import ImgClassificationTrainer
from bias_transfer.trainer.regression_trainer import RegressionTrainer


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = TransferPseudoTrainer(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()


class TransferPseudoTrainer(ImgClassificationTrainer):
    def __init__(self, dataloaders, model, seed, uid, cb, **kwargs):
        super().__init__(dataloaders, model, seed, uid, cb, **kwargs)
        self.config = TransferTrainerConfig.from_dict(kwargs)

    def train(self):
        self.tracker.start_epoch()
        if hasattr(tqdm, "_instances"):
            tqdm._instances.clear()
        train = self.generate_rep_dataset(data="train")
        return train, self.model.state_dict()

    def generate_rep_dataset(self, data):
        for inputs, targets in self.data_loaders[data]["img_classification"]:
            batch = inputs.cpu().numpy().transpose(0, 2, 3, 1)
            n_rows = 2
            n_cols = 5
            fig, axs = plt.subplots(n_rows, n_cols)
            if n_rows == 1:
                axs = [axs]
            for r in range(n_rows):
                for c in range(n_cols):
                    axs[r][c].imshow(batch[r * n_cols + c].squeeze())
                    axs[r][c].set_title((targets[r * n_cols + c]).item())
                    axs[r][c].set_axis_off()
            break

        _, collected_outputs = self.main_loop(
            data_loader=self.data_loaders[data],
            epoch=0,
            mode="Validation",
            return_outputs=True,
        )
        outputs = {}
        for rep_name in collected_outputs[0].keys():
            outputs[rep_name] = torch.cat(
                [batch_output[rep_name] for batch_output in collected_outputs]
            ).numpy()
        if self.config.save_input:
            collected_inputs = []
            for src, _ in self.data_loaders[data]["img_classification"]:
                collected_inputs.append(src)
            outputs["source"] = torch.cat(collected_inputs).numpy()


        batch = outputs["source"][:128].transpose(0, 2, 3, 1)
        targets = outputs["fc3"][:128]
        n_rows = 2
        n_cols = 5
        fig, axs = plt.subplots(n_rows, n_cols)
        if n_rows == 1:
            axs = [axs]
        for r in range(n_rows):
            for c in range(n_cols):
                axs[r][c].imshow(batch[r * n_cols + c].squeeze())
                axs[r][c].set_title((targets[r * n_cols + c]).argmax().item())
                axs[r][c].set_axis_off()
        return outputs





def regression_trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = TransferPseudoTrainerRegression(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()


class TransferPseudoTrainerRegression(RegressionTrainer):
    def __init__(self, dataloaders, model, seed, uid, cb, **kwargs):
        super().__init__(dataloaders, model, seed, uid, cb, **kwargs)
        self.config = TransferTrainerConfig.from_dict(kwargs)

    def train(self):
        self.tracker.start_epoch()
        if hasattr(tqdm, "_instances"):
            tqdm._instances.clear()
        train = self.generate_rep_dataset(data="train")
        return train, self.model.state_dict()

    def generate_rep_dataset(self, data):
        _, collected_outputs = self.main_loop(
            data_loader=self.data_loaders[data],
            epoch=0,
            mode="Validation",
            return_outputs=True,
        )
        outputs = {}
        for rep_name in collected_outputs[0].keys():
            outputs[rep_name] = torch.cat(
                [batch_output[rep_name] for batch_output in collected_outputs]
            ).numpy()
        if self.config.save_input:
            collected_inputs = []
            for src, _ in self.data_loaders[data]["regression"]:
                collected_inputs.append(src)
            outputs["source"] = torch.cat(collected_inputs).numpy()
        return outputs

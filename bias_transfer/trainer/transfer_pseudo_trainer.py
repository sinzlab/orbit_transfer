from tqdm import tqdm
import torch
import numpy as np

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

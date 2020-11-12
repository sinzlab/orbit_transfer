from tqdm import tqdm
import torch
import numpy as np
from bias_transfer.trainer.img_classification_trainer import ImgClassificationTrainer


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = TransferPseudoTrainer(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()


class TransferPseudoTrainer(ImgClassificationTrainer):
    def train(self):
        # train over epochs
        self.tracker.start_epoch()
        if hasattr(tqdm, "_instances"):
            tqdm._instances.clear()
        train = self.generate_rep_dataset(
            data="train", rep_name=self.config.rdm_generation
        )
        test = self.generate_rep_dataset(
            data="test", rep_name=self.config.rdm_generation
        )
        return train, test, self.model.state_dict()

    def generate_rep_dataset(self, data, rep_name):
        _, collected_outputs = self.main_loop(
            data_loader=self.data_loaders[data],
            epoch=0,
            mode="Validation",
            return_outputs=True,
        )
        outputs = [o[rep_name] for o in collected_outputs]
        outputs = torch.cat(outputs).detach().to("cpu").numpy()
        return outputs

        # rep_dataset = TensorDataset(torch.cat(outputs).to("cpu"))
        # orig_dataset = data_loader.dataset
        # combined_dataset = CombinedDataset(
        #     JoinedDataset(
        #         sample_datasets=[orig_dataset],
        #         target_datasets=[orig_dataset, rep_dataset],
        #     )
        # )
        # combined_data_loader = torch.utils.data.DataLoader(
        #     dataset=combined_dataset,
        #     batch_size=data_loader.batch_size,
        #     sampler=data_loader.sampler,
        #     num_workers=data_loader.num_workers,
        #     pin_memory=data_loader.pin_memory,
        #     shuffle=False,
        # )
        # return {key: combined_data_loader}

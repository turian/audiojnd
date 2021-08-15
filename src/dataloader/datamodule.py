import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from src.dataloader.dataset_utils import process_files
from src.dataloader.dataset import PairedDatset


from typing import Optional


class AnnotationsDataModule(pl.LightningDataModule):
    # batch_size = 1 because we have two different audio lengths :\
    # Otherwise we could try writing our own collate_fn
    # There might also be a way to interleave batches from two datasets
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        df = process_files(cfg)
        self.train_data = df[df.kfold != self.cfg.datamodule.fold]
        self.valid_data = df[df.kfold == self.cfg.datamodule.fold]
        self.train_batch_size = self.cfg.datamodule.train_batch_size
        self.val_batch_size = self.cfg.datamodule.val_batch_size
        self.num_workers = self.cfg.datamodule.num_workers
        self.pin_memory = self.cfg.datamodule.pin_memory

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = PairedDatset(self.cfg, self.train_data)
        self.val_dataset = PairedDatset(self.cfg, self.valid_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

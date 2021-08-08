#!/usr/bin/env python3
"""
Given training annotations, retrain the model.
Evaluate using 5-fold random split.
"""

import csv
import glob
import os.path
import re
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from preprocess import LENGTH_SAMPLES, ensure_length
from transforms import CONFIG, pydubread

FOLDS = 5
LENGTHRE = re.compile(".*\.wav-([0-9\.]+)\..*")


class PairedDatset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, idx):
        oldf, newf, y = self.rows[idx]
        print(oldf, newf)
        import sys

        sys.stdout.flush()

        # This is copied from modelscores.py,
        # maybe make this a util
        if LENGTHRE.match(oldf):
            length = float(LENGTHRE.match(oldf).group(1))
            nsamples = dict(LENGTH_SAMPLES)[length]
        else:
            assert False

        oldx = ensure_length(pydubread(oldf), nsamples, from_start=True)
        newx = ensure_length(pydubread(newf), nsamples, from_start=True)
        assert oldx.shape == newx.shape, f"{oldx.shape} != {newx.shape}"
        assert oldx.shape == (nsamples,)

        print(oldx.shape, newx.shape)
        sys.stdout.flush()
        return oldx, newx, y

    def __len__(self):
        return len(self.rows)


class AnnotationsDataModule(pl.LightningDataModule):
    # batch_size = 1 because we have two different audio lengths :\
    # Otherwise we could try writing our own collate_fn
    # There might also be a way to interleave batches from two datasets
    def __init__(self, data_dir: str = "data/iterations", batch_size: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.rows = []
        print(os.path.join(self.data_dir, "*/annotation*csv"))
        for csvfile in glob.glob(os.path.join(self.data_dir, "*/annotation*csv")):
            print(csvfile)
            for infile, transfile, y in csv.reader(open(csvfile)):
                y = int(y)
                if infile == transfile:
                    continue
                # Only use dev, not eval
                if "FSD50K.eval_audio" in infile:
                    continue
                self.rows.append((infile, transfile, y))
        print(len(self.rows))
        self.dataset = PairedDatset(self.rows)
        num, div = len(self.rows), FOLDS
        fold_idx = [num // div + (1 if x < num % div else 0) for x in range(div)]
        self.folds = random_split(
            self.dataset, fold_idx, generator=torch.Generator().manual_seed(42)
        )
        # TODO: This dataloader won't allow kfold

    def train_dataloader(self):
        return DataLoader(ConcatDataset(self.folds[1:]), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.folds[0], batch_size=self.batch_size)


#    def test_dataloader(self):
#        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class VisionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # pretrained?
        self.vision = torchvision.models.resnet101()

    def forward(self, x):
        # Probably want to do smart Mel initialization
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x1, x2, y = batch
        print(x1.shape)
        print(x2.shape)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def retrain():
    annotations_data_module = AnnotationsDataModule()

    model = VisionModel()

    trainer = pl.Trainer()
    trainer.fit(model, annotations_data_module)


if __name__ == "__main__":
    retrain()

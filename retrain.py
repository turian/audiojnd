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
import torch.nn.functional as F
import torch.nn as nn
import torchopenl3
from torch.utils.data import DataLoader, Dataset, random_split
from torchopenl3.utils import preprocess_audio_batch

from preprocess import LENGTH_SAMPLES, ensure_length
from transforms import CONFIG, pydubread

from sklearn.model_selection import StratifiedKFold
import pandas as pd

FOLDS = 5
LENGTHRE = re.compile(".*\.wav-([0-9\.]+)\..*")


class PairedDatset(Dataset):
    def __init__(self, df):
        self.rows = df[["infile", "transfile", "y"]].values

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

        oldX = preprocess_audio_batch(
            torch.tensor(oldx).unsqueeze(0), CONFIG["SAMPLE_RATE"]
        ).to(torch.float32)
        newX = preprocess_audio_batch(
            torch.tensor(newx).unsqueeze(0), CONFIG["SAMPLE_RATE"]
        ).to(torch.float32)

        return oldX, newX, y

    def __len__(self):
        return len(self.rows)


class AnnotationsDataModule(pl.LightningDataModule):
    # batch_size = 1 because we have two different audio lengths :\
    # Otherwise we could try writing our own collate_fn
    # There might also be a way to interleave batches from two datasets
    def __init__(
        self, data_dir: str = "data/iterations/", fold: int = 0, batch_size: int = 1
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fold = fold

    def setup(self, stage: Optional[str] = None):
        self.rows = []
        print(os.path.join(self.data_dir, "*/annotation*csv"))
        for csvfile in glob.glob(os.path.join(self.data_dir, "*/annotation*csv")):
            print(csvfile)
            for infile, transfile, y in csv.reader(open(csvfile)):
                y = float(y)
                if infile == transfile:
                    continue
                # Only use dev, not eval
                if "FSD50K.eval_audio" in infile:
                    continue
                self.rows.append([infile, transfile, y])

        self.df = pd.DataFrame(self.rows, columns=["infile", "transfile", "y"])
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        skf = StratifiedKFold(n_splits=FOLDS)
        for i, (trn_, val_) in enumerate(skf.split(self.df, self.df["y"])):
            self.df.loc[val_, "kfold"] = i

        self.train_dataset = PairedDatset(self.df[self.df.kfold != self.fold])
        self.val_dataset = PairedDatset(self.df[self.df.kfold != self.fold])

    # TODO: num_workers, etc.
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


#    def test_dataloader(self):
#        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value] * 6144))

    def forward(self, input):
        return input * self.scale


class AudioJNDModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchopenl3.models.load_audio_embedding_model(
            input_repr="mel256", content_type="env", embedding_size=6144
        )
        self.scale = ScaleLayer()
        # Could also try l1 with crossentropy
        self.cos = nn.CosineSimilarity(eps=1e-6)

    def forward(self, x1, x2):
        bs, _, in2, in3 = x1.size()

        print(x1.shape, x2.shape)

        # TODO: Also try with gradient?
        with torch.no_grad():
            x1 = self.model(x1.view(-1, in2, in3))
            x2 = self.model(x2.view(-1, in2, in3))

        print(x1.shape, x2.shape)

        x1 = self.scale(x1).view(bs, -1)
        x2 = self.scale(x2).view(bs, -1)

        print(x1.shape, x2.shape)

        prob = self.cos(x1, x2)
        assert torch.all((prob < 1) & (prob > -1))

        prob = (prob + 1) / 2
        assert torch.all((prob < 1) & (prob > 0))

        return prob

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x1, x2, y = batch
        y = y.float()
        y_hat = self.forward(x1, x2)
        # Could also try cosine embedding loss and bceloss and just
        # simple l1 loss
        loss = F.mse_loss(y_hat, y)
        print(y_hat, y, loss)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # TODO: weight decay?
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def retrain():
    annotations_data_module = AnnotationsDataModule()

    model = AudioJNDModel()

    trainer = pl.Trainer()
    trainer.fit(model, annotations_data_module)


if __name__ == "__main__":
    retrain()

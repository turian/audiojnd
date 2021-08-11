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

import nnAudio.Spectrogram
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopenl3
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, random_split
from torchopenl3.utils import preprocess_audio_batch

from preprocess import LENGTH_SAMPLES, ensure_length
from transforms import CONFIG, pydubread

FOLDS = 5
LENGTHRE = re.compile(".*\.wav-([0-9\.]+)\..*")


def process_files(data_dir: str = "data/iterations/"):
    rows = []
    print(os.path.join(data_dir, "*/annotation*csv"))
    for csvfile in glob.glob(os.path.join(data_dir, "*/annotation*csv")):
        print(csvfile)
        for infile, transfile, y in csv.reader(open(csvfile)):
            y = float(y)
            if infile == transfile:
                continue
            # Only use dev, not eval
            if "FSD50K.eval_audio" in infile:
                continue
            rows.append([infile, transfile, y])

    df = pd.DataFrame(rows, columns=["infile", "transfile", "y"])
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)
    for i, (trn_, val_) in enumerate(skf.split(df, df["y"])):
        df.loc[val_, "kfold"] = i
    return df


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
    def __init__(self, df, fold: int = 0, batch_size: int = 1):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.fold = fold

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = PairedDatset(self.df[self.df.kfold != self.fold])
        self.val_dataset = PairedDatset(self.df[self.df.kfold == self.fold])

    # TODO: num_workers, etc.
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


#    def test_dataloader(self):
#        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class ScaleLayer(nn.Module):
    def __init__(self, embedding_dim, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value] * embedding_dim))

    def forward(self, input):
        return input * self.scale


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


# TODO: Remove clamp?
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class MelEmbedding(nn.Module):
    def __init__(self, embedding_dim=80):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Adapted from hifigan, except that is for 22050 Hz audio
        # so FFT sizes are adjusted
        self.mel = nnAudio.Spectrogram.MelSpectrogram(
            sr=48000,
            n_fft=2229,
            n_mels=self.embedding_dim,
            hop_length=557,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            # trainable_mel=True,
            # trainable_STFT=True,
            trainable_mel=False,
            trainable_STFT=False,
        )

    def forward(self, x):
        bs, channels, samples = x.size()
        assert channels == 1
        # Now should be bs x nmels x nframes
        x = self.mel(x.view(bs, samples))
        x = spectral_normalize_torch(x)
        # print(x.permute(0, 2, 1).shape)
        return x.permute(0, 2, 1).reshape(-1, self.embedding_dim)


class AudioJNDModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.embedding_dim = 6144
        # self.embedding = torchopenl3.models.load_audio_embedding_model(
        #    input_repr="mel256", content_type="env", embedding_size=6144
        # )
        self.embedding_dim = 80
        self.embedding = MelEmbedding()

        self.scale = ScaleLayer(embedding_dim=self.embedding_dim)
        # Could also try l1 with crossentropy
        self.cos = nn.CosineSimilarity(eps=1e-6)

    def forward(self, x1, x2):
        # batch size x nframes x 1 channel x 48000 samples
        bs, _, in2, in3 = x1.size()
        # print(x1.shape, x2.shape)

        # TODO: Also try with gradient?
        with torch.no_grad():
            # These will now be (batch_size * nframes, embedding_dim)
            x1 = self.embedding(x1.view(-1, in2, in3))
            x2 = self.embedding(x2.view(-1, in2, in3))

        x1 = self.scale(x1).view(bs, -1)
        x2 = self.scale(x2).view(bs, -1)

        # print(x1.shape, x2.shape)

        prob = self.cos(x1, x2)
        assert torch.all((prob < 1) & (prob > -1))

        prob = (prob + 1) / 2
        assert torch.all((prob < 1) & (prob > 0))

        assert prob.shape == (bs,)
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

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        y = y.float()
        y_hat = self.forward(x1, x2)
        loss = F.mse_loss(y_hat, y)

        self.log("val_loss", loss)

        return {"predictions": y_hat, "labels": y}

    def validation_epoch_end(self, outputs):

        preds = []
        labels = []

        for output in outputs:
            preds += output["predictions"]
            labels += output["labels"]

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        labels = labels.detach().cpu()

        preds = preds.detach().cpu()

        try:
            val_auc = roc_auc_score(labels, preds)
        except ValueError:  # if the batch has only one class
            val_auc = 0.0

        self.log("val_auc", val_auc)

    def configure_optimizers(self):
        # TODO: weight decay?
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def retrain():
    df = process_files()
    annotations_data_module = AnnotationsDataModule(df)

    model = AudioJNDModel()
    early_stopping_callback = EarlyStopping(monitor="val_auc", mode="max", patience=1)
    trainer = pl.Trainer(callbacks=[early_stopping_callback])
    trainer.fit(model, annotations_data_module)


if __name__ == "__main__":
    retrain()

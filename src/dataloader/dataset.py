from torch.utils.data import Dataset
import torch
from torchopenl3.utils import preprocess_audio_batch
import re
from omegaconf import DictConfig
import json
from src.dataloader.dataset_utils import ensure_length, pydubread
import pandas as pd


class PairedDatset(Dataset):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame):
        self.rows = df[["infile", "transfile", "y"]].values
        self.lengthre = re.compile(".*\.wav-([0-9\.]+)\..*")
        self.config = json.loads(open(cfg.datamodule.data_config).read())
        self.length_samples = [
            (l, int(round(self.config["SAMPLE_RATE"] * l))) for l in self.config["AUDIO_LENGTHS"]
        ]

    def __getitem__(self, idx):
        oldf, newf, y = self.rows[idx]
        print(oldf, newf)
        import sys

        sys.stdout.flush()

        # This is copied from modelscores.py,
        # maybe make this a util
        if self.lengthre.match(oldf):
            length = float(self.lengthre.match(oldf).group(1))
            nsamples = dict(self.length_samples)[length]
        else:
            assert False

        oldx = ensure_length(
            pydubread(oldf, self.config["SAMPLE_RATE"]), nsamples, from_start=True
        )
        newx = ensure_length(
            pydubread(newf, self.config["SAMPLE_RATE"]), nsamples, from_start=True
        )
        assert oldx.shape == newx.shape, f"{oldx.shape} != {newx.shape}"
        assert oldx.shape == (nsamples,)

        oldX = preprocess_audio_batch(
            torch.tensor(oldx).unsqueeze(0), self.config["SAMPLE_RATE"]
        ).to(torch.float32)
        newX = preprocess_audio_batch(
            torch.tensor(newx).unsqueeze(0), self.config["SAMPLE_RATE"]
        ).to(torch.float32)

        return oldX, newX, y

    def __len__(self):
        return len(self.rows)

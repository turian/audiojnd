from omegaconf import DictConfig
import pandas as pd
import glob
import os
import csv
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import pydub


def process_files(cfg: DictConfig) -> pd.DataFrame:
    rows = []
    print(os.path.join(cfg.datamodule.data_dir, "*/annotation*csv"))
    for csvfile in glob.glob(os.path.join(cfg.datamodule.data_dir, "*/annotation*csv")):
        print(csvfile)
        for infile, transfile, y in csv.reader(open(csvfile)):
            y = float(y)
            if infile == transfile:
                continue
            # Only use dev, not eval
            if "FSD50K.eval_audio" in infile:
                continue
            infile = os.path.join(cfg.work_dir, infile)
            transfile = os.path.join(cfg.work_dir, transfile)
            rows.append([infile, transfile, y])

    df = pd.DataFrame(rows, columns=["infile", "transfile", "y"])
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    skf = StratifiedKFold(n_splits=cfg.datamodule.num_folds, random_state=cfg.seed, shuffle=True)
    for i, (_, val_) in enumerate(skf.split(df, df["y"])):
        df.loc[val_, "kfold"] = i
    return df


def ensure_length(x, length_in_samples, from_start=False):
    if len(x) < length_in_samples:
        npad = length_in_samples - len(x)
        if from_start:
            nstart = 0
        else:
            nstart = random.randint(0, npad)
        x = np.hstack([np.zeros(nstart), x, np.zeros(npad - nstart)])
    elif len(x) > length_in_samples:
        ntrim = len(x) - length_in_samples
        if from_start:
            nstart = 0
        else:
            nstart = random.randint(0, ntrim)
        x = x[nstart : nstart + length_in_samples]
    assert len(x) == length_in_samples
    return x


def pydubread(f, sample_rate):
    """
    MP3 to numpy array.
    We use pydub since soundfile can't read mp3s.
    """
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples(), dtype=np.float32)
    # Convert to float32 from int16
    y /= -32768
    assert a.frame_rate == sample_rate
    return y

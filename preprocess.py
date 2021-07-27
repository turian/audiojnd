#!/usr/bin/env python3
"""
Preprocess the FSD50K corpus.

For each file:
* Resample to mono, target SR
* For each audio length, randomly pad or trim
* Skip any audio that is almost all silence
"""

import glob
import json
import os.path
import random  # We don't seed, we just want something different every time
import subprocess
from typing import NamedTuple

import librosa
import numpy as np
import soundfile as sf
import resampy
from tqdm.auto import tqdm

files = list(glob.glob("data/orig/FSD50K.dev_audio/*wav")) + list(
    glob.glob("data/orig/FSD50K.eval_audio/*wav")
)

CONFIG = json.loads(open("config.json").read())


class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    result = subprocess.run(
        command_array,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return FFProbeResult(
        return_code=result.returncode, json=result.stdout, error=result.stderr
    )


def ensure_length(x, length_in_samples):
    print(length_in_samples)
    if len(x) < length_in_samples:
        npad = length_in_samples - len(x)
        nstart = random.randint(0, npad)
        x = np.hstack([np.zeros(nstart), x, np.zeros(npad - nstart)])
    elif len(x) > length_in_samples:
        ntrim = len(x) - length_in_samples
        nstart = random.randint(0, ntrim)
        x = x[nstart : nstart + length_in_samples]
    assert len(x) == length_in_samples
    return x


LENGTH_SAMPLES = [
    (l, int(round(CONFIG["SAMPLE_RATE"] * l))) for l in CONFIG["AUDIO_LENGTHS"]
]
for f in tqdm(files[:200]):
    newf = f.replace("/orig/", "/preprocessed/")
    newd = os.path.split(newf)[0]
    if not os.path.exists(newd):
        os.makedirs(newd)

    # TODO: Skip files we have already done
    # done = True
    # for length, samples in LENGTH_SAMPLES:

    x, sr = sf.read(f)
    if x.shape == 2:
        print(f"Skipping {f} {x.shape}...")
        continue
    if sr != CONFIG["SAMPLE_RATE"]:
        print(f"Resampling {f}")
        x = resampy.resample(sr, CONFIG["SAMPLE_RATE"])
        sr = CONFIG["SAMPLE_RATE"]

    for length, samples in LENGTH_SAMPLES:
        # Try up to 10 times to find a snippet that is not too silent
        for i in range(10):
            xl = ensure_length(x, samples)
            # Randomly adjust level
            xl *= random.uniform(0.5, 1)
            rms = np.mean(librosa.feature.rms(xl))
            if rms < CONFIG["MIN_RMS"]:
                xl = None
            break
        if xl is not None:
            sf.write(newf + "-%.2f.ogg" % length, xl, CONFIG["SAMPLE_RATE"])

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

import librosa
import numpy as np
import soundfile as sf
import resampy
from tqdm.auto import tqdm

files = list(glob.glob("data/orig/FSD50K.dev_audio/*wav")) + list(
    glob.glob("data/orig/FSD50K.eval_audio/*wav")
)

CONFIG = json.loads(open("config.json").read())

def ensure_length(x, length_in_samples):
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
for f in tqdm(files):
    newf = f.replace("/orig/", "/preprocessed/")
    newd = os.path.split(newf)[0]
    if not os.path.exists(newd):
        os.makedirs(newd)

    # Skip files we have already done
    done = True
    for length, samples in LENGTH_SAMPLES:
        if not os.path.exists(newf + "-%.2f.ogg" % length):
            done = False
            break
    if done:
        continue

    x, sr = sf.read(f)
    if x.shape == 2:
        print(f"Skipping {f} {x.shape}...")
        continue
    # We use 48K since that is OpenL3's SR
    # TODO: Might be faster to use sox+ffmpeg?
    if sr != CONFIG["SAMPLE_RATE"]:
        print(f"Resampling {f}")
        x = resampy.resample(x, sr, CONFIG["SAMPLE_RATE"])
        sr = CONFIG["SAMPLE_RATE"]

    for length, samples in LENGTH_SAMPLES:
        # Try up to 100 times to find a snippet that is not too silent
        for i in range(100):
            xl = ensure_length(x, samples)
            # Normalize audio to max peak
            if np.max(np.abs(xl)) == 0:
                xl = None
            else:
                xl /= np.max(np.abs(xl))
                rms = np.mean(librosa.feature.rms(xl))
                if rms < CONFIG["MIN_RMS"]:
                    xl = None
                else:
                    break
        if xl is not None:
            sf.write(newf + "-%.2f.ogg" % length, xl, CONFIG["SAMPLE_RATE"])

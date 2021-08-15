#!/usr/bin/env python3
"""
Compute the model score for every transformation in the corpus,
and save it to transformation.{model_name}.json
"""

import glob
import json
import os.path
import random
import re

import click
import numpy as np
import torchopenl3
from scipy import spatial
from tqdm.auto import tqdm

from preprocess import LENGTH_SAMPLES, ensure_length
from transforms import CONFIG, pydubread

# Is this the model we want?
model = torchopenl3.models.load_audio_embedding_model(
    input_repr="mel256", content_type="env", embedding_size=6144
)

LENGTHRE = re.compile(".*\.wav-([0-9\.]+)\..*")


@click.command()
@click.argument("model_name")
def process_files(model_name):
    files = list(glob.glob("data/transforms/*/*/*mp3"))

    random.shuffle(files)
    for newf in tqdm(files):
        jsonf = newf + f"{model_name}.json"
        if os.path.exists(jsonf):
            continue
        oldf = newf.replace("transforms/", "preprocessed/")
        oldf = os.path.split(oldf)[0]
        assert os.path.isfile(oldf)

        if LENGTHRE.match(oldf):
            length = float(LENGTHRE.match(oldf).group(1))
            nsamples = dict(LENGTH_SAMPLES)[length]
        else:
            assert False

        oldx = ensure_length(pydubread(oldf), nsamples, from_start=True)
        newx = ensure_length(pydubread(newf), nsamples, from_start=True)
        assert oldx.shape == newx.shape, f"{oldx.shape} != {newx.shape}"
        assert oldx.shape == (nsamples,)

        # TODO: HOP SIZE
        oldemb, ts = torchopenl3.get_audio_embedding(
            oldx, CONFIG["SAMPLE_RATE"], model=model, hop_size=CONFIG["EMBEDDING_HOP_SIZE"],
        )
        newemb, ts = torchopenl3.get_audio_embedding(
            newx, CONFIG["SAMPLE_RATE"], model=model, hop_size=CONFIG["EMBEDDING_HOP_SIZE"],
        )
        assert oldemb.shape == newemb.shape

        # NOTE: We could normalize each frame's embedding first,
        # rather than normalization over the concatenation
        sim = 1 - spatial.distance.cosine(newemb.flatten(), oldemb.flatten())
        open(jsonf, "wt").write(json.dumps([newf, oldf, sim], indent=4))


if __name__ == "__main__":
    process_files()

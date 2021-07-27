#!/usr/bin/env python3

import copy
import glob
import os.path
import random

import auraloss
import scipy
import soundfile as sf
import sox
import torch
from tqdm.auto import tqdm

N_TRANSFORMS = 32

random.seed(0)
dist = auraloss.freq.MultiResolutionSTFTLoss()


def note_to_freq(note):
    a = 440  # frequency of A (common value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


transforms = [
    ("allpass", [("midi", 21, 108), ("width_q", 1, 10)], []),
    (
        "bandpass",
        [("midi", 21, 108), ("width_q", 1, 10)],
        [("constant_skirt", [True, False])],
    ),
    ("bandreject", [("midi", 21, 108), ("width_q", 1, 10)], []),
    ("bass", [("gain_db", -20, 20), ("midi", 21, 108), ("slope", 0.3, 1.0)], []),
    # bend
    (
        "chorus",
        [("gain_in", 0.0, 1.0), ("gain_out", 0.0, 1.0)],
        [("n_voices", [2, 3, 4])],
    ),
    (
        "compand",
        [
            ("attack_time", 0.01, 1.0),
            ("decay_time", 0.01, 2.0),
            ("soft_knee_db", 0.0, 12.0),
        ],
        [],
    ),
    ("contrast", [("amount", 0, 100)], []),
    # delays and decays and n_echos
    ("echo", [("gain_in", 0, 1), ("gain_out", 0, 1)], []),
    ("equalizer", [("midi", 21, 108), ("gain_db", -20, 20), ("width_q", 1, 10)], []),
    (
        "fade",
        [("fade_in_len", 0, 2), ("fade_out_len", 0, 2)],
        [("fade_shape", ["q", "h", "t", "l", "p"])],
    ),
    (
        "flanger",
        [
            ("delay", 0, 30),
            ("depth", 0, 10),
            ("regen", -95, 95),
            ("width", 0, 100),
            ("speed", 0.1, 10.0),
            ("phase", 0, 100),
        ],
        [("shape", ["triangle", "sine"]), ("interp", ["linear", "quadratic"])],
    ),
    (
        "gain",
        [("gain_db", -20, 20)],
        [
            ("normalize", [True, False]),
            ("limiter", [True, False]),
            ("balance", [None, "e", "B", "b"]),
        ],
    ),
]


def choose_value(name, low, hi):
    v = random.uniform(low, hi)
    if name == "midi":
        # MIDI is one of the few that is transformed to another scale, frequency
        return ("frequency", note_to_freq(v))
    else:
        return (name, v)


def transform_file(f):
    transform_spec = random.choice(transforms)
    transform = transform_spec[0]
    params = {}
    variable_spec = random.choice(transform_spec[1])
    variable = variable_spec[0]
    for param, low, hi in transform_spec[1]:
        if param == variable:
            continue
        param, v = choose_value(param, low, hi)
        params[param] = v
    for param, vals in transform_spec[2]:
        params[param] = random.choice(vals)

    variable_values = [
        choose_value(variable, variable_spec[1], variable_spec[2])[1]
        for i in range(N_TRANSFORMS)
    ]
    variable_values.sort()

    # print(transform, variable, params, variable_values)
    outfiles = []
    if variable == "midi":
        variable = "frequency"
    for v in variable_values:
        outf = os.path.join(
            "data/transforms",
            f"{transform}-{os.path.split(f)[1]}-{variable}-"
            + "{:010.3F}".format(v)
            + ".ogg",
        )
        outfiles.append(outf)
        tfm = sox.Transformer()
        newparams = copy.copy(params)
        newparams[variable] = v
        # print(newparams)
        tfm.__getattribute__(transform)(**newparams)
        tfm.build_file(f, outf)

    dists = []
    for outf2 in outfiles[1:]:
        input = torch.tensor(sf.read(outf2)[0])
        target = torch.tensor(sf.read(outfiles[0])[0])
        if len(input) < len(target):
            target = target[: len(input)]
        else:
            input = input[: len(target)]
        dists.append(dist(input, target).item())
    # print(dists)
    print(scipy.stats.spearmanr(dists, range(len(dists)))[0])


for f in tqdm(list(glob.glob("data/FSD50K.eval_audio/*.wav"))):
    transform_file(f)

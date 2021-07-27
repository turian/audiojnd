#!/usr/bin/env python3

import copy
import glob
import hashlib
import json
import os.path
import random

import soundfile as sf
import sox
from tqdm.auto import tqdm

N_TRANSFORMS = 32

random.seed(0)


def note_to_freq(note):
    a = 440  # frequency of A (common value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


# https://pysox.readthedocs.io/en/latest/api.html
# The format of each element here is:
# (transform name, [(continuous parameter, min, max), ...], [(categorical parameter, [values])])
# TODO: Add other transforms from JND paper?
transforms = [
    ("allpass", [("midi", 0, 127), ("width_q", 0.01, 5)], []),
    (
        "bandpass",
        [("midi", 0, 127), ("width_q", 0.01, 5)],
        [("constant_skirt", [True, False])],
    ),
    ("bandreject", [("midi", 0, 127), ("width_q", 0.01, 5)], []),
    ("bass", [("gain_db", -20, 20), ("midi", 0, 127), ("slope", 0.3, 1.0)], []),
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
    ("equalizer", [("midi", 0, 127), ("gain_db", -20, 20), ("width_q", 0.01, 5)], []),
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
    ("highpass", [("midi", 0, 127), ("width_q", 0.01, 5)], [("n_poles", [1, 2])]),
    # loudness: Loudness control. Similar to the gain effect, but
    # provides equalisation for the human auditory system.
    ("lowpass", [("midi", 0, 127), ("width_q", 0.01, 5)], [("n_poles", [1, 2])]),
    # mcompand
    # noisered
    # This might be too extreme?
    ("overdrive", [("gain_db", 0, 100), ("colour", 0, 100)], []),
    (
        "phaser",
        [
            ("gain_in", 0, 1),
            ("gain_out", 0, 1),
            ("delay", 0, 5),
            ("decay", 0.1, 0.5),
            ("speed", 0.1, 2),
        ],
        [("modulation_shape", ["sinusoidal", "triangular"])],
    ),
    ("pitch", [("n_semitones", -12, 12)], [("quick", [True, False])]),
    # rate
    (
        "reverb",
        [
            ("reverberance", 0, 100),
            ("high_freq_damping", 0, 100),
            ("room_scale", 0, 100),
            ("stereo_depth", 0, 100),
            ("pre_delay", 0, 500),
            ("wet_gain", -10, 10),
        ],
        [("wet_only", [True, False])],
    ),
    ("speed", [("factor", 0.5, 1.5)], []),
    ("stretch", [("factor", 0.5, 1.5)], [("window", [10, 20, 50])]),
    (
        "tempo",
        [("factor", 0.5, 1.5)],
        [("audio_type", ["m", "s", "l"]), ("quick", [True, False])],
    ),
    ("treble", [("gain_db", -20, 20), ("midi", 0, 127), ("slope", 0.3, 1.0)], []),
    ("tremolo", [("speed", 0.1, 10.0), ("depth", 0, 100)], []),
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
    print(transform_spec)
    for param, low, hi in transform_spec[1]:
        param, v = choose_value(param, low, hi)
        params[param] = v
    for param, vals in transform_spec[2]:
        params[param] = random.choice(vals)

    outfiles = []
    # TODO: Save JSON of all transforms
    slug = f"{os.path.split(f)[1]}-{transform}-{hashlib.sha224(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()[:4]}"
    outf = os.path.join(
        "data/transforms",
        os.path.split(os.path.split(f)[0])[1],
        os.path.split(f)[1],
        f"{slug}.ogg",
    )
    outd = os.path.split(outf)[0]
    if not os.path.exists(outd):
        os.makedirs(outd)
    outfiles.append(outf)
    tfm = sox.Transformer()
    tfm.__getattribute__(transform)(**params)
    tfm.build_file(f, outf)


for f in tqdm(list(glob.glob("data/preprocessed/*/*.ogg"))):
    transform_file(f)

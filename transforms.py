#!/usr/bin/env python3

import copy
import glob
import hashlib
import json
import os.path
import random

import numpy as np
import pydub
import soundfile as sf
import sox
from tqdm.auto import tqdm

from preprocess import ensure_length

CONFIG = json.loads(open("config.json").read())

def note_to_freq(note):
    a = 440  # frequency of A (common value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))

def pydubread(f):
    """
    MP3 to numpy array.
    We use pydub since soundfile can't read mp3s.
    """
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples(), dtype=np.float32)
    # Convert to float32 from int16
    y /= -32768
    assert a.frame_rate == CONFIG["SAMPLE_RATE"]
    return y


#def pydubwrite(f, sr, x, normalized=False):
#    """numpy array to MP3"""
#    assert sr == CONFIG["SAMPLE_RATE"]
#    assert x.ndim== 1
#    assert False, "Need to convert from -1, 1 to -32768, 32768"
#    y = np.int16(x)
#    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=1)
#    assert CONFIG["EXTENSION"] == "mp3"
#    song.export(f, format="mp3", bitrate="320k")

# https://pysox.readthedocs.io/en/latest/api.html
# The format of each element here is:
# (transform name, [(continuous parameter, min, max), ...], [(categorical parameter, [values])])
# TODO: Add other transforms from JND paper?
transforms = [
    ("sox", "allpass", [("midi", 0, 127), ("width_q", 0.01, 5)], []),
    (
        "sox", "bandpass",
        [("midi", 0, 127), ("width_q", 0.01, 5)],
        [("constant_skirt", [True, False])],
    ),
    ("sox", "bandreject", [("midi", 0, 127), ("width_q", 0.01, 5)], []),
    ("sox", "bass", [("gain_db", -20, 20), ("midi", 0, 127), ("slope", 0.3, 1.0)], []),
    # bend
    (
        "sox", "chorus",
        [("gain_in", 0.0, 1.0), ("gain_out", 0.0, 1.0)],
        [("n_voices", [2, 3, 4])],
    ),
    (
        "sox", "compand",
        [
            ("attack_time", 0.01, 1.0),
            ("decay_time", 0.01, 2.0),
            ("soft_knee_db", 0.0, 12.0),
        ],
        [],
    ),
    ("sox", "contrast", [("amount", 0, 100)], []),
    # delays and decays and n_echos
    ("sox", "echo", [("gain_in", 0, 1), ("gain_out", 0, 1)], []),
    ("sox", "equalizer", [("midi", 0, 127), ("gain_db", -20, 20), ("width_q", 0.01, 5)], []),
    (
        "sox", "fade",
        [("fade_in_len", 0, 2), ("fade_out_len", 0, 2)],
        [("fade_shape", ["q", "h", "t", "l", "p"])],
    ),
    (
        "sox", "flanger",
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
        "sox", "gain",
        [("gain_db", -20, 20)],
        [
            ("normalize", [True, False]),
            ("limiter", [True, False]),
            ("balance", [None, "e", "B", "b"]),
        ],
    ),
    ("sox", "highpass", [("midi", 0, 127), ("width_q", 0.01, 5)], [("n_poles", [1, 2])]),
    # loudness: Loudness control. Similar to the gain effect, but
    # provides equalisation for the human auditory system.
    ("sox", "lowpass", [("midi", 0, 127), ("width_q", 0.01, 5)], [("n_poles", [1, 2])]),
    # mcompand
    # noisered
    # This might be too extreme?
    ("sox", "overdrive", [("gain_db", 0, 100), ("colour", 0, 100)], []),
    (
        "sox", "phaser",
        [
            ("gain_in", 0, 1),
            ("gain_out", 0, 1),
            ("delay", 0, 5),
            ("decay", 0.1, 0.5),
            ("speed", 0.1, 2),
        ],
        [("modulation_shape", ["sinusoidal", "triangular"])],
    ),
    ("sox", "pitch", [("n_semitones", -12, 12)], [("quick", [True, False])]),
    # rate
    (
        "sox", "reverb",
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
    ("sox", "speed", [("factor", 0.5, 1.5)], []),
    ("sox", "stretch", [("factor", 0.5, 1.5)], [("window", [10, 20, 50])]),
    (
        "sox", "tempo",
        [("factor", 0.5, 1.5)],
        [("audio_type", ["m", "s", "l"]), ("quick", [True, False])],
    ),
    ("sox", "treble", [("gain_db", -20, 20), ("midi", 0, 127), ("slope", 0.3, 1.0)], []),
    ("sox", "tremolo", [("speed", 0.1, 10.0), ("depth", 0, 100)], []),
]


def choose_value(name, low, hi):
    v = random.uniform(low, hi)
    if name == "midi":
        # MIDI is one of the few that is transformed to another scale, frequency
        return ("frequency", note_to_freq(v))
    else:
        return (name, v)


def transform_file(f):
    x = pydubread(f)

    transform_spec = random.choice(transforms)
    transform = "%s-%s" % (transform_spec[0], transform_spec[1])
    params = {}
    #print(transform_spec)
    for param, low, hi in transform_spec[2]:
        param, v = choose_value(param, low, hi)
        params[param] = v
    for param, vals in transform_spec[3]:
        params[param] = random.choice(vals)

    ## Choose a wet/dry ratio between transform and original
    #wet = random.random()
    #wet = 1

    #outfiles = []
    # TODO: Save JSON of all transforms
    slug = f"{os.path.split(f)[1]}-{transform}-{hashlib.sha224(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()[:4]}"
    #print(slug)
    outf = os.path.join(
        "data/transforms",
        #"gold/transforms",
        os.path.split(os.path.split(f)[0])[1],
        os.path.split(f)[1],
        #f"{slug}.{CONFIG['EXTENSION']}",
        # WAV first, then MP3 later
        f"{slug}.wav"
    )
    outjson = os.path.splitext(outf)[0] + ".json"
    outd = os.path.split(outf)[0]
    if not os.path.exists(outd):
        os.makedirs(outd)
    #outfiles.append(outf)
    if transform_spec[0] == "sox":
        tfm = sox.Transformer()
        tfm.__getattribute__(transform_spec[1])(**params)
        # Try to make the same length as the original WAV
        #tfm.trim(0, len(x) / CONFIG["SAMPLE_RATE"])
        newx = tfm.build_array(input_array=x, sample_rate_in=CONFIG["SAMPLE_RATE"])
        # Try to make the same length as the original WAV
        # MP3 compression might fuck this up slightly
        newx = ensure_length(newx, len(x), from_start=True)

        ## Now do a wet/dry mix
        #newx = newx * wet + x * (1 - wet)

        #pydubwrite(outf, CONFIG["SAMPLE_RATE"], newx)
        sf.write(outf, newx, CONFIG["SAMPLE_RATE"])
        # Use lame so we can control the variable bitrate
        os.system(f"lame --quiet -V1 {outf}")
    else:
        assert False, f"Unknown transformer {transform_spec[0]}"
    #assert "wet" not in params
    #params["wet"] = wet
    open(outjson, "wt").write(json.dumps(
        [{"orig": f}, {transform: params}], indent=4))

    """
    # TODO: Different openl3 model?
    import torch
    import torchopenl3
    x1, sr1 = sf.read(f)
    emb1, ts1 = torchopenl3.get_audio_embedding(x1, sr1)
    x2, sr2 = sf.read(outf)
    emb2, ts2 = torchopenl3.get_audio_embedding(x2, sr2)
    # TODO: Use a different distance
    print(torch.mean(torch.abs(emb1 - emb2)).item())
    """

if __name__ == "__main__":
    files = list(glob.glob(f"data/preprocessed/FSD50K.dev_audio/*.{CONFIG['EXTENSION']}"))
    # Always shuffle the files deterministically (seed 0),
    # even if we use non-deterministic transforms of the audio.
    rng = random.Random(0)
    rng.shuffle(files)
    #files = list(glob.glob(f"data/preprocessed/*/*.{CONFIG['EXTENSION']}"))
    while 1:
        #for f in tqdm(files[:30]):
        for f in tqdm(files):
            for i in range(1):
                while 1:
                    try:
                        transform_file(f)
                        break
                    except sox.SoxError:
                        continue

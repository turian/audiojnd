#!/usr/bin/env python3

import copy
import glob
import hashlib
import json
import os.path
import random

import librosa
import numpy as np
import pydub
import soundfile as sf
import sox
from tqdm.auto import tqdm

import native_transformations
from preprocess import ensure_length

import audiomentations

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


# def pydubwrite(f, sr, x, normalized=False):
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
    ("native", "mulaw", [], [("quantization_channels", range(2, 256 + 1))]),
    ("sox", "allpass", [("midi", 0, 127), ("width_q", 0.01, 5)], []),
    (
        "sox",
        "bandpass",
        [("midi", 0, 127), ("width_q", 0.01, 5)],
        [("constant_skirt", [True, False])],
    ),
    ("sox", "bandreject", [("midi", 0, 127), ("width_q", 0.01, 5)], []),
    ("sox", "bass", [("gain_db", -20, 20), ("midi", 0, 127), ("slope", 0.3, 1.0)], []),
    # bend
    (
        "sox",
        "chorus",
        [("gain_in", 0.0, 1.0), ("gain_out", 0.0, 1.0)],
        [("n_voices", [2, 3, 4])],
    ),
    (
        "sox",
        "compand",
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
    (
        "sox",
        "equalizer",
        [("midi", 0, 127), ("gain_db", -20, 20), ("width_q", 0.01, 5)],
        [],
    ),
    (
        "sox",
        "fade",
        [("fade_in_len", 0, 2), ("fade_out_len", 0, 2)],
        [("fade_shape", ["q", "h", "t", "l", "p"])],
    ),
    (
        "sox",
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
        "sox",
        "gain",
        [("gain_db", -20, 20)],
        [
            ("normalize", [True, False]),
            ("limiter", [True, False]),
            ("balance", [None, "e", "B", "b"]),
        ],
    ),
    (
        "sox",
        "highpass",
        [("midi", 0, 127), ("width_q", 0.01, 5)],
        [("n_poles", [1, 2])],
    ),
    # loudness: Loudness control. Similar to the gain effect, but
    # provides equalisation for the human auditory system.
    ("sox", "lowpass", [("midi", 0, 127), ("width_q", 0.01, 5)], [("n_poles", [1, 2])]),
    # mcompand
    # noisered
    # This might be too extreme?
    ("sox", "overdrive", [("gain_db", 0, 100), ("colour", 0, 100)], []),
    (
        "sox",
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
    ("sox", "pitch", [("n_semitones", -12, 12)], [("quick", [True, False])]),
    # rate
    (
        "sox",
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
    ("sox", "speed", [("factor", 0.5, 1.5)], []),
    ("sox", "stretch", [("factor", 0.5, 1.5)], [("window", [10, 20, 50])]),
    (
        "sox",
        "tempo",
        [("factor", 0.5, 1.5)],
        [("audio_type", ["m", "s", "l"]), ("quick", [True, False])],
    ),
    (
        "sox",
        "treble",
        [("gain_db", -20, 20), ("midi", 0, 127), ("slope", 0.3, 1.0)],
        [],
    ),
    ("sox", "tremolo", [("speed", 0.1, 10.0), ("depth", 0, 100)], []),
    # Audiomentations Transforms
    (
        "audiomentations",
        "AddGaussianNoise",
        [],
        [("min_amplitude", [0.00001]), ("max_amplitude", [0.25])],
    ),
    (
        "audiomentations",
        "AddBackgroundNoise",
        [],
        [
            ("sounds_path", ["data/esc-50/ESC-50-master/audio/"]),
            ("min_snr_in_db", [0.001]),
            ("max_snr_in_db", [100]),
        ],
    ),
    (
        "audiomentations",
        "AddShortNoises",
        [("burst_probability", 0.01, 0.85)],
        [
            ("sounds_path", ["data/esc-50/ESC-50-master/audio/"]),
            ("min_snr_in_db", [0.0001]),
            ("max_snr_in_db", [100]),
            ("min_time_between_sounds", [0]),
            ("max_time_between_sounds", [3]),
            ("min_pause_factor_during_burst", [0.0]),
            ("min_pause_factor_during_burst", [1.0]),
        ],
    ),
    (
        "audiomentations",
        "ApplyImpulseResponse",
        [],
        [
            ("ir_path", ["data/MIT-McDermott-ImpulseResponse/Audio/"]),
            ("leave_length_unchanged", [True]),
        ],
    ),
    (
        "audiomentations",
        "Clip",
        [
            ("a_min", -1.0, 0.0),
            ("a_max", 0.0, 1.0),
        ],
        [],
    ),
    # Not really obvious enough
    #(
    #    "audiomentations",
    #    "Mp3Compression",
    #    [],
    #    [
    #        ("min_bitrate", [32]),
    #        ("max_bitrate", [320]),
    #    ],
    #),
    ("audiomentations", "Reverse", [], []),
    # Not yet in pypi
    # (
    #    "audiomentations",
    #    "TanhDistortion",
    #    [],
    #    [("min_distortion_gain", [0.01]), ("max_distortion_gain", [5.0])],
    # ),
    # Has a different __call__ structure
    # ("audiomentations", "SpecChannelShuffle", [], []),
    # ("audiomentations", "SpecChannelMask", [], []),
    (
        "audiomentations",
        "Resample",
        [],
        [("min_sample_rate", [4000]), ("max_sample_rate", [CONFIG["SAMPLE_RATE"] - 1])],
    ),
    (
        "audiomentations",
        "Mp3Compression",
        [],
        [
            ("min_bitrate", [32]),
            ("max_bitrate", [320]),
            ("backend", ["pydub"]),
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
    x = pydubread(f)

    transform_spec = random.choice(transforms)
    transform = "%s-%s" % (transform_spec[0], transform_spec[1])

    #if transform_spec[0] != "audiomentations":
    #    return

    params = {}
    #print(transform_spec)
    for param, low, hi in transform_spec[2]:
        param, v = choose_value(param, low, hi)
        params[param] = v
    for param, vals in transform_spec[3]:
        params[param] = random.choice(vals)

    # Choose a wet/dry ratio between transform and original
    # This is only really necessary for ApplyImpulseResponse
    # and SpecChannelShuffle, but is generally useful and not
    # harmful.
    wet = random.random()

    # outfiles = []
    # TODO: Save JSON of all transforms
    slug = f"{os.path.split(f)[1]}-{transform}-{hashlib.sha224(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()[:4]}"
    # print(slug)
    outf = os.path.join(
        "data/transforms",
        # "gold/transforms",
        os.path.split(os.path.split(f)[0])[1],
        os.path.split(f)[1],
        # f"{slug}.{CONFIG['EXTENSION']}",
        # WAV first, then MP3 later
        f"{slug}.wav",
    )
    outjson = os.path.splitext(outf)[0] + ".json"
    outd = os.path.split(outf)[0]
    if not os.path.exists(outd):
        os.makedirs(outd)
    # outfiles.append(outf)
    if transform_spec[0] == "sox":
        tfm = sox.Transformer()
        tfm.__getattribute__(transform_spec[1])(**params)
        # Try to make the same length as the original WAV
        # tfm.trim(0, len(x) / CONFIG["SAMPLE_RATE"])
        newx = tfm.build_array(input_array=x, sample_rate_in=CONFIG["SAMPLE_RATE"])
    elif transform_spec[0] == "native":
        newx = native_transformations.__getattribute__(transform_spec[1])(x, **params)
    elif transform_spec[0] == "audiomentations":
        tfm = audiomentations.__getattribute__(transform_spec[1])(p=1.0, **params)
        newx = tfm(x, CONFIG["SAMPLE_RATE"])
        if transform_spec[1] == "Resample":
            newx = librosa.core.resample(
            newx,
            orig_sr=tfm.parameters["target_sample_rate"],
            target_sr=CONFIG["SAMPLE_RATE"]
            )
        params = tfm.parameters
    else:
        assert False, f"Unknown transformer {transform_spec[0]}"

    # Try to make the same length as the original WAV
    # MP3 compression might fuck this up slightly
    newx = ensure_length(newx, len(x), from_start=True)

    # Now do a wet/dry mix
    newx = newx * wet + x * (1 - wet)

    # pydubwrite(outf, CONFIG["SAMPLE_RATE"], newx)
    sf.write(outf, newx, CONFIG["SAMPLE_RATE"])
    # Use lame so we can control the variable bitrate
    os.system(f"lame --quiet -V1 {outf}")
    assert "wet" not in params
    params["wet"] = wet
    open(outjson, "wt").write(json.dumps([{"orig": f}, {transform: params}], indent=4))

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
    files = list(
        glob.glob(f"data/preprocessed/FSD50K.dev_audio/*.{CONFIG['EXTENSION']}")
    )
    """
    # Always shuffle the files deterministically (seed 0),
    # even if we use non-deterministic transforms of the audio.
    rng = random.Random(0)
    rng.shuffle(files)
    """
    random.shuffle(files)
    # files = list(glob.glob(f"data/preprocessed/*/*.{CONFIG['EXTENSION']}"))
    while 1:
        # for f in tqdm(files[:30]):
        for f in tqdm(files):
            for i in range(1):
                while 1:
                    try:
                        transform_file(f)
                        break
                    except sox.SoxError:
                        continue

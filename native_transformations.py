import numpy as np

# From https://github.com/ibab/tensorflow-wavenet/blob/master/test/test_mu_law.py
# A set of mu law encode/decode functions implemented
# in numpy
def _manual_mu_law_encode(signal, quantization_channels):
    # Manual mu-law companding and mu-bits quantization
    mu = quantization_channels - 1

    magnitude = np.log1p(mu * np.abs(signal)) / np.log1p(mu)
    signal = np.sign(signal) * magnitude

    # Map signal from [-1, +1] to [0, mu-1]
    signal = (signal + 1) / 2 * mu + 0.5
    quantized_signal = signal.astype(np.int32)

    return quantized_signal


def _manual_mu_law_decode(signal, quantization_channels):
    # Calculate inverse mu-law companding and dequantization
    mu = quantization_channels - 1
    y = signal.astype(np.float32)

    y = 2 * (y / mu) - 1
    x = np.sign(y) * (1.0 / mu) * ((1.0 + mu) ** abs(y) - 1.0)
    return x


def mulaw(signal, quantization_channels):
    return _manual_mu_law_decode(
        _manual_mu_law_encode(signal, quantization_channels), quantization_channels
    )

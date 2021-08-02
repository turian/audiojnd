"""Different type of Augumentation We are Trying"""

import torch
import numpy as np
from audiomentations import (
    AddGaussianNoise,
    AddBackgroundNoise,
    AddShortNoises,
    Clip,
    Mp3Compression,
    Resample,
    SpecChannelShuffle,
)

from audiomentations.core.transforms_interface import BaseWaveformTransform


def apply_aug(audio, sr=16000, transform=None):
    if transform is not None:
        return transform(audio, sample_rate=sr)
    else:
        return audio


class Reverse(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self, p=1.0):
        super().__init__(p)

    def apply(self, samples, sample_rate):
        if len(samples.shape) > 1:
            return np.fliplr(samples)
        else:
            return np.flipud(samples)


class TanhDistortion(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self, min_distortion_gain=1.0, max_distortion_gain=2.0, p=1.0):
        super().__init__(p)
        self.min_distortion_gain = min_distortion_gain
        self.max_distortion_gain = max_distortion_gain

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["max_amplitude"] = np.amax(np.abs(samples))
            self.parameters["gain"] = random.uniform(
                self.min_distortion_gain, self.max_distortion_gain
            )

    def apply(self, samples, sample_rate):
        if self.parameters["max_amplitude"] > 0:
            distorted_samples = np.tanh(self.parameters["gain"] * samples)
            distorted_samples = (
                calculate_rms(distorted_samples) / calculate_rms(samples)
            ) * distorted_samples
        else:
            distorted_samples = samples
        return distorted_samples


if __name__ == "__main__":
    # Create a Batch of audio sample
    audio_samples = torch.rand(size=(8, 2, 32000), dtype=torch.float32) - 0.5
    # Add Gausian Noise
    audio = apply_aug(audio_samples, AddGaussianNoise(p=1.0))
    # Background Noise Folder
    backgroundnoisepaths = "/content/backgroundnoise/FSDKaggle2018.audio_test"
    # Add Background Noise
    audio = apply_aug(audio_samples, AddBackgroundNoise(backgroundnoisepaths, p=1.0))
    # Add Short Noise
    audio = apply_aug(audio_samples, AddShortNoises(backgroundnoisepaths, p=1.0))
    # Add Clip Which will take every value to -1 to 1
    audio = apply_aug(audio_samples, Clip(p=1.0))

    audio = apply_aug(audio_samples, Mp3Compression(p=1.0))
    audio = apply_aug(audio_samples, Reverse(p=1.0))
    audio = apply_aug(audio_samples, TanhDistortion(p=1.0))
    audio = apply_aug(audio_samples, Resample(p=1.0))
    audio = apply_aug(audio_samples, SpecChannelShuffle(p=1.0))


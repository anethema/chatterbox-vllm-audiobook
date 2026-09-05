"""mel-spectrogram extraction in Matcha-TTS"""
from librosa.filters import mel as librosa_mel_fn
import torch
import numpy as np


# Reuse filters and windows across inference calls. The keys include every
# setting that affects the cached tensor, so callers can safely override the
# spectrogram defaults.
mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

"""
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: 24000
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: 8000
    center: False

"""

def mel_spectrogram(y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
    """Copied from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/audio.py
    Set default values according to Cosyvoice's config.
    """

    if isinstance(y, np.ndarray):
        y = torch.tensor(y).float()

    if len(y.shape) == 1:
        y = y[None, ]

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    mel_key = (sampling_rate, n_fft, num_mels, fmin, fmax, y.device, y.dtype)
    window_key = (win_size, y.device, y.dtype)
    if mel_key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[mel_key] = torch.from_numpy(mel).to(device=y.device, dtype=y.dtype)
    if window_key not in hann_window:
        # Keep the original CPU construction before transferring to the
        # waveform device; CUDA-native construction can round differently.
        hann_window[window_key] = torch.hann_window(
            win_size,
            dtype=y.dtype,
        ).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[window_key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[mel_key], spec)
    spec = spectral_normalize_torch(spec)

    return spec

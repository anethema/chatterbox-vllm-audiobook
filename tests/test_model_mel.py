import unittest

import torch
from librosa.filters import mel as librosa_mel_fn

from chatterbox_vllm.models.s3gen.utils import mel as mel_module


class MelSpectrogramCacheTests(unittest.TestCase):
    def setUp(self):
        mel_module.mel_basis.clear()
        mel_module.hann_window.clear()
        self.addCleanup(mel_module.mel_basis.clear)
        self.addCleanup(mel_module.hann_window.clear)

    @staticmethod
    def _original_default_mel(waveform):
        mel_basis = torch.from_numpy(
            librosa_mel_fn(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000)
        ).float().to(waveform.device)
        window = torch.hann_window(1920).to(waveform.device)
        padded = torch.nn.functional.pad(
            waveform.unsqueeze(0).unsqueeze(1),
            (720, 720),
            mode="reflect",
        ).squeeze(1)
        spectrum = torch.view_as_real(
            torch.stft(
                padded,
                1920,
                hop_length=480,
                win_length=1920,
                window=window,
                center=False,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )
        magnitudes = torch.sqrt(spectrum.pow(2).sum(-1) + 1e-9)
        return torch.log(torch.clamp(mel_basis @ magnitudes, min=1e-5))

    def test_default_output_matches_original_window_construction(self):
        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            with self.subTest(device=device):
                waveform = torch.linspace(-0.5, 0.5, 2048, device=device)
                mel_module.mel_basis.clear()
                mel_module.hann_window.clear()
                expected = self._original_default_mel(waveform)
                actual = mel_module.mel_spectrogram(waveform)
                self.assertTrue(torch.equal(actual, expected))

    def test_filter_cache_includes_mel_parameters(self):
        waveform = torch.zeros(256)

        four_bands = mel_module.mel_spectrogram(
            waveform, n_fft=32, num_mels=4, sampling_rate=160,
            hop_size=8, win_size=32, fmin=0, fmax=60,
        )
        six_bands = mel_module.mel_spectrogram(
            waveform, n_fft=32, num_mels=6, sampling_rate=160,
            hop_size=8, win_size=32, fmin=0, fmax=60,
        )

        self.assertEqual(four_bands.shape[1], 4)
        self.assertEqual(six_bands.shape[1], 6)
        self.assertEqual(len(mel_module.mel_basis), 2)
        self.assertEqual(len(mel_module.hann_window), 1)

    def test_window_cache_includes_window_size_and_dtype(self):
        waveform = torch.zeros(256, dtype=torch.float64)

        first = mel_module.mel_spectrogram(
            waveform, n_fft=32, num_mels=4, sampling_rate=160,
            hop_size=8, win_size=32, fmin=0, fmax=60,
        )
        second = mel_module.mel_spectrogram(
            waveform, n_fft=32, num_mels=4, sampling_rate=160,
            hop_size=8, win_size=16, fmin=0, fmax=60,
        )

        self.assertEqual(first.dtype, torch.float64)
        self.assertEqual(second.dtype, torch.float64)
        self.assertEqual(len(mel_module.mel_basis), 1)
        self.assertEqual(len(mel_module.hann_window), 2)


if __name__ == "__main__":
    unittest.main()

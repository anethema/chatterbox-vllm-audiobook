"""Small real-FFmpeg reference checks; all fixtures are synthetic and temporary."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from chatterbox_vllm import audio


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required')
class ReferenceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        audio.clear_reference_cache()
        self.addCleanup(audio.clear_reference_cache)
        self.root = Path(self.directory.name)
        self.source = self.root / 'source.wav'
        samples = np.sin(np.arange(144123) * .06) * 4000
        with wave.open(str(self.source), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(48000)
            writer.writeframes(samples.astype('<i2').tobytes())

    def test_playlist_and_disguised_playlist_cannot_read_secondary_media(self):
        private = self.root / 'private.mp3'
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(self.source), str(private)], check=True, timeout=15)
        content = private.read_bytes()
        for suffix in ('.m3u8', '.mp3'):
            upload = self.root / ('upload' + suffix)
            upload.write_text('#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:3,\n' + str(private) + '\n#EXT-X-ENDLIST\n')
            with self.subTest(suffix=suffix), self.assertRaisesRegex(RuntimeError, 'FFmpeg failed to measure'):
                with audio.prepared_reference_audio(upload, 24000):
                    self.fail('A playlist was accepted as a voice reference')
        self.assertEqual(private.read_bytes(), content)

    def test_wav_mp3_and_browser_webm_are_supported(self):
        for suffix, codec in (('.wav', 'pcm_s16le'), ('.mp3', 'libmp3lame'), ('.webm', 'libopus')):
            path = self.root / ('reference' + suffix)
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(self.source), '-c:a', codec, str(path)], check=True, timeout=15)
            with self.subTest(suffix=suffix), audio.prepared_reference_audio(path, 24000) as prepared:
                audio._check_reference_duration(prepared)
                self.assertGreater(prepared.stat().st_size, 1000)

    def test_identical_saved_copy_reuses_preparation_and_lru_is_bounded(self):
        copy = self.root / 'copy.wav'
        shutil.copyfile(self.source, copy)
        with patch.object(audio, '_prepare_reference_audio', wraps=audio._prepare_reference_audio) as prepare:
            with audio.prepared_reference_audio(self.source, 24000) as first:
                result = first.read_bytes()
            with audio.prepared_reference_audio(copy, 24000) as second:
                self.assertEqual(first, second)
                self.assertEqual(result, second.read_bytes())
            self.assertEqual(prepare.call_count, 1)
        with patch.object(audio, 'REFERENCE_CACHE_BYTES', 1):
            with audio.prepared_reference_audio(copy, 24000) as borrowed:
                self.assertTrue(borrowed.exists())
        self.assertFalse(borrowed.exists())
        self.assertFalse(audio._reference_cache)

    def test_size_duration_and_process_time_are_bounded(self):
        with patch.object(audio, 'MAX_REFERENCE_BYTES', 16), self.assertRaises(ValueError):
            with audio.prepared_reference_audio(self.source, 24000):
                pass
        with patch.object(audio, 'MAX_REFERENCE_SECONDS', 1), self.assertRaisesRegex(ValueError, '5 minutes'):
            with audio.normalized_reference_audio(self.source, 24000):
                pass
        with patch.object(audio.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ffmpeg', 120)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                with audio.normalized_reference_audio(self.source, 24000):
                    pass
            self.assertEqual(run.call_args.kwargs['timeout'], 120)

    def test_streaming_rnnoise_matches_one_shot_including_partial_frame(self):
        from pyrnnoise import RNNoise

        with wave.open(str(self.source), 'rb') as reader:
            samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype='<i2').copy()
        expected = b''.join(np.clip(frame, -32768, 32767).astype('<i2').tobytes()
                            for _, frame in RNNoise(48000).denoise_chunk(samples[np.newaxis, :], partial=True))
        output = audio.denoise_reference_audio(self.source, self.root / 'denoised.wav')
        with wave.open(str(output), 'rb') as reader:
            actual = reader.readframes(reader.getnframes())
        self.assertEqual(actual, expected)
        self.assertEqual(self.source.stat().st_size, samples.nbytes + 44)


if __name__ == '__main__':
    unittest.main()

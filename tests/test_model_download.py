import unittest
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from chatterbox_vllm.model_download import (
    ENGLISH_V1_FILES,
    ENGLISH_V1_REVISION,
    MULTILINGUAL_FILES,
    _download_file,
    _incomplete_blob_path,
    ensure_model_downloaded,
)
from chatterbox_vllm.model_variants import (
    ENGLISH_V1_MODEL_ID,
    MULTILINGUAL_CHECKPOINTS,
    MULTILINGUAL_V2_MODEL_ID,
    MULTILINGUAL_V3_MODEL_ID,
)


class ModelDownloadTests(unittest.TestCase):
    def test_cached_english_files_skip_network_and_return_snapshot_directory(self):
        snapshot = Path("/cache/models--ResembleAI--chatterbox/snapshots/english")
        events: list[tuple[float, str]] = []

        def cached_file(**kwargs):
            return str(snapshot / kwargs["filename"])

        with (
            patch(
                "chatterbox_vllm.model_download.try_to_load_from_cache",
                side_effect=cached_file,
            ) as cache,
            patch("chatterbox_vllm.model_download.hf_hub_download") as download,
        ):
            result = ensure_model_downloaded(
                ENGLISH_V1_MODEL_ID,
                lambda fraction, message: events.append((fraction, message)),
            )

        self.assertEqual(result, snapshot)
        self.assertEqual(cache.call_count, len(ENGLISH_V1_FILES))
        self.assertFalse(download.called)
        self.assertEqual(events[-1], (1.0, "english-v1 checkpoint is ready"))
        self.assertTrue(all(0.0 <= fraction <= 1.0 for fraction, _ in events))
        self.assertTrue(any("Using cached" in message for _, message in events))
        self.assertEqual(
            {call.kwargs["revision"] for call in cache.call_args_list},
            {ENGLISH_V1_REVISION},
        )

    def test_missing_multilingual_files_use_each_pinned_revision(self):
        for model_id in (MULTILINGUAL_V2_MODEL_ID, MULTILINGUAL_V3_MODEL_ID):
            with self.subTest(model_id=model_id):
                t3_filename, expected_revision = MULTILINGUAL_CHECKPOINTS[model_id]
                snapshot = Path(f"/cache/snapshots/{model_id}")
                downloaded_calls: list[dict] = []

                def download_file(**kwargs):
                    downloaded_calls.append(kwargs)
                    return str(snapshot / kwargs["filename"])

                with (
                    patch(
                        "chatterbox_vllm.model_download.try_to_load_from_cache",
                        return_value=None,
                    ),
                    patch(
                        "chatterbox_vllm.model_download.hf_hub_download",
                        side_effect=download_file,
                    ),
                ):
                    result = ensure_model_downloaded(model_id)

                self.assertEqual(result, snapshot)
                self.assertEqual(
                    [call["filename"] for call in downloaded_calls],
                    [t3_filename, *MULTILINGUAL_FILES],
                )
                self.assertEqual(
                    {call["revision"] for call in downloaded_calls}, {expected_revision}
                )

    def test_explicit_revision_overrides_english_default(self):
        snapshot = Path("/cache/snapshots/explicit")

        with (
            patch(
                "chatterbox_vllm.model_download.try_to_load_from_cache",
                return_value=None,
            ),
            patch(
                "chatterbox_vllm.model_download.hf_hub_download",
                side_effect=lambda **kwargs: str(snapshot / kwargs["filename"]),
            ) as download,
        ):
            ensure_model_downloaded(ENGLISH_V1_MODEL_ID, revision="explicit-revision")

        self.assertEqual(
            {call.kwargs["revision"] for call in download.call_args_list},
            {"explicit-revision"},
        )

    def test_network_error_propagates_after_current_file_is_announced(self):
        events: list[tuple[float, str]] = []

        with (
            patch(
                "chatterbox_vllm.model_download.try_to_load_from_cache",
                return_value=None,
            ),
            patch(
                "chatterbox_vllm.model_download.get_hf_file_metadata",
                side_effect=RuntimeError("metadata unavailable"),
            ),
            patch(
                "chatterbox_vllm.model_download.hf_hub_download",
                side_effect=RuntimeError("network unavailable"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "network unavailable"):
                ensure_model_downloaded(
                    ENGLISH_V1_MODEL_ID,
                    lambda fraction, message: events.append((fraction, message)),
                )

        self.assertTrue(any("Downloading checkpoint file 1/5" in text for _, text in events))
        self.assertFalse(any("checkpoint is ready" in text for _, text in events))

    def test_slow_download_reports_elapsed_byte_progress_before_completion(self):
        events: list[tuple[float, str]] = []
        executor = MagicMock()
        executor.__enter__.return_value = executor
        future = MagicMock()
        future.result.side_effect = [
            FutureTimeoutError(),
            "/cache/snapshots/revision/file.safetensors",
        ]
        future.done.return_value = False
        executor.submit.return_value = future

        with tempfile.TemporaryDirectory() as directory:
            incomplete_path = Path(directory) / "file.incomplete"
            incomplete_path.write_bytes(b"x" * 50)
            with (
                patch(
                    "chatterbox_vllm.model_download.ThreadPoolExecutor",
                    return_value=executor,
                ),
                patch(
                    "chatterbox_vllm.model_download._incomplete_blob_path",
                    return_value=(incomplete_path, 100),
                ),
            ):
                result = _download_file(
                    repo_id="example/repo",
                    filename="file.safetensors",
                    revision="revision",
                    index=1,
                    total=2,
                    progress=lambda fraction, message: events.append((fraction, message)),
                )

        self.assertEqual(result, Path("/cache/snapshots/revision/file.safetensors"))
        self.assertTrue(
            any(
                fraction == 0.25
                and "50 B / 100 B (50%)" in message
                and "elapsed" in message
                for fraction, message in events
            )
        )

    def test_xet_metadata_uses_elapsed_progress_instead_of_blob_size(self):
        metadata = SimpleNamespace(
            etag="would-be-preallocated",
            size=2 * 1024**3,
            xet_file_data=object(),
        )

        with patch(
            "chatterbox_vllm.model_download.get_hf_file_metadata",
            return_value=metadata,
        ):
            self.assertEqual(
                _incomplete_blob_path("example/repo", "large.safetensors", "revision"),
                (None, None),
            )

    def test_downloader_timeout_error_is_not_mistaken_for_poll_timeout(self):
        with (
            patch(
                "chatterbox_vllm.model_download.try_to_load_from_cache",
                return_value=None,
            ),
            patch(
                "chatterbox_vllm.model_download.hf_hub_download",
                side_effect=TimeoutError("request timed out"),
            ),
        ):
            with self.assertRaisesRegex(TimeoutError, "request timed out"):
                ensure_model_downloaded(ENGLISH_V1_MODEL_ID)


if __name__ == "__main__":
    unittest.main()


import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chatterbox_vllm.job_status import JobSnapshot, JobStatusStore, render_job_status


class JobStatusStoreTests(unittest.TestCase):
    def test_second_caller_cannot_acquire_active_job(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStatusStore(directory, clock=lambda: 10.0)
            started, first = store.try_start(project_id="resume-token")
            self.assertTrue(started)
            self.assertTrue(first.active)

            started, second = store.try_start(project_id="other")
            self.assertFalse(started)
            self.assertEqual(second.project_id, "resume-token")
            self.assertEqual(second.state, "running")

    def test_active_checkpoint_becomes_interrupted_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            first = JobStatusStore(directory, clock=lambda: 10.0)
            first.try_start(project_id="saved-project")

            second = JobStatusStore(directory, clock=lambda: 20.0)
            snapshot = second.snapshot()
            self.assertEqual(snapshot.state, "interrupted")
            self.assertFalse(snapshot.active)
            self.assertEqual(snapshot.project_id, "saved-project")
            persisted = json.loads(second.path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["state"], "interrupted")

    def test_malformed_or_nonfinite_checkpoint_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("active-job.json").write_text(
                json.dumps({"state": "running", "fraction": float("nan")}),
                encoding="utf-8",
            )
            store = JobStatusStore(root, clock=lambda: 10.0)
            self.assertEqual(store.snapshot().state, "idle")

    def test_stop_transition_cannot_overwrite_a_finished_job(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStatusStore(directory, clock=lambda: 10.0)
            store.try_start()
            store.finish("completed", "done")
            self.assertEqual(store.request_stop().state, "completed")

    def test_write_failure_keeps_in_memory_status_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStatusStore(directory, clock=lambda: 10.0)
            with patch.object(store, "_write", side_effect=OSError("full")):
                with self.assertWarns(RuntimeWarning):
                    started, snapshot = store.try_start()
                self.assertTrue(started)
                self.assertEqual(snapshot.state, "running")
                with self.assertWarns(RuntimeWarning):
                    finished = store.finish("stopped", "saved for resume")
            self.assertEqual(finished.state, "stopped")

    def test_monitor_render_shows_remote_control_relevant_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStatusStore(directory, clock=lambda: 10.0)
            store.try_start(project_id="project-token")
            store.update(
                phase="generating",
                message="Generating speech chunks…",
                fraction=0.5,
                completed_chunks=5,
                total_chunks=10,
                realtime_speed=12.5,
                eta_seconds=42,
            )
            rendered = render_job_status(
                store.snapshot(),
                format_duration=lambda seconds: f"{seconds:.0f}s",
            )
        self.assertIn("Running", rendered)
        self.assertIn("5/10 chunks", rendered)
        self.assertIn("ETA 42s", rendered)
        self.assertIn("project-token", rendered)
        self.assertIn('role="progressbar"', rendered)
        self.assertIn('style="width: 50.0%"', rendered)

    def test_monitor_render_escapes_dynamic_values(self):
        snapshot = JobSnapshot(
            state="running",
            phase="<phase>",
            message="<img src=x onerror=alert(1)>",
            fraction=0.125,
            completed_chunks=1,
            total_chunks=8,
            project_id="<project>",
        )
        rendered = render_job_status(snapshot, format_duration=lambda seconds: f"{seconds}s")
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", rendered)
        self.assertIn("&lt;Phase&gt;", rendered)
        self.assertIn("Project: &lt;project&gt;", rendered)
        self.assertNotIn("<img src=x", rendered)
        self.assertIn('style="width: 12.5%"', rendered)


if __name__ == "__main__":
    unittest.main()

import threading
import unittest

from chatterbox_vllm.background import BackgroundTaskPool


class BackgroundTaskPoolTests(unittest.TestCase):
    def test_tasks_run_concurrently(self):
        barrier = threading.Barrier(3)
        pool = BackgroundTaskPool(max_workers=2, max_pending=4)

        def synchronized_task():
            barrier.wait(timeout=2)

        pool.submit(synchronized_task)
        pool.submit(synchronized_task)
        barrier.wait(timeout=2)
        pool.finish()

    def test_finish_propagates_worker_failure_and_closes_pool(self):
        pool = BackgroundTaskPool(max_workers=1, max_pending=2)

        def fail():
            raise RuntimeError("normalization failed")

        pool.submit(fail)
        with self.assertRaisesRegex(RuntimeError, "normalization failed"):
            pool.finish()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.submit(lambda: None)

    def test_cancel_waits_for_running_work_and_cancels_queue(self):
        started = threading.Event()
        release = threading.Event()
        queued_ran = threading.Event()
        pool = BackgroundTaskPool(max_workers=1, max_pending=3)

        def running_task():
            started.set()
            release.wait(timeout=2)

        pool.submit(running_task)
        queued_future = pool.submit(queued_ran.set)
        self.assertTrue(started.wait(timeout=2))
        cancelled = threading.Event()
        queued_future.add_done_callback(lambda future: cancelled.set())
        cancel_thread = threading.Thread(target=pool.cancel_and_wait)
        cancel_thread.start()
        self.assertTrue(cancelled.wait(timeout=2))
        release.set()
        cancel_thread.join(timeout=2)
        self.assertFalse(cancel_thread.is_alive())
        self.assertTrue(queued_future.cancelled())
        self.assertFalse(queued_ran.is_set())

    def test_rejects_an_unbounded_worker_configuration(self):
        with self.assertRaisesRegex(ValueError, "max_pending"):
            BackgroundTaskPool(max_workers=2, max_pending=1)


if __name__ == "__main__":
    unittest.main()

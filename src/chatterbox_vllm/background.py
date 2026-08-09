from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable
from typing import Any


class BackgroundTaskPool:
    """Run bounded background work and surface worker failures to the caller."""

    def __init__(self, max_workers: int, max_pending: int):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_pending < max_workers:
            raise ValueError("max_pending must be at least max_workers")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="chatterbox-audio",
        )
        self._max_pending = max_pending
        self._pending: set[Future] = set()
        self._closed = False

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _resolve(self, completed: set[Future]) -> None:
        self._pending.difference_update(completed)
        for future in completed:
            future.result()

    def check(self) -> None:
        """Raise failures from completed tasks without waiting for active work."""

        completed = {future for future in self._pending if future.done()}
        if completed:
            self._resolve(completed)

    def submit(
        self,
        function: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future:
        if self._closed:
            raise RuntimeError("Background task pool is closed")
        self.check()
        while len(self._pending) >= self._max_pending:
            completed, _ = wait(self._pending, return_when=FIRST_COMPLETED)
            self._resolve(completed)
        future = self._executor.submit(function, *args, **kwargs)
        self._pending.add(future)
        return future

    def finish(self) -> None:
        """Wait for all work, propagating failures, and close the pool."""

        if self._closed:
            return
        try:
            while self._pending:
                completed, _ = wait(self._pending, return_when=FIRST_COMPLETED)
                self._resolve(completed)
        finally:
            for future in self._pending:
                future.cancel()
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._pending.clear()
            self._closed = True

    def cancel_and_wait(self) -> None:
        """Cancel queued work and wait for already-running tasks to leave files idle."""

        if self._closed:
            return
        for future in self._pending:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._closed = True

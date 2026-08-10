import tempfile
import unittest
from pathlib import Path

from chatterbox_vllm.memory import read_memory_status


class MemoryStatusTests(unittest.TestCase):
    def test_reads_available_memory_and_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text(
                "MemTotal:       100000 kB\n"
                "MemAvailable:    4096 kB\n"
                "SwapFree:        2048 kB\n",
                encoding="utf-8",
            )
            status = read_memory_status(path)
        self.assertEqual(status.available_bytes, 4096 * 1024)
        self.assertEqual(status.swap_free_bytes, 2048 * 1024)
        self.assertEqual(status.headroom_bytes, 6144 * 1024)


if __name__ == "__main__":
    unittest.main()

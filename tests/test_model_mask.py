import unittest

import torch

from chatterbox_vllm.models.s3gen.utils.mask import add_optional_chunk_mask


class OptionalChunkMaskTests(unittest.TestCase):
    def test_all_false_rows_are_repaired_without_crashing(self):
        features = torch.zeros((1, 2, 4))
        mask = torch.zeros((1, 1, 2), dtype=torch.bool)

        with self.assertLogs(level="WARNING") as captured:
            result = add_optional_chunk_mask(
                features,
                mask,
                use_dynamic_chunk=False,
                use_dynamic_left_chunk=False,
                decoding_chunk_size=-1,
                static_chunk_size=0,
                num_decoding_left_chunks=-1,
            )

        self.assertTrue(torch.all(result))
        self.assertIn("all-false", captured.output[0])


if __name__ == "__main__":
    unittest.main()

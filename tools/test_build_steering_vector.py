#!/usr/bin/env python3
"""Unit tests for directional steering vector computation and format conversions."""

import math
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_steering_vector import compute_steering_directions, dot, normalize, save_steering_files
from ds4_to_cvec import ds4_to_gguf, gguf_to_ds4
from gguf_reader import GGUFFile


class TestSteeringVectorMath(unittest.TestCase):
    def setUp(self):
        self.n_layer = 48
        self.n_embd = 2560
        import random
        random.seed(42)

        # Generate synthetic activations for 3 pairs
        self.good_acts = []
        self.bad_acts = []
        for _ in range(3):
            # good: has positive bias on feature 0
            g = [[random.gauss(0.5, 1.0) for _ in range(self.n_embd)] for _ in range(self.n_layer)]
            # bad: has negative bias on feature 0
            b = [[random.gauss(-0.5, 1.0) for _ in range(self.n_embd)] for _ in range(self.n_layer)]
            self.good_acts.append(g)
            self.bad_acts.append(b)

    def test_unit_normalization(self):
        directions = compute_steering_directions(
            self.good_acts, self.bad_acts, self.n_layer, self.n_embd, pair_normalize=False, orthogonalize=False
        )
        self.assertEqual(len(directions), self.n_layer)
        for l in range(self.n_layer):
            norm = math.sqrt(sum(x * x for x in directions[l]))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_orthogonalization_against_control_mean(self):
        directions = compute_steering_directions(
            self.good_acts, self.bad_acts, self.n_layer, self.n_embd, pair_normalize=False, orthogonalize=True
        )
        for l in range(self.n_layer):
            bad_mean = [sum(self.bad_acts[i][l][j] for i in range(3)) / 3.0 for j in range(self.n_embd)]
            base = normalize(bad_mean)
            proj = dot(directions[l], base)
            self.assertAlmostEqual(proj, 0.0, places=5, msg=f"Layer {l} not orthogonal to control mean")

    def test_save_and_convert(self):
        directions = compute_steering_directions(
            self.good_acts, self.bad_acts, self.n_layer, self.n_embd, pair_normalize=True, orthogonalize=True
        )
        with tempfile.TemporaryDirectory(prefix="test-steer-") as td:
            prefix = pathlib.Path(td) / "test_vec"
            f32_p, gguf_p = save_steering_files(
                directions, prefix, "qwen3.8-flash-next", "ffn_out", "dummy_good", "dummy_bad", True, True, write_gguf=True
            )
            self.assertTrue(f32_p.is_file())
            self.assertTrue(gguf_p.is_file())
            self.assertEqual(f32_p.stat().st_size, self.n_layer * self.n_embd * 4)

            # Inspect GGUF
            gf = GGUFFile(gguf_p)
            self.assertEqual(gf.metadata.get("general.architecture"), "controlvector")
            self.assertEqual(gf.metadata.get("controlvector.layer_count"), 47)
            self.assertEqual(len(gf.tensors), 48)  # layers 0..47


if __name__ == "__main__":
    unittest.main()

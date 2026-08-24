import unittest

import numpy as np

from scripts.challenge_metrics import (
    bootstrap_macro_soft_threshold_rae,
    macro_soft_threshold_rae,
    soft_threshold_rae,
)


class ChallengeMetricTests(unittest.TestCase):
    def test_predictions_inside_intervals_have_zero_error(self):
        score = soft_threshold_rae(
            [3.0, 5.0], [3.2, 4.8], [2.8, 4.7], [3.3, 5.2]
        )
        self.assertEqual(score, 0.0)

    def test_baseline_mean_scores_one(self):
        true = np.array([2.0, 4.0, 8.0])
        lower = true - 0.25
        upper = true + 0.25
        pred = np.full_like(true, true.mean())
        self.assertAlmostEqual(soft_threshold_rae(true, pred, lower, upper), 1.0)

    def test_macro_average_weights_endpoints_equally(self):
        true = [0.0, 2.0, 0.0, 2.0]
        pred = [0.0, 2.0, 1.0, 1.0]
        lower = true
        upper = true
        macro, per_endpoint = macro_soft_threshold_rae(
            true, pred, lower, upper, [0, 0, 1, 1], ["A", "B"]
        )
        self.assertEqual(per_endpoint["A"], 0.0)
        self.assertEqual(per_endpoint["B"], 1.0)
        self.assertEqual(macro, 0.5)

    def test_bootstrap_is_deterministic(self):
        args = (
            [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
            [0.1, 0.8, 2.2, 2.7, 0.2, 1.1, 1.8, 3.2],
            [-0.1, 0.9, 1.9, 2.9, -0.1, 0.9, 1.9, 2.9],
            [0.1, 1.1, 2.1, 3.1, 0.1, 1.1, 2.1, 3.1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            ["A", "B"],
        )
        first = bootstrap_macro_soft_threshold_rae(*args, n_bootstrap_samples=50)
        second = bootstrap_macro_soft_threshold_rae(*args, n_bootstrap_samples=50)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

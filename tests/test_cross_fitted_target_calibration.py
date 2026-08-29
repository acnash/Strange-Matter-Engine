import unittest

import numpy as np

from scripts.run_cross_fitted_target_calibrated_ensemble import (
    apply_calibration,
    fit_affine_calibration,
    fit_ridge,
    predict_ridge,
)


class CrossFittedTargetCalibrationTests(unittest.TestCase):
    def test_ridge_recovers_linear_target_with_intercept(self):
        x = np.arange(30, dtype=float).reshape(10, 3)
        target = 2.5 + x @ np.array([0.2, -0.4, 0.6])
        state = fit_ridge(x, target, 1e-8)
        np.testing.assert_allclose(predict_ridge(x, state), target, atol=1e-6)

    def test_affine_calibration_corrects_scale_and_offset(self):
        raw = np.linspace(3.0, 8.0, 50)
        target = 0.8 * raw + 0.7
        state = fit_affine_calibration(raw, target, 0.0)
        np.testing.assert_allclose(apply_calibration(raw, state), target, atol=1e-8)

    def test_identity_calibration_leaves_predictions_unchanged(self):
        raw = np.array([4.2, 5.1, 7.4])
        state = fit_affine_calibration(raw, raw + 2.0, float("inf"))
        np.testing.assert_array_equal(apply_calibration(raw, state), raw)

    def test_constant_member_column_remains_finite(self):
        x = np.column_stack((np.arange(8, dtype=float), np.ones(8)))
        state = fit_ridge(x, np.arange(8, dtype=float), 1.0)
        self.assertTrue(np.isfinite(predict_ridge(x, state)).all())


if __name__ == "__main__":
    unittest.main()

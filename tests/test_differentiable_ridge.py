import unittest

import torch

from scripts.run_graph_ca_visual_prototype import (
    differentiable_ridge_fit,
    differentiable_ridge_predict,
)


class DifferentiableRidgeTests(unittest.TestCase):
    def test_matches_direct_standardized_ridge_solution(self):
        features = torch.tensor([
            [0.0, 1.0],
            [1.0, 1.5],
            [2.0, 2.5],
            [3.0, 4.0],
        ], dtype=torch.float64)
        targets = torch.tensor([1.0, 2.0, 2.5, 4.5], dtype=torch.float64)
        penalty = 0.3

        state = differentiable_ridge_fit(features, targets, penalty)
        standardized = (features - features.mean(0)) / features.std(0, unbiased=False)
        expected = torch.linalg.solve(
            standardized.T @ standardized + penalty * torch.eye(2, dtype=torch.float64),
            standardized.T @ (targets - targets.mean()),
        )

        torch.testing.assert_close(state["coefficients"], expected)
        torch.testing.assert_close(state["intercept"], targets.mean())

    def test_query_loss_backpropagates_to_support_and_query_features(self):
        generator = torch.Generator().manual_seed(19)
        features = torch.randn((12, 5), generator=generator, requires_grad=True)
        targets = torch.randn((12,), generator=generator)

        state = differentiable_ridge_fit(features[:8], targets[:8], 0.1)
        predictions = differentiable_ridge_predict(features[8:], state)
        loss = (predictions - targets[8:]).square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad[:8].norm()), 0.0)
        self.assertGreater(float(features.grad[8:].norm()), 0.0)

    def test_intercept_is_not_shrunk(self):
        features = torch.zeros((5, 3), dtype=torch.float64)
        targets = torch.full((5,), 7.25, dtype=torch.float64)
        state = differentiable_ridge_fit(features, targets, 1000.0)
        predictions = differentiable_ridge_predict(features, state)

        torch.testing.assert_close(predictions, targets)
        torch.testing.assert_close(state["coefficients"], torch.zeros(3, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()

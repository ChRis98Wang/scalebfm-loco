"""XY-local / world-Z target regression; no stepping or running simulator."""
import unittest
from unittest import mock

import numpy as np
import torch

from scripts import evaluate_bfm_sim2sim_height as evaluation
from scalebridge.env.motion_tracking import MotionTrackingEnv
from scalebridge.env.motion_tracking_height import HeightAwareMotionTrackingEnv


class HeightAwareTests(unittest.TestCase):
    def test_local_alignment_removes_xy_but_retains_height_error(self):
        reference = np.arange(42).reshape(14, 3)*.01
        offset = np.array([3., 4., .2])
        actual = reference + offset
        local, world = evaluation.tracking_distances(actual, reference, actual[0], True)
        np.testing.assert_allclose(local, .2)
        np.testing.assert_allclose(world, np.linalg.norm(offset))

    def test_zero_local_pelvis_error_requires_matching_height(self):
        reference = np.zeros((14, 3))
        reference[:, 2] = .8
        actual = reference.copy()
        actual[:, :2] += 5
        local, _ = evaluation.tracking_distances(actual, reference, actual[0], True)
        np.testing.assert_allclose(local, 0.)
        actual[0, 2] = .6
        local, _ = evaluation.tracking_distances(actual, reference, actual[0], True)
        self.assertAlmostEqual(local[0], .2)

    def test_reference_gather_preserves_measured_height_without_mutating_source(self):
        env = HeightAwareMotionTrackingEnv.__new__(HeightAwareMotionTrackingEnv)
        env.reference_forcing = True
        env.state_buffer = {"root_pos_buffer": torch.tensor([[[.1, .2, .7], [.3, .4, .6]]])}
        reference = torch.tensor([[[9., 8., .5]]])
        original_reference = reference.clone()
        def legacy_gather(instance):
            instance.state_buffer["root_pos_buffer"][:, -1] = reference[:, 0]
        with mock.patch.object(MotionTrackingEnv, "_gather_reference_state", legacy_gather):
            env._gather_reference_state()
        torch.testing.assert_close(env.state_buffer["root_pos_buffer"][:, -1], torch.tensor([[9., 8., .6]]))
        torch.testing.assert_close(reference, original_reference)
        # A changed actual height next step must not reuse the previous height.
        env.state_buffer["root_pos_buffer"][:, -1, 2] = .55
        with mock.patch.object(MotionTrackingEnv, "_gather_reference_state", legacy_gather):
            env._gather_reference_state()
        self.assertAlmostEqual(env.state_buffer["root_pos_buffer"][0, -1, 2].item(), .55)

    def test_global_branch_is_unchanged(self):
        actual = np.zeros((14, 3))
        reference = np.ones((14, 3))
        control, world = evaluation.tracking_distances(actual, reference, actual[0], False)
        np.testing.assert_array_equal(control, world)


if __name__ == "__main__":
    unittest.main()

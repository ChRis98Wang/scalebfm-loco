"""Metric coordinate/time contract tests; no physics stepping or GUI."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import evaluate_bfm_sim2sim as evaluation
from scripts import export_bfm_torchscript as exporting


class Sim2SimEvaluationTests(unittest.TestCase):
    def test_world_and_local_translation_are_separate(self):
        reference = np.arange(42).reshape(14, 3) * .01
        offset = np.array([3., 4., .5])
        actual = reference + offset
        global_error, global_world = evaluation.tracking_distances(actual, reference, actual[0], False)
        local_error, local_world = evaluation.tracking_distances(actual, reference, actual[0], True)
        np.testing.assert_allclose(global_error, np.linalg.norm(offset))
        np.testing.assert_allclose(local_error, 0., atol=1e-14)
        np.testing.assert_allclose(local_world, global_world)

    def test_global_height_error_is_not_removed(self):
        reference = np.zeros((14, 3))
        reference[:, 2] = .8
        actual = reference.copy()
        actual[:, 2] -= .1
        error, _ = evaluation.tracking_distances(actual, reference, actual[0], False)
        np.testing.assert_allclose(error, .1)

    def test_bad_positions_rejected_before_serialization(self):
        for actual in (np.zeros((13, 3)), np.full((14, 3), np.nan), np.full((14, 3), np.inf)):
            with self.assertRaises(ValueError):
                evaluation.tracking_distances(actual, np.zeros((14, 3)), np.zeros(3), False)

    def test_environment_config_is_headless_rsi_without_real_backend(self):
        metadata = {"future_idx": [0, 1, 2, 3, 4, 5]}
        cfg = evaluation.environment_config("motion.npz", metadata, "robot.xml", False, True)
        self.assertEqual(cfg.simulator._target_, "scalebridge.simulator.mujoco_simulator.MujocoSimulator")
        self.assertTrue(cfg.rsi)
        self.assertFalse(cfg.reference_forcing)
        self.assertTrue(cfg.simulator.config.headless)
        self.assertFalse(cfg.simulator.config.joystick)
        self.assertFalse(cfg.simulator.config.record_video)
        self.assertEqual(cfg.simulator.config.low_dt * cfg.simulator.config.decimation, .02)

    def test_missing_input_produces_error_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            report = evaluation.evaluate("absent.pt", "absent", "absent.xml", output, 10, [7], [False], "cpu", True)
            self.assertEqual(report["result"], "ERROR")
            self.assertEqual(report["results"], [])
            self.assertFalse(report["hardware_accessed"])
            self.assertEqual(json.loads((output / "report.json").read_text()), report)

    def test_export_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileExistsError):
            exporting.export("absent.pt", "absent.json", "absent.xml", directory)

    def test_preflight_cannot_escape_owned_service(self):
        with mock.patch.object(Path, "read_text", return_value="0::/user.slice/shell.scope\n"):
            with self.assertRaises(ValueError):
                exporting.bounded_unit("bfm-sim2sim-")


if __name__ == "__main__":
    unittest.main()

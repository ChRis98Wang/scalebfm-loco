import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import audit_umr_body_fk as audit


class BodyFkTests(unittest.TestCase):
    def test_quaternion_sign_and_scale_are_equivalent(self):
        q = np.array([[.7, .2, .3, .4]], dtype=np.float32)
        np.testing.assert_allclose(audit.quaternion_angle(q, -q), 0, atol=1e-15)
        np.testing.assert_allclose(audit.quaternion_angle(q, np.asarray(q, dtype=float) * 1.0000001), 0, atol=1e-15)

    def test_small_angle_not_float32_acos_artifact(self):
        theta = 1e-6
        actual = np.array([[np.cos(theta / 2), np.sin(theta / 2), 0, 0]], dtype=np.float32)
        self.assertAlmostEqual(float(audit.quaternion_angle([[1, 0, 0, 0]], actual)[0]), theta, places=12)

    def test_pi_angle(self):
        self.assertAlmostEqual(float(audit.quaternion_angle([1, 0, 0, 0], [0, 0, 1, 0])), np.pi)

    def test_bad_quaternions_fail(self):
        for bad in ([0, 0, 0, 0], [np.nan, 0, 0, 1], [1, 0, 0]):
            with self.assertRaises(ValueError):
                audit.quaternion_angle([1, 0, 0, 0], bad)

    def test_error_stats_and_worst_link(self):
        errors = np.array([[0., .01], [.03, .02]])
        stats = audit.error_statistics(errors)
        self.assertEqual(stats["max"], .03)
        self.assertAlmostEqual(stats["rms"], np.sqrt(.00035))
        self.assertEqual(audit.worst_error(errors, ["pelvis", "wrist"]),
                         {"frame_index": 1, "body_name": "pelvis", "value": .03})

    def test_pose_errors_do_not_fit_away_translation(self):
        p = np.zeros((2, 30, 3)); q = np.tile([1., 0, 0, 0], (2, 30, 1))
        after = p.copy(); after[..., 2] += .002
        errors, angles = audit.pose_errors(p, q, after, -q)
        np.testing.assert_allclose(errors, .002)
        np.testing.assert_allclose(angles, 0)
        self.assertGreater(errors.max(), audit.POSITION_TOLERANCE_M)

    def test_wrong_body_count_rejected(self):
        with self.assertRaises(ValueError):
            audit.pose_errors(np.zeros((2, 29, 3)), np.zeros((2, 29, 4)),
                              np.zeros((2, 29, 3)), np.zeros((2, 29, 4)))

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"; out.write_text("keep")
            with self.assertRaises(FileExistsError):
                audit.run_audit(Path(tmp) / "missing.json", tmp, out)
            self.assertEqual(out.read_text(), "keep")

    def test_integrity_failure_is_preserved_not_passed(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(audit, "remember"), \
             mock.patch.object(audit, "semantic_fingerprint", return_value="geometry"), \
             mock.patch.object(audit, "validate_inputs", side_effect=ValueError("frame mismatch")):
            out = Path(tmp) / "failed.json"
            report = audit.run_audit(Path(tmp) / "manifest.json", tmp, out)
            self.assertEqual(report["result"], "ERROR")
            self.assertIn("frame mismatch", report["error"])
            self.assertFalse(report["inputs_verified_unchanged"])
            self.assertEqual(json.loads(out.read_text()), report)

    def test_negative_or_empty_error_statistics_rejected(self):
        for values in ([], [-1], [np.inf]):
            with self.assertRaises(ValueError):
                audit.error_statistics(values)

    def test_incomplete_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"schema": 1, "training_started": False,
                                           "automatic_promotion": False, "rows": []}))
            with self.assertRaisesRegex(ValueError, "forty"):
                audit.validate_inputs(manifest, tmp, {})

    def test_changed_file_hash_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "input"; file.write_text("old")
            frozen = {}; audit.remember(file, frozen); file.write_text("new")
            with self.assertRaisesRegex(ValueError, "changed"):
                audit.remember(file, frozen)
            link = Path(tmp) / "link"; link.symlink_to(file)
            with self.assertRaisesRegex(ValueError, "nonredirected"):
                audit.remember(link, {})

    def test_unowned_service_rejected(self):
        with mock.patch.object(Path, "read_text", return_value="0::/user.slice/not-owned.service"):
            with self.assertRaisesRegex(ValueError, "bfm-umr-fk"):
                audit.owned_unit()

    def test_unbounded_service_rejected(self):
        with mock.patch.object(Path, "read_text", return_value="0::/bfm-umr-fk-test.service"), \
             mock.patch.object(audit.subprocess, "check_output", return_value=
                               "KillMode=control-group\nRestart=no\nRuntimeMaxUSec=infinity\nMemoryMax=4G\nTasksMax=64\n"):
            with self.assertRaisesRegex(ValueError, "finite"):
                audit.owned_unit()


if __name__ == "__main__":
    unittest.main()

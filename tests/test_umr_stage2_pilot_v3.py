"""No simulator/data required for v3 controller contract regression."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scripts import run_umr_stage2_pilot_v3 as pilot


def result():
    q = np.zeros((3, 36)); q[:, 3] = 1
    return SimpleNamespace(qpos=q, frame_indices=np.arange(3), fps=50., point_error=np.zeros(3),
                           normal_error=np.zeros(3), contact_count=np.zeros(3, dtype=int),
                           floor_rows=np.zeros(3, dtype=int))


def audit():
    return {"result": "COMPLETE_NOT_QUALITY_ACCEPTED", "arm": "both",
            "warmup": {"frame": 0, "warmup": True, "failures": 1},
            "frames": [{"frame": i, "warmup": False, "failures": 0} for i in range(3)],
            "warmup_failures": 1, "output_failures": 0, "total_observed_failures": 1,
            "rate_enabled": True, "wrist_tasks": ["left", "right"], "postprocessing_performed": False}


class ResultTests(unittest.TestCase):
    def test_finite_matching_result(self):
        pilot.validate_result(result(), 3)

    def test_nonfinite_and_wrong_shapes_all_fields(self):
        for field in ("qpos", "point_error", "normal_error", "frame_indices", "contact_count", "floor_rows"):
            with self.subTest(field=field):
                r = result(); setattr(r, field, np.ones(2))
                with self.assertRaises(ValueError):
                    pilot.validate_result(r, 3)
                r = result(); v = getattr(r, field).astype(float); v.flat[0] = np.nan; setattr(r, field, v)
                with self.assertRaises(ValueError):
                    pilot.validate_result(r, 3)

    def test_clock_quaternion_angles_integer_counts(self):
        for change in (lambda r: setattr(r, "fps", 30), lambda r: r.qpos.__setitem__((0, 3), 0),
                       lambda r: r.frame_indices.__setitem__(0, 1), lambda r: r.normal_error.__setitem__(0, 4),
                       lambda r: setattr(r, "floor_rows", np.zeros(3)),
                       lambda r: r.point_error.__setitem__(0, -1)):
            with self.subTest(change=change):
                r = result(); change(r)
                with self.assertRaises(ValueError):
                    pilot.validate_result(r, 3)

    def test_exact_cached_arrays_require_dtype_shape_and_values(self):
        good = np.arange(3, dtype=np.int64)
        pilot.exact_array(good, good.copy(), "good")
        for bad in (good.astype(np.float64), good[:2], good+1):
            with self.assertRaises(ValueError):
                pilot.exact_array(bad, good, "bad")


class AuditTests(unittest.TestCase):
    def test_warmup_separate_from_zero_output_failures(self):
        pilot.validate_execution_audit(audit(), "both", 3, 0)

    def test_reject_counter_and_sequence_mismatches(self):
        changes = (lambda a: a.__setitem__("arm", "control"), lambda a: a.__setitem__("result", "RUNNING"),
                   lambda a: a.__setitem__("total_observed_failures", 0),
                   lambda a: a.__setitem__("output_failures", 1),
                   lambda a: a.__setitem__("rate_enabled", False),
                   lambda a: a.__setitem__("wrist_tasks", []),
                   lambda a: a.__setitem__("postprocessing_performed", True),
                   lambda a: a["frames"][0].__setitem__("frame", 1),
                   lambda a: a["frames"][0].__setitem__("failures", -1),
                   lambda a: a["frames"][0].__setitem__("warmup", True),
                   lambda a: a["warmup"].__setitem__("frame", 1),
                   lambda a: a.__setitem__("warmup_failures", 0))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                a = audit(); change(a); pilot.validate_execution_audit(a, "both", 3, 0)


class BoundaryTests(unittest.TestCase):
    def test_unit_rejects_wrong_and_unbounded(self):
        with patch.object(pilot.Path, "read_text", return_value="0::/other.service\n"):
            with self.assertRaises(ValueError):
                pilot.unit_guard("test")
        good = "KillMode=control-group\nRestart=no\nRuntimeMaxUSec=100s\nMemoryMax=120000\nTasksMax=128\n"
        with patch.object(pilot.Path, "read_text", return_value="0::/bfm-umr-stage2-test.service\n"):
            with patch.object(pilot.subprocess, "check_output", return_value=good):
                self.assertEqual(pilot.unit_guard("test"), "bfm-umr-stage2-test.service")
            with patch.object(pilot.subprocess, "check_output", return_value=good.replace("100s", "infinity")):
                with self.assertRaises(ValueError):
                    pilot.unit_guard("test")

    def test_cli_dry_run_does_not_execute(self):
        with patch.object(pilot, "build_plan", return_value=({"result": "PLAN_ONLY"}, None)), \
                patch.object(pilot, "execute") as execute, patch("builtins.print"):
            self.assertEqual(pilot.main(["--run-id", "example"]), 0)
            execute.assert_not_called()

    def test_cli_mixed_or_incomplete_modes_rejected(self):
        for args in ([], ["--worker"], ["--run-id", "a", "--arm", "both"],
                     ["--worker", "--run-id", "a", "--inputs", "x", "--inputs-sha256", "a", "--case", "0", "--arm", "both"]):
            with self.subTest(args=args), patch("sys.stderr"), self.assertRaises(SystemExit):
                pilot.main(args)


if __name__ == "__main__":
    unittest.main()

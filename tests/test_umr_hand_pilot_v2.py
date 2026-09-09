"""CPU-only controller contracts; no UMR, licensed data, GPU or physics."""
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from scripts import run_umr_hand_pilot_v2 as pilot


class SelectionTests(unittest.TestCase):
    def origins(self):
        return list(pilot.SELECTED) + [f"KIT/zzz/action_{i}" for i in range(13)]

    def test_fixed_selection_is_order_independent(self):
        self.assertEqual(pilot.select_origins(self.origins()[::-1]), pilot.SELECTED)

    def test_reject_subset(self):
        with self.assertRaises(ValueError):
            pilot.select_origins(self.origins()[:-1])

    def test_reject_duplicate(self):
        values = self.origins()
        values[-1] = values[0]
        with self.assertRaises(ValueError):
            pilot.select_origins(values)

    def test_reject_changed_first_or_source(self):
        for replacement in ("KIT/0/new", "CNRS/subject/action", "../unsafe/action"):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                pilot.select_origins(self.origins()[:-1] + [replacement])


class FilesTests(unittest.TestCase):
    def test_output_new_direct_child_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "local").mkdir()
            with patch.object(pilot, "ROOT", root):
                self.assertEqual(pilot.new_directory(root / "local" / "new"), root / "local" / "new")
                for p in (root / "local", root / "elsewhere", root / "local" / "nested" / "new"):
                    with self.subTest(path=p), self.assertRaises(ValueError):
                        pilot.new_directory(p)
                (root / "local" / "exists").mkdir()
                (root / "local" / "link").symlink_to(root / "missing")
                for p in (root / "local" / "exists", root / "local" / "link"):
                    with self.assertRaises(ValueError):
                        pilot.new_directory(p)

    def test_json_exclusive_and_owned_status_update(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "status.json"
            pilot.write_json(p, {"state": 1})
            with self.assertRaises(FileExistsError):
                pilot.write_json(p, {"state": 2})
            pilot.write_json(p, {"state": 3}, replace=True)
            self.assertEqual(json.loads(p.read_text()), {"state": 3})
            self.assertEqual(list(Path(d).iterdir()), [p])

    def test_json_nonfinite_and_missing_replacement_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "status.json"
            with self.assertRaises(ValueError):
                pilot.write_json(p, {"bad": float("nan")})
            with self.assertRaises(ValueError):
                pilot.write_json(p, {}, replace=True)
            self.assertFalse(p.exists())


class MathTests(unittest.TestCase):
    def test_native_comparison_sign_invariant(self):
        q = np.zeros((3, 36))
        q[:, 3] = 1
        r = q.copy()
        r[:, 3] = -1
        stats = pilot.compare_native(q, r)
        self.assertEqual(stats["root_orientation_deg"]["max"], 0)
        self.assertEqual(stats["joint_absolute_difference_rad"]["max"], 0)

    def test_native_comparison_actual_units(self):
        q = np.zeros((3, 36))
        q[:, 3] = 1
        r = q.copy()
        r[:, 0] = .1
        r[:, 7:] = .2
        stats = pilot.compare_native(q, r)
        self.assertAlmostEqual(stats["root_translation_m"]["mean"], .1)
        self.assertAlmostEqual(stats["joint_absolute_difference_rad"]["mean"], .2)

    def test_shape_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            pilot.compare_native(np.zeros((2, 36)), np.zeros((3, 36)))
        for value in ([], [float("nan")], [float("inf")]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.statistics(value)


class ProcessTests(unittest.TestCase):
    def test_actual_device_evidence_requires_completed_cuda(self):
        good = "[stage1] 训练 2500 epochs，设备 = cuda\n[stage1] 训练耗时 12.34s  device=cuda\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log"
            for content, valid in ((good, True), (good.replace("cuda", "cpu"), False),
                                   (good.splitlines()[0], False), (good.replace("2500", "20"), False),
                                   (good + good, False), ("[stage1] 复用 existing", False)):
                with self.subTest(content=content), patch.object(pilot.Path, "read_text", return_value=content), \
                        patch.object(pilot.reference, "sha256", return_value="a" * 64):
                    if valid:
                        self.assertEqual(pilot.stage_evidence(path)["resolved_device"], "cuda")
                    else:
                        with self.assertRaises(ValueError):
                            pilot.stage_evidence(path)

    def test_success_reaps_own_group(self):
        with tempfile.TemporaryDirectory() as d, patch.object(pilot.subprocess, "Popen") as popen, \
                patch.object(pilot.os, "killpg") as kill:
            process = popen.return_value
            process.pid, process.wait.return_value = 123456, 0
            pilot.run_child(["unused"], Path(d) / "log", 10)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(kill.call_args_list[0].args, (123456, signal.SIGTERM))
            self.assertEqual(kill.call_args_list[1].args, (123456, signal.SIGKILL))

    def test_timeout_and_nonzero_reap(self):
        for mode in ("timeout", "nonzero"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d, \
                    patch.object(pilot.subprocess, "Popen") as popen, patch.object(pilot.os, "killpg") as kill:
                process = popen.return_value
                process.pid = 123456
                process.wait.side_effect = [subprocess.TimeoutExpired("unused", 1) if mode == "timeout" else 3, 0, 0, 0]
                with self.assertRaises((subprocess.TimeoutExpired, RuntimeError)):
                    pilot.run_child(["unused"], Path(d) / "log", 1)
                self.assertEqual(kill.call_count, 2)

    def test_unit_rejects_wrong_group_before_systemctl(self):
        with patch.object(pilot.Path, "read_text", return_value="0::/user/other.service\n"), \
                patch.object(pilot.subprocess, "check_output") as query:
            with self.assertRaises(ValueError):
                pilot.bounded_unit("test")
            query.assert_not_called()

    def test_unit_exact_name_and_finite_bounds(self):
        text = "0::/user.slice/bfm-umr-hand-test.service\n"
        good = "KillMode=control-group\nRestart=no\nRuntimeMaxUSec=10s\nMemoryMax=12345\nTasksMax=128\n"
        with patch.object(pilot.Path, "read_text", return_value=text), \
                patch.object(pilot.subprocess, "check_output", return_value=good):
            self.assertEqual(pilot.bounded_unit("test"), "bfm-umr-hand-test.service")
        with patch.object(pilot.Path, "read_text", return_value=text), \
                patch.object(pilot.subprocess, "check_output", return_value=good.replace("12345", "infinity")):
            with self.assertRaises(ValueError):
                pilot.bounded_unit("test")


if __name__ == "__main__":
    unittest.main()

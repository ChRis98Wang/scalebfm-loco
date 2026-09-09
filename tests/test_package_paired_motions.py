"""Paired integer-clock regression tests; importing this loader never starts Kit."""
import builtins
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from scripts.umr_pair_dataset import package_interpolation

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "ScaleTrack/scripts/pretrain/data_process/package_paired_motions.py"
BATCH = ROOT / "local/umr_amass_pilot_20260908a"
PILOT_FRAME_COUNTS = (82, 86, 120, 131, 138, 145, 148, 149, 150, 155, 165, 167,
                      172, 175, 176, 186, 195, 201, 202, 205, 221, 227, 230, 251)


def load_entry():
    spec = importlib.util.spec_from_file_location("bfm_paired_clock_test_entry", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paired = load_entry()


def motion(frames):
    t = np.arange(frames, dtype=np.float64) / 50.
    return {"fps": 50., "root_pos": np.stack((t * t, t * .3, .8 + .02 * t * t), axis=1),
            "root_rot": np.tile([0., 0., 0., 1.], (frames, 1)),
            "dof_pos": t[:, None] ** 2 * np.linspace(.001, .1, 29)[None, :]}


def clock_interpolation(payload):
    """Independent use of clock output with the frozen float32 interpolation rule."""
    count = len(payload["root_pos"])
    duration, times = paired.paired_clock(count, payload["fps"], .02)
    phase = times / duration * (count - 1)
    lower = phase.floor().long()
    upper = torch.minimum(lower + 1, torch.tensor(count - 1))
    blend = phase - lower
    values = torch.from_numpy(np.concatenate((payload["root_pos"], payload["dof_pos"]), axis=1)).float()
    interpolated = values[lower] * (1 - blend[:, None]) + values[upper] * blend[:, None]
    return times.numpy(), interpolated.numpy()


def identity_runtime():
    # These tests exercise clocks/positions/derivatives, not Isaac quaternion
    # implementation. Identity root rotations make the angular result exact.
    return SimpleNamespace(
        RUNTIME_QUATERNION_ORDER="wxyz",
        source_xyzw_to_runtime=lambda value, order: value[:, [3, 0, 1, 2]],
        runtime_to_packed_wxyz=lambda value, order: value,
        quat_slerp=lambda left, right, blend: left,
        quat_mul=lambda left, right: torch.cat((torch.ones_like(left[:, :1]), torch.zeros_like(left[:, 1:])), dim=1),
        quat_conjugate=lambda value: value * torch.tensor([1., -1., -1., -1.]),
        axis_angle_from_quat=lambda value: torch.zeros_like(value[:, 1:]),
        relative_archive_path=lambda root, path: Path(path).relative_to(root),
    )


class PairedClockTests(unittest.TestCase):
    def test_every_bounded_source_length_has_exact_integer_half_open_length(self):
        for count in range(3, 252):
            with self.subTest(count=count):
                duration, times = paired.paired_clock(count, 50., .02)
                self.assertEqual(duration, (count - 1) / 50.)
                self.assertEqual(len(times), count - 1)
                self.assertEqual(times.dtype, torch.float32)
                self.assertEqual(float(times[0]), 0.)
                self.assertTrue(bool(torch.all(times[1:] > times[:-1])))
                self.assertLess(float(times[-1]), duration)
                reference = torch.arange(0, duration, .02, dtype=torch.float32)[:count - 1]
                self.assertTrue(torch.equal(times, reference))

    def test_202_and_227_samples_do_not_reintroduce_upward_product_rounding(self):
        for count in (202, 227):
            with self.subTest(count=count):
                legacy_duration = (count - 1) * (1. / 50.)
                duration, times = paired.paired_clock(count, 50., .02)
                self.assertGreater(legacy_duration, duration)
                self.assertEqual(len(torch.arange(0, legacy_duration, .02, dtype=torch.float32)), count)
                self.assertEqual(len(times), count - 1)

    def test_division_rounding_also_cannot_add_an_endpoint(self):
        for count in (8, 15, 29, 57, 112, 113, 223, 225, 248, 250):
            with self.subTest(count=count):
                _, times = paired.paired_clock(count, 50., .02)
                self.assertEqual(len(times), count - 1)

    def test_invalid_frame_counts_and_clocks_rejected(self):
        for count in (True, 0, 2, 252, 202., "202"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                paired.paired_clock(count, 50., .02)
        for fps in (True, 0., -50., 30., 49.999, float("nan"), float("inf"), "50", None):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                paired.paired_clock(202, fps, .02)
        for dt in (True, 0., -.02, 1. / 30., .0200001, float("nan"), float("inf"), None):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                paired.paired_clock(202, 50., dt)

    def test_all_current_pilot_lengths_match_frozen_proof_times_and_interpolation(self):
        for count in PILOT_FRAME_COUNTS:
            with self.subTest(count=count):
                payload = motion(count)
                frozen = package_interpolation(payload)
                times, values = clock_interpolation(payload)
                self.assertEqual(len(times), count - 1)
                np.testing.assert_array_equal(times, frozen["times"])
                np.testing.assert_array_equal(values[:, :3], frozen["root_pos"])
                np.testing.assert_array_equal(values[:, 3:], frozen["dof_pos"])

    @unittest.skipUnless((BATCH / "paired_manifest.json").is_file(), "Local licensed 40-origin pilot not present")
    def test_actual_forty_pairs_remain_compatible_with_frozen_time_value_proofs(self):
        import joblib
        manifest = json.loads((BATCH / "paired_manifest.json").read_text())
        self.assertEqual(len(manifest["rows"]), 40)
        for row in manifest["rows"]:
            for side in ("baseline", "candidate"):
                with self.subTest(origin=row["origin_id"], side=side):
                    payload = joblib.load(row[f"{side}_pkl"])
                    frozen = package_interpolation(payload)
                    times, values = clock_interpolation(payload)
                    self.assertEqual(len(times), row["expected_packed_frames"])
                    np.testing.assert_array_equal(times, frozen["times"])
                    np.testing.assert_array_equal(values[:, :3], frozen["root_pos"])
                    np.testing.assert_array_equal(values[:, 3:], frozen["dof_pos"])


class PairedLoaderTests(unittest.TestCase):
    def test_loader_interpolates_then_recomputes_all_velocity_arrays(self):
        runtime = identity_runtime()
        for count in (3, 202, 227, 251):
            with self.subTest(count=count), mock.patch("joblib.load", return_value=motion(count)), redirect_stdout(io.StringIO()):
                loaded = paired.load_paired_motion("/trusted", "/trusted/pilot.pkl", .02, runtime=runtime)
            expected = package_interpolation(motion(count))
            self.assertEqual(loaded["output_frames"], count - 1)
            np.testing.assert_array_equal(loaded["base_pos"].numpy(), expected["root_pos"])
            np.testing.assert_array_equal(loaded["dof_pos"].numpy(), expected["dof_pos"])
            for position, velocity in (("base_pos", "base_lin_vel"), ("dof_pos", "dof_vel")):
                torch.testing.assert_close(loaded[velocity], torch.gradient(loaded[position], spacing=.02, dim=0)[0], rtol=0, atol=0)
            self.assertEqual(tuple(loaded["base_ang_vel"].shape), (count - 1, 3))
            self.assertEqual(int(torch.count_nonzero(loaded["base_ang_vel"])), 0)
            self.assertEqual(loaded["file_name"], "pilot")
            self.assertEqual(loaded["source_path"], "/trusted/pilot.pkl")

    def test_import_does_not_import_legacy_entry_or_launch_isaac(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "package_motions" or name.split(".")[0] in ("isaaclab", "isaacsim", "omni"):
                raise AssertionError(f"Import unexpectedly starts simulator path: {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=guarded):
            imported = load_entry()
        self.assertTrue(callable(imported.paired_clock))
        self.assertTrue(callable(imported.load_paired_motion))


if __name__ == "__main__":
    unittest.main()

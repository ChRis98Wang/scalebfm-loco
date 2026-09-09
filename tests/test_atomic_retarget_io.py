from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib

from scaleretarget.utils import atomic_io


class AtomicRetargetIoTests(unittest.TestCase):
    def test_failed_dump_preserves_the_previous_retargeted_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "walk.pkl"
            joblib.dump({"known": "good"}, output)
            previous = output.read_bytes()

            def fail_after_partial_write(value, stream):
                del value
                stream.write(b"partial")
                raise RuntimeError("injected retarget dump failure")

            with mock.patch.object(
                atomic_io.joblib,
                "dump",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected retarget dump failure"):
                    atomic_io.atomic_joblib_dump({"new": "motion"}, output)

            self.assertEqual(output.read_bytes(), previous)
            self.assertEqual(list(output.parent.glob(".walk.pkl.*.tmp")), [])

    def test_publish_replaces_a_symlink_without_touching_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            victim = temp_path / "victim.pkl"
            victim.write_bytes(b"preserve-me")
            output = temp_path / "walk.pkl"
            output.symlink_to(victim)

            atomic_io.atomic_joblib_dump({"motion": [1, 2, 3]}, output)

            self.assertFalse(output.is_symlink())
            self.assertEqual(victim.read_bytes(), b"preserve-me")
            self.assertEqual(joblib.load(output), {"motion": [1, 2, 3]})


if __name__ == "__main__":
    unittest.main()

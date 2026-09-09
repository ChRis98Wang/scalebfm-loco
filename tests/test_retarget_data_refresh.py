from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import retarget_data_refresh as refresh
from scripts import amass_to_scalebfm as pipeline


def fake_indexes():
    result = {key: {} for key in refresh.INDEXES}
    for source in refresh.SOURCES:
        for split in ("train", "validation"):
            label = "train" if split == "train" else ("kit_validation" if source == "KIT" else "legacy_validation")
            for i, family in enumerate(("walk", "jump", "squat", "wave", "unknown", "run", "turn", "stand")):
                name = f"{source}/{split}/{family}_{i}"
                result[label][name] = f"/example/{name}.npz"
    return result


class PilotSelectionTest(unittest.TestCase):
    def test_balanced_reproducible_and_split_preserved(self):
        indexes = fake_indexes()
        rows = refresh.select_origins(indexes)
        self.assertEqual(rows, refresh.select_origins(indexes))
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(row["split"] == "train" for row in rows), 30)
        for source in refresh.SOURCES:
            self.assertEqual(sum(row["dataset"] == source for row in rows), 8)
        for row in rows:
            self.assertEqual(indexes[row["index_label"]][row["origin_id"]], row["baseline_packed"])

    def test_different_seed_changes_selected_origins(self):
        self.assertNotEqual(refresh.select_origins(fake_indexes(), seed=1), refresh.select_origins(fake_indexes(), seed=2))

    def test_train_dev_origin_overlap_rejected(self):
        indexes = fake_indexes()
        name, value = next(iter(indexes["train"].items()))
        indexes["legacy_validation"][name] = value
        with self.assertRaisesRegex(ValueError, "Duplicate origin"):
            refresh.select_origins(indexes)

    def test_traversal_rejected(self):
        indexes = fake_indexes()
        indexes["train"]["KIT/../../bad"] = "/bad.npz"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            refresh.select_origins(indexes)

    def test_insufficient_or_invalid_count_rejected(self):
        for count in (0, 10):
            with self.subTest(count=count), self.assertRaises(ValueError):
                refresh.select_origins(fake_indexes(), train_per_source=count)

    def test_same_source_retargeted_differently_cannot_leak_split(self):
        rows = [{"source_sha256": "a", "split": "train"}, {"source_sha256": "a", "split": "validation"}]
        with self.assertRaisesRegex(ValueError, "Same prepared AMASS"):
            refresh.validate_source_disjoint(rows)

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit):
            refresh.main(["--output", directory])

    def test_filename_hints_not_behavior_acceptance(self):
        self.assertEqual(refresh.action_bucket("KIT/unknown"), "unlabelled")
        self.assertEqual(refresh.action_bucket("ACCAD/jump"), "dynamic_hint")


class MotionContractTest(unittest.TestCase):
    def setUp(self):
        self.payload = {"root_pos": np.zeros((3, 3)), "root_rot": np.tile([0., 0., 0., 1.], (3, 1)),
                        "dof_pos": np.zeros((3, 29)), "fps": 50}

    def test_quaternion_order_and_no_mutation(self):
        result, fps = refresh.kinematic_qpos(self.payload)
        np.testing.assert_array_equal(result[:, 3:7], np.tile([1., 0., 0., 0.], (3, 1)))
        result[:] = 100
        self.assertEqual(self.payload["root_pos"].max(), 0)
        self.assertEqual(fps, 50)

    def test_bad_quaternion(self):
        self.payload["root_rot"][:] = 0
        with self.assertRaisesRegex(ValueError, "quaternion"):
            refresh.kinematic_qpos(self.payload)

    def test_bad_values_and_shapes(self):
        for key, value in (("fps", float("nan")), ("fps", -1), ("dof_pos", np.zeros((3, 28))),
                           ("root_pos", np.full((3, 3), float("inf")))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                refresh.kinematic_qpos({**self.payload, key: value})


class InputProofTest(unittest.TestCase):
    def test_chain_is_content_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "ScaleRetarget/retargeted_dataset"
            pkl = base / "batch/sub/motion.pkl"
            source = root / "ScaleRetarget/dataset/batch/sub/motion.npz"
            packed = base / "batch_processed/sub/motion.npz"
            for file in (pkl, source, packed):
                file.parent.mkdir(parents=True, exist_ok=True)
            pkl.write_bytes(b"trusted local test placeholder")
            source.write_bytes(b"source data")
            canonical = root / "ScaleRetarget/dataset/amass/KIT/sub/motion.npz"
            canonical.parent.mkdir(parents=True)
            canonical.write_bytes(source.read_bytes())
            pkl.with_suffix(".pkl.source.sha256").write_text(refresh.sha256(source))
            pkl.with_suffix(".pkl.pipeline.sha256").write_text("a" * 64)
            np.savez(packed, format_version=3, quaternion_order="wxyz", fps=50,
                     source_sha256=refresh.sha256(pkl), joint_pos=np.zeros((3, 29)))
            row = {"baseline_packed": str(packed), "origin_id": "KIT/sub/motion"}
            proof = refresh.resolve_inputs(row, root)
            self.assertEqual(proof["source_sha256"], refresh.sha256(source))
            with self.assertRaisesRegex(ValueError, "Origin identity"):
                refresh.resolve_inputs({**row, "origin_id": "ACCAD/incorrect-label"}, root)
            pkl.write_bytes(b"changed after packing")
            with self.assertRaisesRegex(ValueError, "provenance"):
                refresh.resolve_inputs(row, root)


class PipelineModeTest(unittest.TestCase):
    def test_opt_in_changes_argv_and_fingerprint(self):
        root = Path("/not-a-real-repo")
        options = pipeline.PipelineOptions(repo_root=root, source=root / "source", run_name="test",
            smplx_model=root / "smplx", retarget_python=Path("/python"), isaaclab_python=Path("/isaacpython"))
        alternative = replace(options, height_adjust_mode="collision_sole")
        self.assertNotEqual(pipeline.retarget_pipeline_fingerprint(options, "env"),
                            pipeline.retarget_pipeline_fingerprint(alternative, "env"))
        self.assertIn("formatter.config.height_adjust_mode=collision_sole", pipeline.build_command_plan(alternative)[0].argv)
        self.assertFalse(any("height_adjust_mode" in arg for arg in pipeline.build_command_plan(options)[0].argv))
        with self.assertRaises(ValueError):
            pipeline.build_command_plan(replace(options, height_adjust_mode="typo"))


if __name__ == "__main__":
    unittest.main()

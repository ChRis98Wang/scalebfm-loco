import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import run_umr_amass_pilot as pilot


class PilotControllerTest(unittest.TestCase):
    def test_plan_uses_existing_interpreters_and_no_train(self):
        row = {"origin_id": "KIT/1/a_stageii", "source": "/licensed/source.npz", "body_model": "/licensed/model.npz"}
        commands = pilot.command_plan(row, Path("/new/run"))
        self.assertEqual(commands["prepare"][0], str(pilot.RETARGET_PYTHON))
        self.assertEqual(commands["retarget"][0], str(pilot.UMR_PYTHON))
        self.assertIn("--setup-cache", commands["retarget"])
        self.assertIn("5.0", commands["prepare"])
        self.assertFalse(any("train.py" in str(value) for value in commands.values()))

    def test_existing_output_refused_before_input_reads(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(SystemExit):
            pilot.main(["--output", folder])

    def test_incomplete_selection_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "manifest.json"
            source.write_text(json.dumps({"schema": 1, "automatic_promotion": False, "motions": []}))
            with self.assertRaisesRegex(ValueError, "40-origin"):
                pilot.validate_selection(source)

    def test_owned_unit_guard(self):
        with mock.patch.object(Path, "read_text", return_value="0::/user.slice/unrelated.service"), self.assertRaises(ValueError):
            pilot.owned_unit()

    def test_pair_publishing_binds_files_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            pairs = output / "pairs/KIT/a"
            pairs.mkdir(parents=True)
            receipt = {"outputs": {}, "expected_packed_frames": 150}
            for side in ("baseline", "candidate"):
                file = pairs / f"{side}.pkl"
                file.write_bytes(side.encode())
                receipt["outputs"][side] = {"path": str(file), "sha256": pilot.sha256(file)}
            for name in ("pair_receipt.json", "sampling_proof.npz"):
                (pairs / name).write_bytes(b"proof")
            source = {"origin_id": "KIT/a", "split": "validation", "index_label": "kit_validation", "dataset": "KIT",
                      "source": "/raw/a.npz", "source_sha256": "a" * 64}
            row = pilot.publish_pair_row(source, receipt, output, pairs)
            self.assertEqual(row["expected_packed_frames"], 150)
            self.assertEqual(pilot.sha256(Path(row["baseline_pkl"])), receipt["outputs"]["baseline"]["sha256"])
            with self.assertRaises(FileExistsError):
                pilot.publish_pair_row(source, receipt, output, pairs)


if __name__ == "__main__":
    unittest.main()

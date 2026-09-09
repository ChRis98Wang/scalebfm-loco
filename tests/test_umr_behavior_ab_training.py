"""CPU-only tests for the finite two-arm training controller."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_umr_behavior_ab as driver


def audit(updates=6):
    origins = [f"KIT/origin_{i:03}" for i in range(256)]
    totals = dict.fromkeys(origins, 0)
    rows = []
    for i in range(updates):
        cohort = origins[(i // 5 % 2) * 128: (i // 5 % 2 + 1) * 128]
        counts = dict.fromkeys(cohort, 64)
        for name in cohort:
            totals[name] += 64
        rows.append({"motion_steps": counts, "completed_environment_steps": (i + 1) * 8192})
    return origins, {"result": "PASS", "completed_updates": updates,
                     "completed_environment_steps": updates * 8192,
                     "updates": rows, "motion_steps": totals,
                     "mode_steps": {str(i): updates * 1024 for i in range(8)},
                     "dataset_steps": {"KIT": updates * 8192}}


class TrainingContractTest(unittest.TestCase):
    def test_smoke_and_formal_real_coverage(self):
        for count in (6, 100):
            names, report = audit(count)
            result = driver.check_cohorts(report, count, names)
            self.assertEqual(result["actual_unique_origins"], 256)
            self.assertEqual(len(result["cohorts"]), 2 if count == 6 else 20)
            if count == 100:
                self.assertEqual(set(result["per_origin_steps"].values()), {3200})

    def test_reject_missing_or_fabricated_rollout(self):
        mutations = [
            lambda r: r.update(result="FAIL"),
            lambda r: r["updates"].pop(),
            lambda r: r["updates"][0]["motion_steps"].update({"KIT/foreign": 64}),
            lambda r: r["updates"][0]["motion_steps"].update({"KIT/origin_000": 63}),
            lambda r: r["updates"][0].update(completed_environment_steps=0),
            lambda r: r["updates"][1].update(motion_steps=r["updates"][5]["motion_steps"]),
            lambda r: r["updates"][5].update(motion_steps=r["updates"][0]["motion_steps"]),
            lambda r: r["motion_steps"].update({"KIT/origin_000": 1}),
            lambda r: r["mode_steps"].update({"0": 0}),
            lambda r: r.update(dataset_steps={"ACCAD": 49152}),
        ]
        for change in mutations:
            names, report = audit()
            change(report)
            with self.subTest(change=change), self.assertRaises(ValueError):
                driver.check_cohorts(report, 6, names)

    def test_jobs_fixed_budgets_arms_and_no_hydra_passthrough(self):
        for count in (6, 100):
            jobs = driver.build_jobs("/python", "/manifest.json", "fixed42", Path("/out"), count)
            self.assertEqual([row["arm"] for row in jobs], ["a", "b"])
            for row in jobs:
                command = row["command"]
                self.assertEqual(command[command.index("--updates") + 1], str(count))
                self.assertEqual(command[command.index("--dataset") + 1], "/manifest.json")
                self.assertNotIn("--test_motion_file", command)
                self.assertNotIn("--", command)
                self.assertEqual(command[command.index("--run-name") + 1], row["run_name"])
        with self.assertRaises(ValueError):
            driver.build_jobs("/python", "/manifest", "bad", "/out", 101)

    def test_dry_run_does_not_read_dataset_or_create_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from io import StringIO
            stream = StringIO()
            with patch.object(driver, "ROOT", root), patch.object(sys, "argv", [
                    "driver", "--dataset", str(root / "absent.json"), "--run-id", "plan"]), \
                    patch("sys.stdout", stream):
                driver.main()
            result = json.loads(stream.getvalue())
            self.assertFalse(result["automatic_promotion"])
            self.assertEqual(len(result["jobs"]), 2)
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_or_symlink_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                driver.new_directory(root)
            (root / "alias").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                driver.new_directory(root / "alias/new")
            self.assertEqual(driver.new_directory(root / "new"), root / "new")

    def test_atomic_status_finite_and_no_stale_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            driver.atomic_json(path, {"result": "RUNNING"})
            driver.atomic_json(path, {"result": "COMPLETE"})
            self.assertEqual(json.loads(path.read_text()), {"result": "COMPLETE"})
            path.with_name("status.json.tmp").write_text("preserve")
            with self.assertRaises(FileExistsError):
                driver.atomic_json(path, {"result": "FAIL"})

    def test_smoke_proof_requires_both_arms_and_identical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names, report = audit()
            frozen = {"ordered_origins": names, "runtime": "frozen"}
            proof = {"schema": driver.SCHEMA, "result": "COMPLETE", "updates_requested": 6,
                     "inputs_verified_unchanged": True, "matched_cohorts": True,
                     "frozen_inputs": frozen, "arms": {}}
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"test artifact, not a real model")
            for arm in ("a", "b"):
                file = root / f"{arm}_train_audit.json"
                file.write_text(json.dumps(report))
                proof["arms"][arm] = {"result": "PASS", "returncode": 0,
                                      "audit_sha256": driver.sha256(file),
                                      "final_checkpoint": str(checkpoint),
                                      "final_checkpoint_sha256": driver.sha256(checkpoint)}
            driver.validate_smoke(proof, frozen, root)
            changes = [lambda p: p.update(matched_cohorts=False),
                       lambda p: p.update(updates_requested=5),
                       lambda p: p["arms"].pop("b"),
                       lambda p: p["arms"]["a"].update(returncode=1),
                       lambda p: p["frozen_inputs"].update(runtime="changed")]
            for change in changes:
                changed = copy.deepcopy(proof)
                change(changed)
                with self.assertRaises(ValueError):
                    driver.validate_smoke(changed, frozen, root)
            checkpoint.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                driver.validate_smoke(proof, frozen, root)

    def test_child_is_reaped_on_success_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(driver.run_child([sys.executable, "-c", "print('finite')"], root / "ok.log", 10), 0)
            with self.assertRaises(subprocess.TimeoutExpired):
                driver.run_child([sys.executable, "-c", "import time; time.sleep(20)"], root / "timeout.log", .02)


if __name__ == "__main__":
    unittest.main()

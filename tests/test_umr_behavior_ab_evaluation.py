"""Pure command-plan and development-evidence checks, without simulator imports."""
import copy
from pathlib import Path
import unittest

from scripts import evaluate_umr_behavior_ab as driver
from scripts.compare_mask_evaluations import G1_BFM_MODE_PRESETS, MASK_METRIC_KEYS


class EvaluationContractTest(unittest.TestCase):
    def test_complete_plan_is_32_development_then_24_fixed_benchmark(self):
        manifest = {"heldout": {"index": "/heldout.yaml"}, "development": {"indices": {
            side: {"index": f"/{side}.yaml"} for side in ("baseline", "candidate")}}}
        checkpoints = {label: f"/{label}.pt" for label in ("official", "a", "b")}
        jobs = driver.build_jobs("/python", Path("/new"), manifest, checkpoints, "all")
        self.assertEqual(len(jobs), 56)
        self.assertEqual(len({job["name"] for job in jobs}), 56)
        self.assertTrue(all(job["reference"].startswith("development") for job in jobs[:32]))
        for job in jobs:
            command = job["command"]
            def value(key):
                self.assertEqual(command.count(key), 1)
                return command[command.index(key) + 1]
            self.assertEqual(value("--seed"), "42")
            self.assertEqual(value("--max_steps"), "1000")
            self.assertEqual(value("--num_envs"), "1024" if job["reference"] == "heldout" else "64")
            self.assertEqual(value("--checkpoint_path"), checkpoints[job["label"]])
            self.assertEqual(value("--motion_file"), job["index"])
            self.assertNotIn("--resume", command)
            self.assertTrue(command[2].endswith("evaluate.py"))
        self.assertEqual([job["label"] for job in jobs[32:35]], ["official", "a", "b"])
        self.assertEqual(len(driver.build_jobs("/python", "/new", manifest, checkpoints, "heldout")), 24)
        self.assertEqual(len(driver.build_jobs("/python", "/new", manifest, checkpoints, "development")), 32)

    def evidence(self):
        frames = {f"KIT/dev_{i}": 251 - i for i in range(10)}
        mode, bodies = G1_BFM_MODE_PRESETS[3]
        report = {"schema_version": 4, "mode_index": 3, "mode": mode,
                  "active_body_names": list(bodies), "task": "G1-BFM-Transformer-Tracking",
                  "num_envs": 64, "max_steps": 1000, "seed": 42, "device": "cuda:0", "step_dt": .02,
                  "scene_variant": "baseline", "target_object": None, "checkpoint_sha256": "a" * 64,
                  "input_manifest_verified_unchanged": True,
                  "protocol": {"training_updates": 0, "observation_noise": False,
                               "reset_disturbance": False, "interval_pushes": False},
                  "summary": {"num_motions": 10, "evaluated_steps": sum(frames.values()) - 10, "truncated_motions": 0},
                  "motions": [{"motion_id": i, "motion": name, "source_frames": count,
                               "evaluated_steps": count - 1, "truncated": False,
                               "metrics": {key: {"mean": .1, "max": .2} for key in MASK_METRIC_KEYS}}
                              for i, (name, count) in enumerate(frames.items())]}
        return frames, report

    def test_development_complete_windows(self):
        frames, report = self.evidence()
        result = driver.validate_development_evidence(report, {"name": "dev", "mode": 3}, "a" * 64, frames)
        self.assertEqual(result["num_motions"], 10)
        self.assertEqual(result["truncated_motions"], 0)

    def test_development_tampering_and_incomplete_coverage_rejected(self):
        frames, original = self.evidence()
        changes = [lambda r: r.update(seed=43), lambda r: r.update(num_envs=1024),
                   lambda r: r.update(checkpoint_sha256="b" * 64),
                   lambda r: r["protocol"].update(training_updates=1),
                   lambda r: r["protocol"].update(interval_pushes=True),
                   lambda r: r["motions"].pop(),
                   lambda r: r["motions"][0].update(source_frames=252, evaluated_steps=251),
                   lambda r: r["summary"].update(evaluated_steps=10)]
        for change in changes:
            report = copy.deepcopy(original)
            change(report)
            with self.assertRaises(ValueError):
                driver.validate_development_evidence(report, {"name": "dev", "mode": 3}, "a" * 64, frames)


if __name__ == "__main__":
    unittest.main()

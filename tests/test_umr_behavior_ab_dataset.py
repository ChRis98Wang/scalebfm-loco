"""CPU-only data-contract tests; no simulator, checkpoints or training jobs."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from scripts import build_umr_behavior_ab as builder
from tests.test_evaluate_umr_pairs import archive


def original_fixture(root, origin="KIT/actor/walk", content=b"source-human-one"):
    base = root / "ScaleRetarget/retargeted_dataset"
    native = base / "batch/actor/walk.pkl"
    prepared = root / "ScaleRetarget/dataset/batch/actor/walk.npz"
    canonical = root / "ScaleRetarget/dataset/amass" / f"{origin}.npz"
    packed = base / "batch_processed/actor/walk.npz"
    for path in (native, prepared, canonical, packed):
        path.parent.mkdir(parents=True, exist_ok=True)
    native.write_bytes(b"native trusted local bytes")
    canonical.write_bytes(content)
    prepared.symlink_to(canonical)
    native.with_suffix(".pkl.source.sha256").write_text(builder.digest_bytes(content))
    native.with_suffix(".pkl.pipeline.sha256").write_text("a" * 64)
    np.savez(packed, **archive(builder.digest_bytes(native.read_bytes())))
    return packed, native, prepared, canonical


def fk_fixture():
    rows, records = [], []
    for i in range(40):
        origin = f"KIT/person/motion{i}"
        row = {"origin_id": origin, "dataset": "KIT", "split": "train", "packed_frames": 10}
        for side in builder.pairs.SIDES:
            path = f"/a/{side}/{origin}.npz"
            digest = builder.digest_bytes(path.encode())
            row.update({f"{side}_packed": path, f"{side}_packed_sha256": digest})
            records.append({"origin_id": origin, "side": side, "dataset": "KIT", "split": "train",
                "path": path, "sha256": digest, "frames": 10, "passed": True,
                "joint_names": list(builder.pairs.ARTICULATION_JOINT_NAMES), "body_names": list(builder.pairs.BODY_NAMES),
                "position_error_m": {"max": 1e-6, "p95": 1e-7, "rms": 1e-8},
                "rotation_error_rad": {"max": 1e-6, "p95": 1e-7, "rms": 1e-8}})
        rows.append(row)
    validated = {"experiment_id": "fixed", "packaging_fingerprint": "p", "rows": rows}
    report = {"schema": "bfm.umr_packed_body_fk/1", "result": "PASS", "experiment_id": "fixed",
        "manifest_sha256": "m", "geometry_tree_sha256": "g", "packaging_fingerprint": "p",
        "thresholds": {"position_m": 1e-4, "rotation_rad": 1e-4}, "inputs_verified_unchanged": True,
        "physics_stepped": False, "training_updates": 0, "automatic_promotion": False,
        "archive_count": 80, "passed_archives": 80, "total_frames": 800, "results": records}
    return validated, report


class SourceContractTests(unittest.TestCase):
    def test_prepared_symlink_is_bound_to_real_canonical_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packed, _, prepared, canonical = original_fixture(root)
            files = builder.DataFiles()
            proof = builder.original_source_proof("KIT/actor/walk", packed, "train", files, root)
            self.assertEqual(proof["source"], str(canonical))
            self.assertEqual(files.aliases[str(prepared)], str(canonical))
            self.assertEqual(proof["original_frames"], 2)
            files.recheck()

    def test_self_consistent_wrong_source_cannot_be_relabelled_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packed, _, _, _ = original_fixture(root)
            alternate = root / "ScaleRetarget/dataset/amass/KIT/actor/heldout.npz"
            alternate.write_bytes(b"a different real human motion")
            with self.assertRaisesRegex(ValueError, "Canonical source identity"):
                builder.original_source_proof("KIT/actor/heldout", packed, "train", builder.DataFiles(), root)

    def test_changed_native_or_source_receipt_rejected(self):
        for kind in ("native", "receipt"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                packed, native, _, _ = original_fixture(root)
                if kind == "native":
                    native.write_bytes(b"new native but old packed SHA")
                else:
                    native.with_suffix(".pkl.source.sha256").write_text("0" * 64)
                with self.assertRaises(ValueError):
                    builder.original_source_proof("KIT/actor/walk", packed, "train", builder.DataFiles(), root)

    def test_symlink_retarget_after_hash_is_detected_even_for_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, prepared, canonical = original_fixture(root)
            files = builder.DataFiles()
            files.prepared_source(prepared)
            duplicate = root / "other.npz"
            duplicate.write_bytes(canonical.read_bytes())
            prepared.unlink()
            prepared.symlink_to(duplicate)
            with self.assertRaisesRegex(ValueError, "link changed"):
                files.recheck()

    def test_origin_and_real_source_payload_split_leakage(self):
        left = {key: "train_" + key for key in
                ("origin_id", "original_packed", "original_packed_sha256", "source_sha256", "native_sha256")}
        right = {key: "dev_" + key for key in left}
        builder.validate_source_separation([left], [right])
        for key in left:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "overlap"):
                builder.validate_source_separation([left], [{**right, key: left[key]}])


class SamplingAndEvidenceTests(unittest.TestCase):
    def test_actual_protocol_source_quotas_exclude_targets_before_allocation(self):
        counts = dict(ACCAD=197, BMLmovi=1528, BMLrub=2483, CNRS=39, KIT=2927)
        targets_per_source = dict(ACCAD=5, BMLmovi=5, BMLrub=3, CNRS=0, KIT=4)
        train = [f"{source}/person/{i:04}" for source, count in counts.items() for i in range(count)]
        targets = [f"{source}/person/{i:04}" for source, count in targets_per_source.items() for i in range(count)]
        selected, quotas = builder.replay_selection(train, targets)
        self.assertEqual(quotas, dict(ACCAD=6, BMLmovi=51, BMLrub=83, CNRS=1, KIT=98))
        self.assertEqual(len(selected), 239)
        self.assertFalse(set(selected) & set(targets))
        self.assertEqual((selected, quotas), builder.replay_selection(list(reversed(train)), list(reversed(targets))))

    def test_integer_remainder_and_hash_ties_are_lexical(self):
        train = ["KIT/x/c", "ACCAD/x/b", "KIT/x/a", "ACCAD/x/a"]
        with mock.patch.object(builder, "digest_bytes", return_value="same"):
            selected, quotas = builder.replay_selection(train, [], 1)
        self.assertEqual(selected, ["ACCAD/x/a"])
        self.assertEqual(quotas, {"ACCAD": 1, "KIT": 0})

    def test_bad_origins_and_targets_rejected(self):
        for origin in ("/KIT/x", "KIT/../x", "KIT//x", "KIT/x\n", "KIT\\x", "Other/x"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                builder.origin_name(origin)
        with self.assertRaises(ValueError):
            builder.replay_selection(["KIT/a"], ["KIT/dev"], 1)

    def test_duplicate_keys_and_weighted_yaml_rejected(self):
        for content in ("KIT/a: /a\nKIT/a: /b\n", "KIT/a: {path: /a, weight: 2}\n", "KIT/a: relative.npz\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "index.yaml"
                path.write_text(content)
                with self.assertRaises(ValueError):
                    builder.read_index(path, builder.DataFiles())

    def test_fk_full_archive_and_numeric_proof(self):
        validated, report = fk_fixture()
        builder.validate_fk(report, validated, "g", "m")
        changes = {
            "missing": lambda r: r["results"].pop(),
            "duplicate": lambda r: r["results"].__setitem__(1, copy.deepcopy(r["results"][0])),
            "side": lambda r: r["results"][0].update(side="different"),
            "sha": lambda r: r["results"][0].update(sha256="changed"),
            "frames": lambda r: r["results"][0].update(frames=9),
            "names": lambda r: r["results"][0]["joint_names"].reverse(),
            "nan": lambda r: r["results"][0]["position_error_m"].update(max=float("nan")),
            "bad_error": lambda r: r["results"][0]["rotation_error_rad"].update(max=.1),
            "geometry": lambda r: r.update(geometry_tree_sha256="other"),
            "threshold": lambda r: r.update(thresholds={"position_m": 1., "rotation_rad": 1.}),
            "updates_bool": lambda r: r.update(training_updates=False),
        }
        for name, change in changes.items():
            bad = copy.deepcopy(report)
            change(bad)
            with self.subTest(name=name), self.assertRaises(ValueError):
                builder.validate_fk(bad, validated, "g", "m")

    def test_integer_clock_receipt_binds_packer_and_prevents_posthoc_cropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            packer = Path(tmp) / "packer.py"
            packer.write_text("source")
            identity = {"schema": "bfm.paired_integer_half_open_clock/1", "parent_pipeline": "a" * 64,
                "entrypoint_sha256": builder.digest_bytes(packer.read_bytes()),
                "rule": "50 Hz inclusive N native samples -> N-1 packed samples",
                "velocity_rule": "recompute all derivatives before simulator link-state generation"}
            fingerprint = builder.digest_bytes(json.dumps(identity, sort_keys=True).encode())
            receipt = {**identity, "pipeline_fingerprint": fingerprint, "result": "COMPLETE",
                       "original_packer_modified": False, "posthoc_archive_cropping": False}
            with mock.patch.object(builder, "PACKER", packer):
                builder.validate_clock(receipt, fingerprint, builder.DataFiles())
                for field, value in (("posthoc_archive_cropping", True), ("entrypoint_sha256", "b" * 64),
                                     ("pipeline_fingerprint", "wrong"), ("result", "ERROR")):
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        builder.validate_clock({**receipt, field: value}, fingerprint, builder.DataFiles())

    def test_checkpoint_identity_is_content_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            official, derived = Path(tmp) / "official.pt", Path(tmp) / "derived.pt"
            official.write_bytes(b"official"); derived.write_bytes(b"derived")
            with mock.patch.multiple(builder, OFFICIAL=official, CHECKPOINT=derived,
                 OFFICIAL_SHA=builder.digest_bytes(b"official"), CHECKPOINT_SHA=builder.digest_bytes(b"derived")):
                self.assertEqual(builder.checkpoint_proof(builder.DataFiles())["initial_iteration"], 22199)
                derived.write_bytes(b"other checkpoint named derived")
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    builder.checkpoint_proof(builder.DataFiles())


class PublicationTests(unittest.TestCase):
    def test_dryrun_creates_nothing_and_execute_is_new_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new"
            manifest = {"schema": builder.SCHEMA, "input_sha256": {}, "counts": {}, "arms": {}}
            content = {key: "KIT/a: /payload.npz\n" for key in builder.FILENAMES}
            with mock.patch.object(builder, "make_plan", return_value=(manifest, content)):
                self.assertEqual(builder.build(output), manifest)
                self.assertFalse(output.exists())
                builder.build(output, execute=True)
                self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
                self.assertEqual(set(p.name for p in output.iterdir()), {*builder.FILENAMES.values(), "manifest.json"})
                with self.assertRaises(FileExistsError):
                    builder.build(output, execute=True)

    def test_failed_preflight_and_redirected_output_cannot_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(builder, "make_plan", side_effect=ValueError("source changed")):
                with self.assertRaises(ValueError):
                    builder.build(root / "new", execute=True)
                self.assertFalse((root / "new").exists())
            (root / "linked").symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                builder.output_location(root / "linked/new", require_new=True)
            with self.assertRaisesRegex(ValueError, "separate"):
                builder.output_location(builder.refresh.DATA / "new_ab", require_new=True)

    def test_manifest_and_generated_yaml_are_reconstructed_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"schema": builder.SCHEMA, "target_origins": ["KIT/train"]}
            content = {key: "KIT/train: /good.npz\n" for key in builder.FILENAMES}
            for key, value in content.items():
                (root / builder.FILENAMES[key]).write_text(value)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            with mock.patch.object(builder, "make_plan", return_value=(manifest, content)):
                self.assertEqual(builder.validate_manifest(path), manifest)
                (root / "train_b.yaml").write_text("KIT/train: /dev_with_changed_sha.npz\n")
                with self.assertRaises(ValueError):
                    builder.validate_manifest(path)
                (root / "train_b.yaml").write_text(content["b"])
                path.write_text(json.dumps({**manifest, "target_origins": ["KIT/dev"]}))
                with self.assertRaisesRegex(ValueError, "reconstructed"):
                    builder.validate_manifest(path)


if __name__ == "__main__":
    unittest.main()

"""Synthetic CPU tests; no licensed model, SMPL-X runtime, GPU or physics."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import umr_hand_reference_v2 as v2


def fixture(height_changes=False):
    core = v2.load_frozen_core(v2.FrozenInputs())
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 2], [1, 5, 4]], dtype=np.int64)
    vertices = np.array([[0., 0., 0.], [0., 1.6, 0.], [.3, .7, .2],
                         [-.3, .7, .2], [.2, 1., -.2], [-.2, 1., -.2]], dtype=np.float32)
    parents = np.array([-1] + [0] * 54, dtype=np.int64)
    skinning = np.zeros((6, 55), dtype=np.float32)
    skinning[:, 0] = 1
    calls = []
    def evaluate(hand):
        hand = np.array(hand, copy=True)
        calls.append(hand)
        v = vertices.copy()
        v[3, 0] += .1 * hand[0]
        if height_changes:
            v[1, 1] += .5 * hand[0]
        j = np.zeros((55, 3), dtype=np.float32)
        j[:, 1] = .8
        j[25, 0] = .02 * hand[0]
        r = np.tile(np.eye(3), (55, 1, 1))
        c, s = np.cos(hand[0]), np.sin(hand[0])
        r[25] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        return {"vertices": v, "joints": j, "joint_rotations": r}
    model = {"faces": faces, "parents": parents, "skinning_weights": skinning, "evaluate": evaluate}
    n, points = 2, 512
    parent = {
        "face_indices": np.arange(points, dtype=np.int64) % len(faces),
        "barycentric": np.tile([.2, .3, .5], (points, 1)),
        "segment": np.zeros(points, dtype=np.int64), "binding_joint_ids": np.zeros(points, dtype=np.int64),
        "parents": parents.copy(), "betas": np.zeros(16, dtype=np.float32),
        "sequence_points": np.arange(n * points * 3, dtype=np.float32).reshape(n, points, 3) / 1e4,
        "sequence_normals": np.tile([0., 0., 1.], (n, points, 1)).astype(np.float32),
        "joint_positions": np.zeros((n, 55, 3), dtype=np.float32),
        "joint_rotations": np.tile(np.eye(3), (n, 55, 1, 1)).astype(np.float32),
        "times": np.array([0., .02]), "foot_low": np.array([.01, .02]),
        "sample_lower": np.array([0, 1]), "sample_upper": np.array([0, 1]),
        "sample_alpha": np.zeros(n), "fps": np.array(50.), "segment_names": np.array(core.SEGMENTS),
    }
    parent.update(v2.canonical_geometry(evaluate(np.zeros(90)), model, parent, core)[0])
    metadata = {"schema": core.SCHEMA, "frames": n, "points": points,
                "shape_policy": "source_betas_all_static", "flat_hand_mean": True,
                "expression": "zero", "canonical_rotation": core.CANONICAL_ROTATION.tolist(),
                "source_gender": "neutral", "num_betas": 16, "actor_height": float(parent["actor_height"]),
                "adapter_sha256": v2.CORE_SHA256, "sampling_seed": 0}
    parent["metadata"] = metadata
    parent["metadata_json"] = np.asarray(json.dumps(metadata))
    raw = {"root_orient": np.zeros((3, 3)), "pose_body": np.zeros((3, 63)), "trans": np.zeros((3, 3)),
           "pose_hand": np.tile(np.linspace(.1, .5, 90), (3, 1)), "betas": parent["betas"].copy(),
           "gender": "neutral", "surface_model_type": "smplx", "mocap_frame_rate": 50.}
    calls.clear()
    return parent, raw, model, core, calls


class HandContractTests(unittest.TestCase):
    def test_exact_constant_full_source_is_accepted_without_discarding_components(self):
        _, raw, _, _, _ = fixture()
        np.testing.assert_array_equal(v2.constant_source_hand(raw), raw["pose_hand"][0])

    def test_dynamic_missing_nonfinite_and_bad_hand_shapes_fail_closed(self):
        _, raw, _, _, _ = fixture()
        bad = []
        missing = copy.deepcopy(raw); missing.pop("pose_hand"); bad.append(missing)
        for value in (np.zeros((3, 89)), np.full((3, 90), np.nan), np.zeros((2, 90))):
            altered = copy.deepcopy(raw); altered["pose_hand"] = value; bad.append(altered)
        dynamic = copy.deepcopy(raw); dynamic["pose_hand"][-1, -1] += 1e-12; bad.append(dynamic)
        for item in bad:
            with self.subTest(keys=item.keys()), self.assertRaises(ValueError):
                v2.constant_source_hand(item)

    def test_only_canonical_arrays_change_and_model_sees_exact_hand(self):
        parent, raw, model, core, calls = fixture()
        before = {k: v2.array_fingerprint(v) for k, v in parent.items() if k != "metadata"}
        # Explicitly prove the v2 transform never calls a point sampler.
        with mock.patch.object(core, "sample_canonical_surface", side_effect=AssertionError("resampling forbidden")):
            arrays, proof = v2.build_counterfactual(parent, raw, model, core)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0], np.zeros(90))
        np.testing.assert_array_equal(calls[1], raw["pose_hand"][0])
        self.assertTrue(proof["moving_arrays_unchanged"])
        self.assertFalse(proof["material_samples_resampled"])
        self.assertTrue(proof["downstream_scale_unchanged"])
        self.assertIn("canonical_points", proof["changed_canonical_arrays"])
        self.assertIn("canonical_joint_rotations", proof["changed_canonical_arrays"])
        for k in proof["protected_arrays"]:
            self.assertEqual(v2.array_fingerprint(arrays[k]), before[k])
        for k, fingerprint in before.items():
            self.assertEqual(v2.array_fingerprint(parent[k]), fingerprint)

    def test_each_nonreproduced_flat_canonical_field_is_rejected(self):
        for key in v2.CANONICAL_FIELDS:
            parent, raw, model, core, _ = fixture()
            parent[key] = np.array(parent[key], copy=True)
            parent[key].flat[0] += .001
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "reproduce frozen"):
                v2.build_counterfactual(parent, raw, model, core)

    def test_shape_gender_and_canonical_convention_mismatches_are_rejected(self):
        changes = [lambda p, r: r["betas"].__setitem__(0, 1.),
                   lambda p, r: r.update(gender="male"),
                   lambda p, r: p["metadata"].update(flat_hand_mean=False),
                   lambda p, r: p["metadata"].update(expression="nonzero"),
                   lambda p, r: p["metadata"].update(actor_height=1.2)]
        for change in changes:
            parent, raw, model, core, _ = fixture(); change(parent, raw)
            with self.assertRaises(ValueError):
                v2.build_counterfactual(parent, raw, model, core)

    def test_material_binding_and_hierarchy_are_not_silently_recomputed(self):
        for key in ("segment", "binding_joint_ids", "parents"):
            parent, raw, model, core, _ = fixture()
            parent[key][1] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                v2.build_counterfactual(parent, raw, model, core)

    def test_invalid_face_or_normal_geometry_is_rejected(self):
        parent, _, model, core, _ = fixture()
        parent["face_indices"][0] = 9999
        with self.assertRaises(ValueError):
            v2.canonical_geometry(model["evaluate"](np.zeros(90)), model, parent, core)
        parent, _, model, core, _ = fixture()
        evaluated = model["evaluate"](np.zeros(90)); evaluated["joint_rotations"][0] *= 2
        with self.assertRaises(ValueError):
            v2.canonical_geometry(evaluated, model, parent, core)

    def test_real_height_change_is_recorded_as_downstream_scaling_confound(self):
        parent, raw, model, core, _ = fixture(height_changes=True)
        arrays, proof = v2.build_counterfactual(parent, raw, model, core)
        self.assertFalse(proof["downstream_scale_unchanged"])
        self.assertGreater(proof["actor_height_change_m"], 0)
        self.assertAlmostEqual(proof["downstream_scale_ratio_v2_over_v1"],
                               float(parent["actor_height"]) / float(arrays["actor_height"]))
        np.testing.assert_array_equal(arrays["sequence_points"], parent["sequence_points"])


class FrozenInputTests(unittest.TestCase):
    def test_core_hash_is_verified_before_import(self):
        with mock.patch.object(v2, "CORE_SHA256", "0" * 64), mock.patch.object(v2.importlib.util, "spec_from_file_location") as load:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                v2.load_frozen_core(v2.FrozenInputs())
            load.assert_not_called()

    def test_byte_changes_and_alias_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, link = (Path(tmp) / n for n in ("a", "b", "link"))
            a.write_bytes(b"same"); b.write_bytes(b"same"); link.symlink_to(a)
            frozen = v2.FrozenInputs(); frozen.add(link, follow=True); frozen.recheck()
            link.unlink(); link.symlink_to(b)
            with self.assertRaisesRegex(ValueError, "alias changed"):
                frozen.recheck()
            frozen = v2.FrozenInputs(); frozen.add(a); a.write_bytes(b"else")
            with self.assertRaisesRegex(ValueError, "changed"):
                frozen.recheck()


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "local").mkdir()
        self.parent, self.raw, self.model, self.core, _ = fixture()
        self.source = self.root / "original_source.npz"
        self.body_model = self.root / "SMPLX_NEUTRAL.npz"
        self.body_model.write_bytes(b"FAKE MODEL: tests must never load licensed SMPLX")
        np.savez(self.source, **self.raw)
        metadata = self.parent["metadata"]
        metadata.update(source_file=str(self.source), source_sha256=v2.sha256(self.source),
                        body_model=str(self.body_model), body_model_sha256=v2.sha256(self.body_model))
        self.parent["metadata_json"] = np.asarray(json.dumps(metadata))
        self.parent_path = self.root / "parent.npz"
        np.savez(self.parent_path, **{k: value for k, value in self.parent.items() if k != "metadata"})
        self.parent_sha = v2.sha256(self.parent_path)
        self.output = self.root / "local" / "v2"
        self.patches = [mock.patch.object(v2, "ROOT", self.root),
                        mock.patch.object(v2, "dependency_snapshot", return_value={"versions": {"fixture": "1"}, "file_count": 0}),
                        mock.patch.object(v2, "make_cpu_model", return_value=self.model)]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def test_default_plan_creates_nothing_and_preserves_parent(self):
        before = set(self.root.rglob("*"))
        report = v2.prepare(self.parent_path, self.parent_sha, self.output)
        self.assertEqual(report["result"], "PLAN_ONLY")
        self.assertEqual(before, set(self.root.rglob("*")))
        self.assertEqual(v2.sha256(self.parent_path), self.parent_sha)
        self.assertFalse(report["retargeting_started"])

    def test_execute_is_loader_compatible_and_all_moving_values_survive(self):
        report = v2.prepare(self.parent_path, self.parent_sha, self.output, execute=True)
        self.assertEqual({p.name for p in self.output.iterdir()}, {"prepared_source.npz", "receipt.json"})
        prepared = self.core.load_prepared_source(self.output / "prepared_source.npz")
        for key in report["proof"]["protected_arrays"]:
            self.assertEqual(v2.array_fingerprint(prepared[key]), v2.array_fingerprint(self.parent[key]))
        metadata = prepared["metadata"]
        self.assertEqual(metadata["schema"], self.core.SCHEMA)
        self.assertEqual(metadata["parent_adapter_sha256"], v2.CORE_SHA256)
        self.assertEqual(metadata["adapter_sha256"], v2.sha256(v2.__file__))
        self.assertTrue(metadata["canonical_hand_experiment"]["post_hoc"])
        self.assertFalse(metadata["canonical_hand_experiment"]["demonstrated_bug_fix"])
        self.assertEqual(report["prepared_source_sha256"], v2.sha256(self.output / "prepared_source.npz"))

    def test_existing_directory_or_parent_alias_is_rejected_without_writes(self):
        self.output.mkdir(); (self.output / "sentinel").write_text("keep")
        with self.assertRaises(ValueError):
            v2.prepare(self.parent_path, self.parent_sha, self.output, execute=True)
        self.assertEqual((self.output / "sentinel").read_text(), "keep")
        alias = self.root / "local" / "alias"; alias.symlink_to(self.output, target_is_directory=True)
        with self.assertRaises(ValueError):
            v2.output_location(alias / "child")

    def test_outside_local_missing_parent_and_dangling_symlink_are_rejected(self):
        for output in (self.root / "outside", self.root / "local" / "missing" / "child"):
            with self.assertRaises(ValueError):
                v2.output_location(output)
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(ValueError):
            v2.output_location(self.output)

    def test_wrong_parent_hash_and_source_hash_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            v2.prepare(self.parent_path, "0" * 64, self.output)
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            v2.prepare(self.parent_path, self.parent_sha, self.output)
        self.assertFalse(self.output.exists())

    def test_dynamic_source_hand_fails_before_model_runtime(self):
        self.raw["pose_hand"][-1, 0] += .01
        np.savez(self.source, **self.raw)
        self.parent["metadata"]["source_sha256"] = v2.sha256(self.source)
        self.parent["metadata_json"] = np.asarray(json.dumps(self.parent["metadata"]))
        np.savez(self.parent_path, **{k: value for k, value in self.parent.items() if k != "metadata"})
        with mock.patch.object(v2, "make_cpu_model") as construct, self.assertRaisesRegex(ValueError, "Dynamic"):
            v2.prepare(self.parent_path, v2.sha256(self.parent_path), self.output)
        construct.assert_not_called()

    def test_dispatched_actual_model_file_must_have_the_original_model_hash(self):
        other = self.root / "SMPLX_NEUTRAL_other.npz"
        other.write_bytes(b"different model despite a plausible gender filename")
        with mock.patch.object(self.core, "model_file_for_gender", return_value=other), \
                mock.patch.object(v2, "load_frozen_core", return_value=self.core), \
                mock.patch.object(v2, "make_cpu_model") as construct:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                v2.prepare(self.parent_path, self.parent_sha, self.output)
        construct.assert_not_called()

    def test_cli_defaults_to_plan_and_stdout(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(v2.main(["--parent", str(self.parent_path), "--parent-sha256", self.parent_sha,
                                      "--output", str(self.output)]), 0)
        self.assertEqual(json.loads(stream.getvalue())["result"], "PLAN_ONLY")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

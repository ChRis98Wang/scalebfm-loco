#!/usr/bin/env python3
"""Build NEW, origin-preserving UMR A/B indexes; dry-run by default.

This CPU-only data contract never starts training, physics or IsaacLab. The
fixed 17 paired train windows replace references only in an isolated 256-origin
pool. Every development origin retains its original split. Existing indexes,
payloads, pilot evidence and checkpoints are read-only inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import evaluate_umr_pairs as pairs
from scripts import evaluate_retarget_pilot as archives
from scripts import retarget_data_refresh as refresh
from scripts.summarize_umr_pairs import VerifiedFiles, geometry_summary, validate_batch_status

SCHEMA = "bfm.umr_behavior_ab_dataset/1"
BATCH = ROOT / "local/umr_amass_pilot_20260908a"
EVALUATION = ROOT / "logs/behavior_learning/umr_amass_pairs_20260909a"
PACKED = BATCH / "packed_paired_clock"
PROTOCOL = ROOT / "docs/UMR_BEHAVIOR_AB_PROTOCOL_20260909.md"
PACKER = ROOT / "ScaleTrack/scripts/pretrain/data_process/package_paired_motions.py"
OFFICIAL = ROOT / "logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt"
CHECKPOINT = ROOT / "logs/rsl_rl/g1_bfm_tracking_exp/official_lr1e5_derived_20260907/model_22200.pt"
OFFICIAL_SHA = "88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3"
CHECKPOINT_SHA = "269f17e040ad0c27f651a25097a2650380ff1a05714dde102625aabd8d327c46"
PINS = {
    "manifest": "ab7b1ca1b4bd4f9244d6465b4306770534c7b50fbe4a8d71dea5898578819d07",
    "geometry": "9aba7833532d58e518cbaa69d079a5f6123d8ad4eedbd1639b76460a4f76f78b",
    "fk": "5d6bb5a7a306af4e460434b536492f3fa7956a9069626e1dc4b5a99a8ececc5f",
    "comparison": "c398ba5a38292528031b13819d9b8fca316ef3bbb005460d704594d92b8d3a8f",
}
INDEX_COUNTS = {"train": 7174, "legacy_validation": 962, "kit_validation": 789}
GEOMETRY_THRESHOLDS = {"foot_penetration_m": .001, "self_or_ground_penetration_m": .005,
                       "joint_limit_tolerance_rad": 1e-6}
REPLAY_SALT = "umr_behavior_ab_20260909/v1/replay/"
SAMPLING = {"strategy": "coverage", "weights": "uniform", "weight_per_origin": 1 / 256,
            "num_envs": 128, "resample_interval": 5, "steps_per_env": 64}
FILENAMES = {"a": "train_a.yaml", "b": "train_b.yaml", "baseline": "development_baseline.yaml",
             "candidate": "development_candidate.yaml", "heldout": "heldout_union.yaml"}


class DataFiles(VerifiedFiles):
    """Prepared AMASS legitimately uses symlinks; bind their resolved targets."""

    def __init__(self):
        super().__init__()
        self.aliases = {}

    def follow(self, path, expected=None):
        path = Path(path).absolute()
        resolved = path.resolve(strict=True)
        if str(path) in self.aliases and self.aliases[str(path)] != str(resolved):
            raise ValueError("Bound input link changed during validation")
        self.aliases[str(path)] = str(resolved)
        return self.verify(resolved, expected)

    prepared_source = follow

    def recheck(self):
        super().recheck()
        if any(str(Path(path).resolve(strict=True)) != target for path, target in self.aliases.items()):
            raise ValueError("Bound input link changed before publication")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def origin_name(value):
    if (not isinstance(value, str) or "\\" in value or any(ord(c) < 32 for c in value)
            or len(value.split("/")) < 2 or value.split("/")[0] not in refresh.SOURCES
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"Invalid canonical origin: {value!r}")
    return value


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError("Index contains a non-string or duplicate origin key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def read_index(path, files, expected=None):
    files.verify(path, expected)
    result = yaml.load(Path(path).read_text(), Loader=UniqueKeyLoader)
    if not isinstance(result, dict) or not result or any(not isinstance(v, str) for v in result.values()):
        raise ValueError("Index must contain only origin-to-path entries; weights/aliases are forbidden")
    for name, value in result.items():
        origin_name(name)
        if not Path(value).is_absolute():
            raise ValueError("Index payload paths must be explicit absolute paths")
    files.verify(path, expected)
    return result


def replay_selection(train_origins, targets, count=239):
    """Integer Hamilton allocation, with lexical ties and fixed origin hashes."""
    train, targets = set(train_origins), set(targets)
    if not targets <= train or type(count) is not int or count < 1:
        raise ValueError("Targets must be original train origins and replay count positive")
    groups = defaultdict(list)
    for name in train - targets:
        groups[origin_name(name).split("/", 1)[0]].append(name)
    total = sum(map(len, groups.values()))
    if count > total:
        raise ValueError("Insufficient unique train origins for replay")
    divisions = {source: divmod(count * len(names), total) for source, names in groups.items()}
    quotas = {source: values[0] for source, values in divisions.items()}
    order = sorted(groups, key=lambda source: (-divisions[source][1], source))
    for source in order[:count - sum(quotas.values())]:
        quotas[source] += 1
    selected = []
    for source in sorted(groups):
        names = sorted(groups[source], key=lambda name: (digest_bytes((REPLAY_SALT + name).encode()), name))
        selected.extend(names[:quotas[source]])
    return sorted(selected), dict(sorted(quotas.items()))


def original_source_proof(origin, packed, label, files, repo_root=ROOT):
    """Bind a YAML identity to real canonical AMASS bytes, not a renamed NPZ.

    Mirrors the frozen resolve_inputs contract, with a per-build hash cache so
    shared prepared/canonical paths and full heldout provenance are read once.
    Native pickle bytes are hashed here, never deserialized.
    """
    origin_name(origin)
    packed = Path(packed)
    packed_sha = files.verify(packed)
    packed = packed.resolve(strict=True)
    base = Path(repo_root) / "ScaleRetarget/retargeted_dataset"
    relative = packed.relative_to(base.resolve(strict=True))
    if len(relative.parts) < 2 or not relative.parts[0].endswith("_processed"):
        raise ValueError("Original payload is not in a canonical processed batch")
    batch, tail = relative.parts[0].removesuffix("_processed"), Path(*relative.parts[1:])
    native = (base / batch / tail).with_suffix(".pkl")
    prepared = (Path(repo_root) / "ScaleRetarget/dataset" / batch / tail).with_suffix(".npz")
    canonical = Path(repo_root) / "ScaleRetarget/dataset/amass" / f"{origin}.npz"
    native_sha, source_sha = files.verify(native), files.prepared_source(prepared)
    if files.verify(canonical) != source_sha:
        raise ValueError(f"Canonical source identity differs from prepared AMASS: {origin}")
    source_receipt = native.with_suffix(".pkl.source.sha256")
    pipeline_receipt = native.with_suffix(".pkl.pipeline.sha256")
    files.verify(source_receipt); files.verify(pipeline_receipt)
    if source_receipt.read_text().strip() != source_sha:
        raise ValueError("Native source receipt differs from real source bytes")
    pipeline = pipeline_receipt.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", pipeline):
        raise ValueError("Invalid original retarget pipeline fingerprint")
    with np.load(packed, allow_pickle=False) as payload:
        frames = len(payload["joint_pos"])
    if frames < 2:
        raise ValueError("Reference requires at least two packed frames")
    package = archives.validate_archive(packed, native_sha=native_sha, frames=frames)
    return {"origin_id": origin, "dataset": origin.split("/")[0], "index_label": label,
            "split": "train" if label == "train" else "validation",
            "original_packed": str(packed), "original_packed_sha256": packed_sha,
            "native": str(native.resolve()), "native_sha256": native_sha,
            "prepared_path": str(prepared.absolute()), "source": str(prepared.resolve()),
            "canonical_source": str(canonical.resolve()),
            "source_sha256": source_sha, "original_frames": frames,
            "retarget_fingerprint": pipeline, "packaging_fingerprint": package}


def validate_source_separation(train_rows, development_rows):
    for key in ("origin_id", "original_packed", "original_packed_sha256", "source_sha256", "native_sha256"):
        if {row[key] for row in train_rows} & {row[key] for row in development_rows}:
            raise ValueError(f"Train/development overlap in real {key}")


def validate_clock(receipt, packaging_fingerprint, files):
    expected = {"schema": "bfm.paired_integer_half_open_clock/1",
                "rule": "50 Hz inclusive N native samples -> N-1 packed samples",
                "velocity_rule": "recompute all derivatives before simulator link-state generation"}
    if (any(receipt.get(k) != v for k, v in expected.items()) or receipt.get("result") != "COMPLETE"
            or receipt.get("original_packer_modified") is not False
            or receipt.get("posthoc_archive_cropping") is not False):
        raise ValueError("Require the complete integer-clock, recomputed-velocity receipt")
    identity = {key: receipt[key] for key in (*expected, "parent_pipeline", "entrypoint_sha256")}
    if (files.verify(PACKER, receipt["entrypoint_sha256"]) != receipt["entrypoint_sha256"]
            or not re.fullmatch(r"[0-9a-f]{64}", receipt["parent_pipeline"])
            or digest_bytes(json.dumps(identity, sort_keys=True).encode()) != packaging_fingerprint
            or receipt.get("pipeline_fingerprint") != packaging_fingerprint):
        raise ValueError("Integer-clock fingerprint does not bind this packer and payloads")


def validate_fk(report, validated, geometry_tree, manifest_sha):
    if (report.get("schema") != "bfm.umr_packed_body_fk/1" or report.get("result") != "PASS"
            or report.get("experiment_id") != validated["experiment_id"]
            or report.get("manifest_sha256") != manifest_sha
            or report.get("geometry_tree_sha256") != geometry_tree
            or report.get("packaging_fingerprint") != validated["packaging_fingerprint"]
            or report.get("thresholds") != {"position_m": 1e-4, "rotation_rad": 1e-4}
            or report.get("inputs_verified_unchanged") is not True
            or report.get("physics_stepped") is not False or type(report.get("training_updates")) is not int
            or report["training_updates"] != 0 or report.get("automatic_promotion") is not False):
        raise ValueError("FK report does not bind the exact complete non-training pilot")
    expected = {(row["origin_id"], side): row for row in validated["rows"] for side in pairs.SIDES}
    records = report.get("results", [])
    seen = set()
    for record in records:
        key = (record.get("origin_id"), record.get("side"))
        if key not in expected or key in seen:
            raise ValueError("Missing, duplicated or foreign FK side/origin")
        seen.add(key)
        row, side = expected[key], key[1]
        if (record.get("path") != row[f"{side}_packed"] or record.get("sha256") != row[f"{side}_packed_sha256"]
                or record.get("frames") != row["packed_frames"] or record.get("split") != row["split"]
                or record.get("dataset") != row["dataset"]
                or record.get("joint_names") != list(pairs.ARTICULATION_JOINT_NAMES)
                or record.get("body_names") != list(pairs.BODY_NAMES)):
            raise ValueError("FK row differs from actual named packed input")
        for metric in ("position_error_m", "rotation_error_rad"):
            stats = record.get(metric, {})
            if set(stats) != {"max", "p95", "rms"} or any(
                isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                or v < 0 or v > 1e-4 for v in stats.values()
            ) or stats["p95"] > stats["max"] or stats["rms"] > stats["max"]:
                raise ValueError("FK numeric errors exceed fixed finite thresholds")
        if record.get("passed") is not True:
            raise ValueError("FK row did not pass")
    frames = sum(row["packed_frames"] * 2 for row in validated["rows"])
    if (seen != set(expected) or len(records) != 80 or report.get("archive_count") != 80
            or report.get("passed_archives") != 80 or report.get("total_frames") != frames):
        raise ValueError("FK report must cover all eighty archives and every frame")


def checkpoint_proof(files):
    files.follow(OFFICIAL, OFFICIAL_SHA); files.follow(CHECKPOINT, CHECKPOINT_SHA)
    # These pinned complete file hashes bind the already-audited LR-only derived
    # checkpoint, including both Adam states; no weight inference or mutation.
    return {"path": str(CHECKPOINT), "sha256": CHECKPOINT_SHA,
            "official_path": str(OFFICIAL), "official_sha256": OFFICIAL_SHA,
            "relationship": "audited official model and Adam states; only both LR fields changed to 1e-5",
            "initial_iteration": 22199, "learning_rate": 1e-5}


def load_evidence(files):
    manifest = files.read_json(BATCH / "paired_manifest.json", PINS["manifest"])
    status = files.read_json(BATCH / "status.json")
    validate_batch_status(status, manifest["rows"], PINS["manifest"])
    validated = pairs.validate_pairs(BATCH / "paired_manifest.json", PACKED)
    for path, digest in validated["frozen_files"].items():
        files.verify(path, digest)
    geometry = files.read_json(EVALUATION / "geometry_audit.json", PINS["geometry"])
    if geometry.get("thresholds") != GEOMETRY_THRESHOLDS:
        raise ValueError("Geometry thresholds differ from the frozen screening protocol")
    geometry_summary(geometry, validated["rows"])
    validated["rows"] = pairs.bind_geometry(validated, geometry)
    from scripts.amass_to_scalebfm import semantic_fingerprint
    geometry_tree = semantic_fingerprint({"robot_geometry": pairs.ROBOT_XML.parent}, {})
    if geometry.get("geometry_tree_sha256") != geometry_tree:
        raise ValueError("Robot geometry differs from the accepted packed audit")
    for path in sorted(pairs.ROBOT_XML.parent.rglob("*")):
        if path.is_file():
            files.verify(path)
    fk = files.read_json(BATCH / "body_fk_audit_20260909a.json", PINS["fk"])
    validate_fk(fk, validated, geometry_tree, PINS["manifest"])
    for path, digest in fk["input_sha256"].items():
        files.verify(path, digest)
    clock = files.read_json(PACKED / "paired_clock_receipt.json")
    validate_clock(clock, validated["packaging_fingerprint"], files)
    comparison = files.read_json(EVALUATION / "comparison.json", PINS["comparison"])
    if (comparison.get("checkpoint_sha256") != OFFICIAL_SHA or comparison.get("training_updates") != 0
            or comparison.get("geometry_audit_sha256") != PINS["geometry"]
            or comparison.get("experiment_id") != validated["experiment_id"]
            or comparison.get("geometry_rejects_excluded") != 0):
        raise ValueError("Frozen tracking report has a different checkpoint/data identity")
    # Tracking scores are deliberately not read when choosing train targets.
    passed = {row["origin_id"] for row in geometry["results"] if row["candidate"]["kinematic_pass"]}
    targets = sorted(row["origin_id"] for row in validated["rows"] if row["split"] == "train" and row["origin_id"] in passed)
    development = sorted(row["origin_id"] for row in validated["rows"] if row["split"] == "validation")
    if len(targets) != 17 or len(development) != 10 or len(passed & set(development)) != 7:
        raise ValueError("Expected exactly seventeen train targets and ten development origins (seven geometry passes)")
    return manifest, validated, targets, development, sorted(passed & set(development))


def output_location(output, *, require_new):
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Output cannot redirect through a symlink")
    protected = (refresh.DATA, ROOT / "ScaleRetarget/dataset", BATCH, EVALUATION,
                 OFFICIAL.parent, CHECKPOINT.parent)
    if any(output == path or path in output.parents or output in path.parents for path in protected):
        raise ValueError("Output must be separate from all original data/evidence/checkpoint directories")
    if require_new and output.exists():
        raise FileExistsError(f"Refusing existing dataset output: {output}")
    return output


def make_plan(output):
    """Return a fully verified manifest and exact YAML text, without writing."""
    output = output_location(output, require_new=False)
    files = DataFiles()
    for path in (Path(__file__).resolve(), PROTOCOL, Path(refresh.__file__).resolve()):
        files.verify(path)
    original, validated, targets, development, dev_pass = load_evidence(files)
    indexes, index_specs = {}, original["origin_indexes"]
    for label, count in INDEX_COUNTS.items():
        spec = index_specs[label]
        if Path(spec["path"]).resolve() != refresh.INDEXES[label].resolve():
            raise ValueError("Only the frozen original full-v2/validation indexes are eligible")
        indexes[label] = read_index(spec["path"], files, spec["sha256"])
        if len(indexes[label]) != count:
            raise ValueError(f"Incorrect original {label} count")
    train = indexes["train"]
    heldout = {**indexes["legacy_validation"], **indexes["kit_validation"]}
    if (len(heldout) != 1751 or set(train) & set(heldout) or not set(targets) <= set(train)
            or not set(development) <= set(heldout)):
        raise ValueError("Target/development origins moved across original split indexes")
    path_groups = [{str(Path(p).resolve()) for p in index.values()} for index in indexes.values()]
    if any(path_groups[i] & path_groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Original train/development payload paths overlap")
    replay, quotas = replay_selection(train, targets)
    ordered = sorted(targets + replay)
    rows_by_origin = {row["origin_id"]: row for row in validated["rows"]}
    provenance = []
    for name in ordered + sorted(heldout):
        label = "train" if name in train else "kit_validation" if name in indexes["kit_validation"] else "legacy_validation"
        proof = original_source_proof(name, indexes[label][name], label, files)
        if name in rows_by_origin and proof["source_sha256"] != rows_by_origin[name]["source_sha256"]:
            raise ValueError("Pilot source differs from the origin's original payload chain")
        provenance.append(proof)
    validate_source_separation(provenance[:256], provenance[256:])
    maps = {side: {name: (rows_by_origin[name][f"{reference}_packed"] if name in targets else train[name])
                   for name in ordered} for side, reference in (("a", "baseline"), ("b", "candidate"))}
    for side in pairs.SIDES:
        maps[side] = {name: rows_by_origin[name][f"{side}_packed"] for name in development}
    maps["heldout"] = {name: heldout[name] for name in sorted(heldout)}
    serialized = {name: yaml.safe_dump(index, sort_keys=False, allow_unicode=True) for name, index in maps.items()}
    records = {name: {"index": str(output / FILENAMES[name]), "sha256": digest_bytes(value.encode())}
               for name, value in serialized.items()}
    checkpoint = checkpoint_proof(files)
    target_rows = [{**rows_by_origin[name], "sampling_weight": 1.0} for name in targets]
    manifest = {"schema": SCHEMA, "result": "COMPLETE_DATASET_ONLY", "training_started": False,
        "training_updates": 0, "automatic_promotion": False, "physics_stepped": False,
        "protocol": str(PROTOCOL), "protocol_sha256": files.verify(PROTOCOL),
        "output_directory": str(output), "arms": {name: records[name] for name in ("a", "b")},
        "target_origins": targets, "replay_origins": replay, "ordered_origins": ordered,
        "target_rows": target_rows, "original_indexes": index_specs, "source_provenance": provenance,
        "development": {"origin_ids": development, "geometric_pass_origins": dev_pass,
                        "indices": {name: records[name] for name in pairs.SIDES}, "final_test": False},
        "heldout": {**records["heldout"], "counts": {"legacy": 962, "kit": 789}, "final_test": False},
        "checkpoint": checkpoint, "sampling": dict(SAMPLING),
        "selection": {"replay_salt": REPLAY_SALT, "allocation": "integer-largest-remainder; lexical dataset ties",
                      "replay_per_dataset": quotas, "ordering": "lexical origin_id", "scores_used": False},
        "counts": {"train": 256, "targets": 17, "replay": 239, "development": 10,
                   "development_geometric_pass": 7, "heldout": 1751,
                   "train_by_dataset": dict(sorted(Counter(n.split('/')[0] for n in ordered).items()))},
        "evidence": {"batch": str(BATCH), "evaluation": str(EVALUATION), "pins": dict(PINS),
                     "paired_packaging_fingerprint": validated["packaging_fingerprint"]},
        "limitations": ["Only matched <=5 second target windows; not whole-motion replacement.",
                        "Geometry eligibility and frozen tracking do not establish learning or task success.",
                        "Development and original validation have been used for development, not final testing.",
                        "Exact origin/path/payload/native/source separation checked; near duplicates and upstream overlap not excluded."],
        "input_sha256": {str(path): record[0] for path, record in sorted(files.files.items())},
        "input_stat_signatures": {str(path): list(record[1]) for path, record in sorted(files.files.items())},
        "resolved_input_aliases": dict(sorted(files.aliases.items())),
        "inputs_verified_unchanged": True}
    files.recheck()
    return manifest, serialized


def validate_manifest(path):
    """Rebuild the entire pinned contract and verify every published index."""
    path = Path(path).absolute()
    if path.name != "manifest.json" or path.is_symlink() or not path.is_file():
        raise ValueError("Require the regular published manifest.json")
    before = VerifiedFiles()
    actual = before.read_json(path)
    expected, serialized = make_plan(path.parent)
    if actual != expected:
        raise ValueError("Dataset manifest differs from the reconstructed frozen data contract")
    for name, content in serialized.items():
        index = path.parent / FILENAMES[name]
        before.verify(index, digest_bytes(content.encode()))
        if index.read_text() != content:
            raise ValueError("Published index differs from the fixed A/B ordering and payload mapping")
    before.recheck()
    return actual


load_manifest = validate_manifest


def build(output, *, execute=False):
    output = output_location(output, require_new=True)
    manifest, serialized = make_plan(output)
    if execute:
        # Hash/stat verification above finishes before creating any outputs.
        output.mkdir(parents=True, exist_ok=False)
        for name, content in serialized.items():
            with (output / FILENAMES[name]).open("x", encoding="utf-8") as stream:
                stream.write(content)
        for path, expected in manifest.get("input_stat_signatures", {}).items():
            if Path(path).is_symlink() or list(VerifiedFiles.signature(Path(path))) != expected:
                raise ValueError("Frozen input changed while publishing indexes; no success manifest written")
        for path, target in manifest.get("resolved_input_aliases", {}).items():
            if str(Path(path).resolve(strict=True)) != target:
                raise ValueError("Input link changed while publishing indexes; no success manifest written")
        # Publish the success manifest last; an interrupted build is never a
        # usable dataset and is retained for inspection, never overwritten.
        with (output / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Publish only NEW index/manifest files; never train")
    args = parser.parse_args(argv)
    manifest = build(args.output, execute=args.execute)
    print(json.dumps({"plan_only": not args.execute, "output": str(args.output.absolute()),
                      "counts": manifest["counts"], "arms": manifest["arms"],
                      "source_files_verified": len(manifest["input_sha256"]),
                      "training_updates": 0, "automatic_promotion": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

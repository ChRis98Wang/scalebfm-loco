#!/usr/bin/env python3
"""Export/verify a native ScaleBFM actor, without starting or upgrading IsaacLab.

Writes only a NEW output directory. Run under a bounded bfm-export-*.service.
All eight masks, two input seeds and independent FK/observation construction
are checked; a failing artifact is preserved for diagnosis, never promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ScaleBridge"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bounded_unit(prefix="bfm-export-"):
    found = re.findall(r"(?:^|/)(" + re.escape(prefix) + r"[A-Za-z0-9_.-]+\.service)(?:/|$)",
                       Path("/proc/self/cgroup").read_text(), flags=re.MULTILINE)
    if len(found) != 1:
        raise ValueError(f"Use a bounded {prefix}*.service user unit")
    values = dict(line.split("=", 1) for line in subprocess.check_output([
        "systemctl", "--user", "show", found[0], "-p", "KillMode", "-p", "Restart",
        "-p", "RuntimeMaxUSec", "-p", "MemoryMax", "-p", "TasksMax"], text=True).splitlines())
    if (values.get("KillMode") != "control-group" or values.get("Restart") != "no"
            or any(values.get(key) in (None, "", "0", "infinity")
                   for key in ("RuntimeMaxUSec", "MemoryMax", "TasksMax"))):
        raise ValueError("Finite time/memory/tasks bounds and control-group cleanup required")
    return found[0]


def export(checkpoint, metadata_path, xml, output):
    import torch
    from scalebridge.agent import portable_policy as portable

    output = Path(output).absolute()
    if any(parent.is_symlink() for parent in (output, *output.parents)):
        raise ValueError("Output cannot redirect through symlinks")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "bfm.portable_export/1", "result": "ERROR", "training_updates": 0,
              "automatic_promotion": False, "hardware_accessed": False, "physics_stepped": False,
              "isaac_runtime_observation_comparison": False, "runtime_backend": "torchscript",
              "thresholds": {"action_absolute": 1e-4, "pd_target_rad": 1e-4,
                             "observation_absolute": 2e-5}, "checks": []}
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt("Owned export service stopped")
    signal.signal(signal.SIGTERM, terminate)
    try:
        checkpoint, metadata_path, xml = map(lambda path: Path(path).resolve(strict=True), (checkpoint, metadata_path, xml))
        metadata = json.loads(metadata_path.read_text())
        from scripts.build_bfm_deployment_metadata import build_metadata
        # Shapes alone cannot establish num_heads, mode identity or PD scales.
        # Reconstruct the complete pinned profile from its original evidence.
        if metadata != build_metadata(metadata["archive"], checkpoint):
            raise ValueError("Metadata fields differ from the evidence-bound deployment profile")
        if (metadata.get("source_profile_verified_unchanged") is not True
                or metadata.get("mode_feature_dims") != [3, 3, 6, 6]
                or metadata.get("mode_mapping_with_time") is not True
                or metadata.get("step_dt") != .02
                or metadata.get("joint_velocity_observation_scale") != .05
                or metadata.get("future_idx") != [0, 1, 2, 3, 4, 5]):
            raise ValueError("Require the audited nominal, six-future, 50 Hz deployment contract")
        frozen = {str(path): sha256(path) for path in (checkpoint, metadata_path, xml, Path(__file__).resolve(),
                  Path(portable.__file__).resolve(), portable.NETWORK_SOURCE)}
        for path, expected in metadata.get("source_hashes", {}).items():
            if sha256(path) != expected:
                raise ValueError(f"Metadata source changed: {path}")
            frozen[path] = expected
        for mesh in sorted(xml.parent.rglob("*")):
            if mesh.suffix.lower() in (".stl", ".obj", ".xml") and mesh.is_file():
                frozen[str(mesh)] = sha256(mesh)
        if metadata.get("checkpoint_sha256") != frozen[str(checkpoint)]:
            raise ValueError("Checkpoint does not match the deployment metadata")
        report.update(input_sha256=frozen, torch_version=torch.__version__, source_checkpoint=str(checkpoint))
        torch.set_num_threads(2)
        actor, embedder = portable.load_native_actor(checkpoint, metadata)
        wrapper = portable.PortableBFMPolicy(actor, embedder, metadata, xml).eval()
        example = portable.example_inputs(metadata)
        with torch.inference_mode():
            # Trace fixed dimensions, but not a fixed mask. Other masks and new
            # random inputs are independently tested after save/reload below.
            traced = torch.jit.trace(wrapper, example, check_trace=True, strict=True)
            artifact = output / "policy.pt"
            torch.jit.save(traced, str(artifact))
            loaded = torch.jit.load(str(artifact), map_location="cpu").eval()
            for seed in (42, 314159):
                for mode in range(8):
                    inputs = list(portable.example_inputs(metadata, seed))
                    inputs[7] = torch.tensor([mode], dtype=torch.long)
                    pd_eager, raw_eager = wrapper(*inputs)
                    pd_saved, raw_saved = loaded(*inputs)
                    prop, actions, task = portable.independent_observations(inputs, metadata, xml)
                    wrapper_prop, wrapper_task = wrapper.observations(*inputs[:4], *inputs[5:])
                    raw_native = actor(prop, actions, embedder(task))
                    pd_native = raw_native * wrapper.action_scale + wrapper.default_dof_pos
                    errors = {"saved_vs_eager_action": float((raw_saved - raw_eager).abs().max()),
                              "saved_vs_eager_pd_rad": float((pd_saved - pd_eager).abs().max()),
                              "saved_vs_native_action": float((raw_saved - raw_native).abs().max()),
                              "saved_vs_native_pd_rad": float((pd_saved - pd_native).abs().max()),
                              "prop_observation_max_error": float((wrapper_prop - prop).abs().max()),
                              "task_observation_max_error": float((wrapper_task - task).abs().max())}
                    if not all(np_isfinite(value) for value in errors.values()):
                        raise ValueError(f"Nonfinite export verification metric at seed={seed}, mode={mode}")
                    passed = all(np_isfinite(value) and value <= (2e-5 if "observation" in key else 1e-4)
                                 for key, value in errors.items())
                    report["checks"].append({"seed": seed, "mode_index": mode, **errors, "passed": passed})
        if any(sha256(path) != digest for path, digest in frozen.items()):
            raise ValueError("Input changed during export")
        passed = all(row["passed"] for row in report["checks"])
        metadata.update(runtime_backend="torchscript", export_batch_size=1,
                        policy_sha256=sha256(artifact), fk_xml_sha256=sha256(xml),
                        inference_input_quaternion_order="wxyz", export_verification="PASS" if passed else "FAIL")
        (output / "policy_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        report.update(result="PASS" if passed else "FAIL", artifact=str(artifact),
                      artifact_sha256=sha256(artifact), metadata_sha256=sha256(output / "policy_metadata.json"),
                      inputs_verified_unchanged=True, masks_checked=8, input_seeds_checked=2)
    except (Exception, KeyboardInterrupt) as error:
        report["result"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "ERROR"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.signal(signal.SIGTERM, previous)
    report["elapsed_seconds"] = time.monotonic() - started
    (output / "export_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def np_isfinite(value):
    import math
    return math.isfinite(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "metadata", "xml", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    bounded_unit()
    result = export(args.checkpoint, args.metadata, args.xml, args.output)
    print(json.dumps({key: result[key] for key in ("result", "error", "artifact", "elapsed_seconds") if key in result}))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

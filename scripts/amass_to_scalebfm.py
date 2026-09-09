#!/usr/bin/env python3
"""Run the AMASS -> ScaleRetarget -> ScaleTrack preparation pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import zipfile

import numpy as np


PUBLIC_AMASS_SAMPLE_URL = (
    "https://raw.githubusercontent.com/nghorbani/human_body_prior/"
    "78c86eae5ed518ae22bf197fd74211bbfa45551a/support_data/dowloads/amass_sample.npz"
)
PUBLIC_AMASS_SAMPLE_SHA256 = "683975321ac32ffe600259cd000e42d338eb0637bb8c9ef92c1733dc582a79a1"
PACKED_MOTION_FORMAT_VERSION = 3
PACKED_QUATERNION_ORDER = "wxyz"
PREPARED_MOTION_MANIFEST = ".bfm_motion_manifest"
ISAACLAB_PACKAGE_MODULES = (
    "isaaclab",
    "isaaclab.app",
    "isaaclab.assets",
    "isaaclab.scene",
    "isaaclab.sim",
    "isaaclab.utils.math",
    "isaaclab_physx",
    "joblib",
    "numpy",
    "scaletrack",
    "scaletrack.robots.g1_29dof",
    "torch",
    "yaml",
)
ISAACLAB_POLICY_MODULES = ("my_rsl_rl",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_python_interpreter(path: Path, *, label: str) -> None:
    """Reject a missing or non-executable stage interpreter."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} Python interpreter does not exist: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{label} Python interpreter is not executable: {path}")


def probe_python_environment(
    interpreter: Path,
    modules: Sequence[str],
    *,
    tree_modules: Sequence[str] = (),
    label: str,
    cwd: Path,
    env: dict[str, str],
) -> str:
    """Import stage dependencies and return a content/version fingerprint."""
    fingerprint_prefix = "__BFM_ENV_FINGERPRINT__="
    import_code = f"""
import hashlib
import importlib
import json
from pathlib import Path
import sys

records = {{}}
tree_module_names = set({tuple(tree_modules)!r})
for module_name in {tuple(modules)!r}:
    module = importlib.import_module(module_name)
    module_path = getattr(module, "__file__", None)
    module_sha256 = None
    if module_path and Path(module_path).is_file():
        module_sha256 = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
    tree_sha256 = None
    tree_file_count = 0
    if module_name in tree_module_names:
        roots = [Path(root) for root in getattr(module, "__path__", ())]
        if not roots and module_path:
            roots = [Path(module_path).parent]
        tree_digest = hashlib.sha256()
        for root_index, root in enumerate(sorted(roots, key=str)):
            source_files = sorted(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix in {{".py", ".pyi"}}
                and "__pycache__" not in path.parts
            )
            for source_file in source_files:
                relative_path = source_file.relative_to(root).as_posix()
                tree_digest.update(
                    f"{{root_index}}:{{relative_path}}\\0".encode("utf-8")
                )
                tree_digest.update(source_file.read_bytes())
                tree_file_count += 1
        tree_sha256 = tree_digest.hexdigest()
    records[module_name] = {{
        "file": str(module_path),
        "file_sha256": module_sha256,
        "tree_file_count": tree_file_count,
        "tree_sha256": tree_sha256,
        "version": str(getattr(module, "__version__", "unknown")),
    }}
payload = {{
    "executable": sys.executable,
    "python": sys.version,
    "modules": records,
}}
print({fingerprint_prefix!r} + json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""
    try:
        result = subprocess.run(
            (str(interpreter), "-c", import_code),
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"{label} dependency probe could not run with {interpreter}: {exc}"
        ) from exc
    if result.returncode != 0:
        output = result.stdout.strip()
        details = f"\n{output}" if output else ""
        raise RuntimeError(
            f"{label} dependency probe failed with {interpreter}; required modules: "
            f"{', '.join(modules)}{details}"
        )
    payload_line = next(
        (
            line[len(fingerprint_prefix) :]
            for line in reversed(result.stdout.splitlines())
            if line.startswith(fingerprint_prefix)
        ),
        None,
    )
    if payload_line is None:
        raise RuntimeError(f"{label} dependency probe returned no environment fingerprint")
    return hashlib.sha256(payload_line.encode("utf-8")).hexdigest()


def isaaclab_dependency_groups(
    *, needs_policy: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep packaging identity independent from optional train/play dependencies."""
    policy_modules = ISAACLAB_POLICY_MODULES if needs_policy else ()
    return ISAACLAB_PACKAGE_MODULES, policy_modules


def _semantic_files(path: Path) -> list[Path]:
    """Return deterministic semantic inputs while excluding generated caches."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    excluded_parts = {".git", "__pycache__", "logs", "outputs"}
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix not in {".pyc", ".pyo"}
        and excluded_parts.isdisjoint(candidate.relative_to(path).parts)
    )


def semantic_fingerprint(
    artifacts: dict[str, Path], settings: dict[str, object]
) -> str:
    """Hash named model/code/config inputs without depending on workspace location."""
    digest = hashlib.sha256(b"ScaleBFM semantic fingerprint v1\0")
    digest.update(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for label, artifact in sorted(artifacts.items()):
        artifact = Path(artifact)
        semantic_files = _semantic_files(artifact)
        digest.update(f"\0artifact:{label}\0".encode("utf-8"))
        if not semantic_files:
            digest.update(b"<missing-or-empty>")
            continue
        for semantic_file in semantic_files:
            relative_name = (
                semantic_file.name
                if artifact.is_file()
                else semantic_file.relative_to(artifact).as_posix()
            )
            digest.update(f"\0file:{relative_name}\0".encode("utf-8"))
            digest.update(bytes.fromhex(_sha256(semantic_file)))
    return digest.hexdigest()


def download_verified_file(url: str, destination: Path, expected_sha256: str) -> Path:
    """Download atomically and reject content that does not match its pinned hash."""
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError(f"Refusing a symlink as a downloaded file: {destination}")
    if destination.is_file():
        actual_sha256 = _sha256(destination)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for existing file {destination}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as output:
            partial = Path(output.name)
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output)
            output.flush()
            os.fsync(output.fileno())
        actual_sha256 = _sha256(partial)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for downloaded file: expected {expected_sha256}, got {actual_sha256}"
            )
        partial.replace(destination)
    finally:
        if partial is not None and partial.exists():
            partial.unlink()
    return destination


def _validate_stageii_arrays(data: object, source_file: Path) -> None:
    root_orient = np.asarray(data["root_orient"])
    pose_body = np.asarray(data["pose_body"])
    trans = np.asarray(data["trans"])
    if root_orient.ndim != 2 or root_orient.shape[1] != 3:
        raise ValueError(f"{source_file}: root_orient must have shape (N, 3)")
    if pose_body.ndim != 2 or pose_body.shape[1] != 63:
        raise ValueError(f"{source_file}: pose_body must have shape (N, 63)")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"{source_file}: trans must have shape (N, 3)")
    if not (root_orient.shape[0] == pose_body.shape[0] == trans.shape[0]):
        raise ValueError(f"{source_file}: root_orient, pose_body and trans frame counts must match")
    if root_orient.shape[0] < 3:
        raise ValueError(f"{source_file}: AMASS motion needs at least 3 frames")
    betas = np.asarray(data["betas"])
    if betas.ndim not in {1, 2} or betas.size == 0:
        raise ValueError(
            f"{source_file}: betas must be a non-empty 1D or 2D numeric array"
        )
    gender_array = np.asarray(data["gender"])
    if gender_array.ndim != 0:
        raise ValueError(f"{source_file}: gender must be a scalar string")
    gender_value = gender_array.item()
    if isinstance(gender_value, bytes):
        gender_value = gender_value.decode("utf-8")
    normalized_gender = str(gender_value).lower()
    if normalized_gender not in {"female", "male", "neutral"}:
        raise ValueError(
            f"{source_file}: gender must be one of female, male, or neutral"
        )
    for name, array in (
        ("root_orient", root_orient),
        ("pose_body", pose_body),
        ("trans", trans),
        ("betas", betas),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{source_file}: {name} contains NaN or Inf")
    field_names = set(data.files) if hasattr(data, "files") else set(data.keys())
    fps_key = "mocap_frame_rate" if "mocap_frame_rate" in field_names else "mocap_framerate"
    fps = float(np.asarray(data[fps_key]).item())
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{source_file}: motion frame rate must be positive and finite")


def _prepare_generated_output(path: Path, *, force: bool) -> bool:
    """Return whether a generated file should be written without following symlinks."""
    if path.is_symlink():
        if not force:
            raise FileExistsError(f"Refusing to replace generated output symlink: {path}")
        return True
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as existing:
            _validate_stageii_arrays(existing, path)
        return False
    return True


def _atomic_savez_compressed(destination: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a NumPy archive beside its destination, then publish it atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".npz",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_prepared_motion_manifest(
    prepared_dir: Path,
    prepared_files: Sequence[Path],
) -> Path:
    """Atomically publish the exact Stage-II list consumed by ScaleRetarget."""
    prepared_root = Path(prepared_dir).resolve()
    relative_paths = [
        Path(path).absolute().relative_to(Path(prepared_dir).absolute()).as_posix()
        for path in prepared_files
    ]
    if not relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Prepared motion manifest requires unique, non-empty inputs")
    manifest_path = prepared_root / PREPARED_MOTION_MANIFEST
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=prepared_root,
            prefix=f".{PREPARED_MOTION_MANIFEST}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write("".join(f"{relative_path}\n" for relative_path in relative_paths))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(manifest_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return manifest_path


def _ensure_generated_output_matches(
    output_path: Path,
    expected: dict[str, np.ndarray],
    source_file: Path,
) -> None:
    """Reject cached compatibility data derived from different source content."""
    with np.load(output_path, allow_pickle=False) as existing:
        matches = set(existing.files) == set(expected) and all(
            np.array_equal(np.asarray(existing[name]), np.asarray(value))
            for name, value in expected.items()
        )
    if not matches:
        raise ValueError(
            f"{source_file}: legacy source changed since {output_path} was created; "
            "rerun with --force-data or use a new run-name"
        )


@dataclass(frozen=True)
class PipelineOptions:
    repo_root: Path
    source: Path
    run_name: str
    smplx_model: Path
    retarget_python: Path
    isaaclab_python: Path
    output_fps: int = 50
    package_num_envs: int = 1
    package_loader_workers: int = 4
    retarget_workers: int = 1
    device: str = "cuda:0"
    train: bool = False
    task: str = "G1-BFM-Transformer-Tracking"
    train_num_envs: int = 256
    max_iterations: int = 1000
    seed: int = 42
    base_run: str = "humanoid_transformer_m"
    base_checkpoint: str = "model_22200.pt"
    training_run_name: str | None = None
    play: bool = False
    play_run: str | None = None
    play_checkpoint: str | None = None
    mode_index: int = 7
    play_num_envs: int = 1
    play_headless: bool = False
    local_tracking: bool = False
    record_video: bool = False
    video_length: int = 250
    force_data: bool = False
    force_package: bool = False
    adopt_legacy_retarget: bool = False
    height_adjust_mode: str = "body_origin"


@dataclass(frozen=True)
class PipelineCommand:
    name: str
    argv: tuple[str, ...]
    cwd: Path


def retarget_pipeline_fingerprint(
    options: PipelineOptions, environment_fingerprint: str
) -> str:
    """Fingerprint every local semantic input to AMASS retargeting."""
    retarget_root = options.repo_root / "ScaleRetarget"
    return semantic_fingerprint(
        {
            "smplx_model": options.smplx_model,
            "entry_point": retarget_root / "retarget.py",
            "implementation": retarget_root / "scaleretarget",
            "configuration": retarget_root / "config",
            "robot_kinematics": retarget_root
            / "assets/unitree_g1/g1_mocap_29dof.xml",
        },
        {
            "environment": environment_fingerprint,
            "pipeline": "amass-to-g1-retarget",
            "provenance_version": 2,
            "height_adjust_mode": options.height_adjust_mode,
            "ground_offset_m": 0.002 if options.height_adjust_mode == "collision_sole" else 0.0,
        },
    )


def package_pipeline_fingerprint(
    options: PipelineOptions, environment_fingerprint: str
) -> str:
    """Fingerprint code, robot description, runtime, and output contract."""
    track_root = options.repo_root / "ScaleTrack"
    package_root = track_root / "scripts/pretrain/data_process"
    scaletrack_root = track_root / "source/scaletrack/scaletrack"
    robot_assets = scaletrack_root / "assets/robots/g1_29dof"
    return semantic_fingerprint(
        {
            "package_entry_point": package_root / "package_motions.py",
            "package_helpers": package_root / "motion_processing.py",
            "archive_contract": scaletrack_root / "utils/motion_archive.py",
            "quaternion_contract": scaletrack_root / "utils/quaternion_compat.py",
            "robot_configuration": scaletrack_root / "robots/g1_29dof.py",
            "robot_usda": robot_assets / "g1_29dof.usda",
            "robot_usd_configuration": robot_assets / "configuration",
        },
        {
            "environment": environment_fingerprint,
            "output_fps": options.output_fps,
            "pipeline": "g1-isaaclab-forward-kinematics-package",
            "robot_type": "g1_29dof",
            "format_version": PACKED_MOTION_FORMAT_VERSION,
        },
    )


def build_command_plan(options: PipelineOptions) -> list[PipelineCommand]:
    """Build subprocess boundaries without importing IsaacLab into this process."""
    repo_root = options.repo_root
    retarget_root = repo_root / "ScaleRetarget"
    track_root = repo_root / "ScaleTrack"
    prepared_dir = retarget_root / "dataset" / options.run_name
    retarget_output_root = retarget_root / "retargeted_dataset"
    retargeted_dir = retarget_output_root / options.run_name
    processed_dir = retarget_output_root / f"{options.run_name}_processed"
    yaml_path = retarget_output_root / f"{options.run_name}.yaml"

    retarget_argv = (
        str(options.retarget_python),
        str(retarget_root / "retarget.py"),
        "+loader=amass",
        "+formatter=kinematic",
        f"data_path={prepared_dir}",
        f"output_dir={retarget_output_root}",
        f"loader.config.body_model_path={options.smplx_model}",
        f"loader.config.motion_manifest={prepared_dir / PREPARED_MOTION_MANIFEST}",
    )
    if options.retarget_workers > 1:
        retarget_argv += ("multi_process=True", f"num_workers={options.retarget_workers}")
    if options.height_adjust_mode not in {"body_origin", "collision_sole"}:
        raise ValueError("Unknown height adjustment mode")
    if options.height_adjust_mode != "body_origin":
        retarget_argv += (f"formatter.config.height_adjust_mode={options.height_adjust_mode}",
                         "formatter.config.ground_offset_m=0.002")
    if options.force_data:
        retarget_argv += ("loader.config.overwrite=true",)

    commands = [
        PipelineCommand(
            name="retarget",
            argv=retarget_argv,
            cwd=retarget_root,
        ),
        PipelineCommand(
            name="package",
            argv=(
                str(options.isaaclab_python),
                str(track_root / "scripts/pretrain/data_process/package_motions.py"),
                "--data_dir",
                str(retargeted_dir),
                "--data_format",
                "pkl",
                "--output_dir",
                str(processed_dir),
                "--output_fps",
                str(options.output_fps),
                "--num_envs",
                str(options.package_num_envs),
                "--loader_workers",
                str(options.package_loader_workers),
                "--robot_type",
                "g1_29dof",
                "--device",
                options.device,
                "--headless",
            ),
            cwd=track_root,
        ),
        PipelineCommand(
            name="yaml",
            argv=(
                str(options.retarget_python),
                str(track_root / "scripts/pretrain/data_process/create_yaml.py"),
                "--data_dir",
                str(processed_dir),
                "--output_path",
                str(yaml_path),
            ),
            cwd=track_root,
        ),
    ]
    if options.train:
        commands.append(
            PipelineCommand(
                name="train",
                argv=(
                    str(options.isaaclab_python),
                    str(track_root / "scripts/pretrain/rsl_rl/train.py"),
                    "--task",
                    options.task,
                    "--motion_file",
                    str(yaml_path),
                    "--run_name",
                    options.training_run_name or f"{options.run_name}_finetune",
                    "--logger",
                    "tensorboard",
                    "--num_envs",
                    str(options.train_num_envs),
                    "--max_iterations",
                    str(options.max_iterations),
                    "--seed",
                    str(options.seed),
                    "--resume",
                    "True",
                    "--load_run",
                    options.base_run,
                    "--checkpoint",
                    options.base_checkpoint,
                    "--device",
                    options.device,
                    "--headless",
                ),
                cwd=track_root,
            )
        )
    if options.play:
        training_run_name = options.training_run_name or f"{options.run_name}_finetune"
        play_run = options.play_run or (training_run_name if options.train else options.base_run)
        play_checkpoint = options.play_checkpoint or (
            "__LATEST__" if options.train else options.base_checkpoint
        )
        play_argv = [
            str(options.isaaclab_python),
            str(track_root / "scripts/pretrain/rsl_rl/play.py"),
            "--task",
            options.task,
            "--load_run",
            play_run,
            "--checkpoint",
            play_checkpoint,
            "--motion_file",
            str(yaml_path),
            "--num_envs",
            str(options.play_num_envs),
            "--mode_index",
            str(options.mode_index),
            "--device",
            options.device,
        ]
        if options.local_tracking:
            play_argv.append("--local_tracking")
        if options.record_video:
            play_argv.extend(("--video", "--video_length", str(options.video_length)))
        if options.play_headless:
            play_argv.append("--headless")
        else:
            play_argv.extend(("--viz", "kit"))
        commands.append(PipelineCommand(name="play", argv=tuple(play_argv), cwd=track_root))
    return commands


def prepare_amass_dataset(source: Path, prepared_dir: Path, *, force: bool = False) -> list[Path]:
    """Prepare native Stage-II files or convert legacy AMASS samples."""
    source = Path(source)
    prepared_dir = Path(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        source_files = sorted(
            path for path in source.rglob("*.npz") if not path.name.endswith("_stagei.npz")
        )
        relative_paths = [path.relative_to(source) for path in source_files]
    else:
        source_files = [source]
        relative_paths = [Path(source.name)]
    if not source_files:
        raise FileNotFoundError(f"No AMASS .npz motions found under: {source}")

    prepared_files: list[Path] = []
    native_core = {"root_orient", "pose_body", "trans", "betas", "gender"}
    legacy_core = {"poses", "trans", "betas", "gender"}
    for source_file, relative_path in zip(source_files, relative_paths, strict=True):
        with np.load(source_file, allow_pickle=False) as data:
            fields = set(data.files)
            fps_field = (
                "mocap_frame_rate"
                if "mocap_frame_rate" in fields
                else "mocap_framerate" if "mocap_framerate" in fields else None
            )
            if native_core.issubset(fields) and fps_field == "mocap_frame_rate":
                _validate_stageii_arrays(data, source_file)
                output_path = prepared_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    already_prepared = source_file.samefile(output_path)
                except OSError:
                    already_prepared = source_file.absolute() == output_path.absolute()
                if already_prepared:
                    prepared_files.append(output_path)
                    continue
                if force and (output_path.exists() or output_path.is_symlink()):
                    output_path.unlink()
                if output_path.is_symlink():
                    if output_path.resolve() != source_file.resolve():
                        raise FileExistsError(
                            f"{output_path} already points to a different source: {output_path.resolve()}"
                        )
                elif output_path.exists():
                    if output_path.resolve() != source_file.resolve():
                        raise FileExistsError(
                            f"{output_path} already exists for a different source"
                        )
                else:
                    output_path.symlink_to(source_file.resolve())
            elif legacy_core.issubset(fields) and fps_field is not None:
                poses = np.asarray(data["poses"])
                trans = np.asarray(data["trans"])
                if poses.ndim != 2 or poses.shape[1] < 66:
                    raise ValueError(f"{source_file}: poses must have shape (N, >=66)")
                if trans.shape != (poses.shape[0], 3):
                    raise ValueError(
                        f"{source_file}: trans must have shape ({poses.shape[0]}, 3)"
                    )
                output_name = relative_path.stem
                if not output_name.endswith("_stageii"):
                    output_name += "_stageii"
                output_path = prepared_dir / relative_path.parent / f"{output_name}.npz"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                converted = {
                    "root_orient": poses[:, :3],
                    "pose_body": poses[:, 3:66],
                    "trans": trans,
                    "betas": np.asarray(data["betas"]),
                    "gender": np.asarray(data["gender"]),
                    "mocap_frame_rate": np.asarray(data[fps_field]),
                }
                _validate_stageii_arrays(converted, source_file)
                if _prepare_generated_output(output_path, force=force):
                    _atomic_savez_compressed(output_path, converted)
                else:
                    _ensure_generated_output_matches(output_path, converted, source_file)
            elif native_core.issubset(fields) and fps_field == "mocap_framerate":
                _validate_stageii_arrays(data, source_file)
                output_path = prepared_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                converted = {
                    "root_orient": np.asarray(data["root_orient"]),
                    "pose_body": np.asarray(data["pose_body"]),
                    "trans": np.asarray(data["trans"]),
                    "betas": np.asarray(data["betas"]),
                    "gender": np.asarray(data["gender"]),
                    "mocap_frame_rate": np.asarray(data[fps_field]),
                }
                if _prepare_generated_output(output_path, force=force):
                    _atomic_savez_compressed(
                        output_path,
                        converted,
                    )
                else:
                    _ensure_generated_output_matches(output_path, converted, source_file)
            else:
                raise ValueError(
                    f"{source_file}: expected AMASS Stage-II fields "
                    "(root_orient, pose_body, trans, betas, gender, mocap_frame_rate) "
                    "or the pinned public sample's legacy fields"
                )
            prepared_files.append(output_path)

    return prepared_files


def expected_retargeted_files(
    prepared_files: Sequence[Path], prepared_dir: Path, retargeted_dir: Path
) -> list[Path]:
    return [
        (retargeted_dir / prepared_file.relative_to(prepared_dir)).with_suffix(".pkl")
        for prepared_file in prepared_files
    ]


def expected_processed_files(
    retargeted_files: Sequence[Path], retargeted_dir: Path, processed_dir: Path
) -> list[Path]:
    return [
        processed_dir / retargeted_file.relative_to(retargeted_dir).with_suffix(".npz")
        for retargeted_file in retargeted_files
    ]


def _files_complete(paths: Sequence[Path]) -> bool:
    return bool(paths) and all(
        not path.is_symlink() and path.is_file() and path.stat().st_size > 0
        for path in paths
    )


def _file_stat_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def newer_inputs(
    input_files: Sequence[Path], output_files: Sequence[Path]
) -> list[tuple[Path, Path]]:
    """Return input/output pairs whose cached output predates its source."""
    if len(input_files) != len(output_files):
        return list(zip(input_files, output_files))
    return [
        (input_path, output_path)
        for input_path, output_path in zip(input_files, output_files, strict=True)
        if input_path.stat().st_mtime_ns > output_path.stat().st_mtime_ns
    ]


def _retarget_source_digest_path(retargeted_file: Path) -> Path:
    return retargeted_file.with_suffix(f"{retargeted_file.suffix}.source.sha256")


def _retarget_pipeline_digest_path(retargeted_file: Path) -> Path:
    return retargeted_file.with_suffix(f"{retargeted_file.suffix}.pipeline.sha256")


def _atomic_write_digest(destination: Path, digest: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Invalid SHA-256 digest for {destination}: {digest!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(f"{digest}\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def record_retarget_provenance(
    prepared_files: Sequence[Path],
    retargeted_files: Sequence[Path],
    pipeline_fingerprint: str,
) -> None:
    """Bind each cache entry to its AMASS input and semantic pipeline."""
    if len(prepared_files) != len(retargeted_files):
        raise ValueError("Retarget provenance requires one output per prepared input")
    for prepared_file, retargeted_file in zip(
        prepared_files, retargeted_files, strict=True
    ):
        _atomic_write_digest(
            _retarget_source_digest_path(retargeted_file), _sha256(prepared_file)
        )
        _atomic_write_digest(
            _retarget_pipeline_digest_path(retargeted_file), pipeline_fingerprint
        )


def verify_retarget_provenance(
    prepared_files: Sequence[Path],
    retargeted_files: Sequence[Path],
    expected_pipeline_fingerprint: str,
) -> list[Path]:
    """Verify source and pipeline provenance; report every missing sidecar."""
    if len(prepared_files) != len(retargeted_files):
        raise ValueError("Retarget provenance requires one output per prepared input")
    missing: list[Path] = []
    for prepared_file, retargeted_file in zip(
        prepared_files, retargeted_files, strict=True
    ):
        source_digest_path = _retarget_source_digest_path(retargeted_file)
        pipeline_digest_path = _retarget_pipeline_digest_path(retargeted_file)
        for digest_path in (source_digest_path, pipeline_digest_path):
            if not digest_path.exists() and not digest_path.is_symlink():
                missing.append(digest_path)
                continue
            if digest_path.is_symlink() or not digest_path.is_file():
                raise RuntimeError(
                    f"Retarget provenance is not a regular file: {digest_path}; "
                    "rerun with --force-data"
                )
            recorded_digest = digest_path.read_text(encoding="ascii").strip()
            if not re.fullmatch(r"[0-9a-f]{64}", recorded_digest):
                raise RuntimeError(
                    f"Retarget provenance is malformed: {digest_path}; "
                    "rerun with --force-data"
                )
        if source_digest_path in missing or pipeline_digest_path in missing:
            continue
        recorded_digest = source_digest_path.read_text(encoding="ascii").strip()
        if recorded_digest != _sha256(prepared_file):
            raise RuntimeError(
                f"Prepared input content no longer matches cached retarget output "
                f"{retargeted_file}; rerun with --force-data or use a new run-name"
            )
        recorded_pipeline = pipeline_digest_path.read_text(encoding="ascii").strip()
        if recorded_pipeline != expected_pipeline_fingerprint:
            raise RuntimeError(
                f"Cached retarget pipeline fingerprint does not match current model, "
                f"configuration, code, or environment for {retargeted_file}; "
                "rerun with --force-data or use a new run-name"
            )
    return missing


def latest_checkpoint(run_dir: Path) -> Path:
    """Return the model checkpoint with the greatest numeric iteration suffix."""
    candidates: list[tuple[int, Path]] = []
    for path in Path(run_dir).glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No model_<iteration>.pt checkpoints found in: {run_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def validate_retargeted_motion(path: Path) -> dict[str, float | int | str]:
    """Validate ScaleRetarget's G1 trajectory contract before launching IsaacLab."""
    import joblib

    motion = joblib.load(path)
    required_fields = {"fps", "root_pos", "root_rot", "dof_pos"}
    missing = required_fields.difference(motion)
    if missing:
        raise ValueError(f"{path}: missing retargeted fields: {sorted(missing)}")
    root_pos = np.asarray(motion["root_pos"])
    root_rot = np.asarray(motion["root_rot"])
    dof_pos = np.asarray(motion["dof_pos"])
    fps = float(np.asarray(motion["fps"]).item())
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path}: root_pos must have shape (N, 3)")
    frame_count = root_pos.shape[0]
    if root_rot.shape != (frame_count, 4):
        raise ValueError(f"{path}: root_rot must have shape ({frame_count}, 4)")
    if dof_pos.shape != (frame_count, 29):
        raise ValueError(f"{path}: dof_pos must have shape ({frame_count}, 29)")
    if frame_count < 3:
        raise ValueError(f"{path}: retargeted motion needs at least 3 frames")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: fps must be a positive finite scalar")
    for name, array in (("root_pos", root_pos), ("root_rot", root_rot), ("dof_pos", dof_pos)):
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: {name} contains NaN or Inf")
    if not np.allclose(np.linalg.norm(root_rot, axis=-1), 1.0, atol=1.0e-3):
        raise ValueError(f"{path}: root_rot contains non-unit quaternions")
    return {"frames": frame_count, "fps": fps, "dofs": 29, "quat_order": "xyzw"}


def validate_processed_motion(
    path: Path,
    *,
    source_path: Path | None = None,
    require_current_format: bool = False,
    expected_pipeline_fingerprint: str | None = None,
) -> dict[str, float | int]:
    """Validate the on-disk tensor contract consumed by ScaleTrack."""
    required_fields = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }
    if require_current_format and path.is_symlink():
        raise ValueError(f"{path}: refusing a symlink as a generated motion archive")
    with np.load(path, allow_pickle=False) as motion:
        missing = required_fields.difference(motion.files)
        if missing:
            raise ValueError(f"{path}: missing processed fields: {sorted(missing)}")
        joint_pos = np.asarray(motion["joint_pos"])
        joint_vel = np.asarray(motion["joint_vel"])
        body_pos = np.asarray(motion["body_pos_w"])
        body_quat = np.asarray(motion["body_quat_w"])
        body_lin_vel = np.asarray(motion["body_lin_vel_w"])
        body_ang_vel = np.asarray(motion["body_ang_vel_w"])
        fps = float(np.asarray(motion["fps"]).item())
        format_version = (
            np.asarray(motion["format_version"]).item()
            if "format_version" in motion.files
            else None
        )
        quaternion_order = (
            str(np.asarray(motion["quaternion_order"]).item())
            if "quaternion_order" in motion.files
            else None
        )
        source_sha256 = (
            str(np.asarray(motion["source_sha256"]).item())
            if "source_sha256" in motion.files
            else None
        )
        pipeline_fingerprint = (
            str(np.asarray(motion["pipeline_fingerprint"]).item())
            if "pipeline_fingerprint" in motion.files
            else None
        )
        reference_root_pos = (
            np.asarray(motion["reference_root_pos"])
            if "reference_root_pos" in motion.files
            else None
        )
        reference_root_quat = (
            np.asarray(motion["reference_root_quat_w"])
            if "reference_root_quat_w" in motion.files
            else None
        )

    if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
        raise ValueError(f"{path}: joint_pos must have shape (N, 29)")
    frame_count = joint_pos.shape[0]
    expected_shapes = {
        "joint_vel": (frame_count, 29),
        "body_pos_w": (frame_count, 30, 3),
        "body_quat_w": (frame_count, 30, 4),
        "body_lin_vel_w": (frame_count, 30, 3),
        "body_ang_vel_w": (frame_count, 30, 3),
    }
    arrays = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(f"{path}: {name} must have shape {expected_shape}")
    if frame_count < 3:
        raise ValueError(f"{path}: processed motion needs at least 3 frames")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: fps must be a positive finite scalar")
    if fps != 50.0:
        raise ValueError(f"{path}: processed motion must be exactly 50 FPS, got {fps}")
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: {name} contains NaN or Inf")
    quaternion_norms = np.linalg.norm(body_quat, axis=-1)
    if not np.allclose(quaternion_norms, 1.0, atol=1.0e-3):
        raise ValueError(f"{path}: body_quat_w contains non-unit quaternions")
    if format_version is not None and (
        isinstance(format_version, bool) or not isinstance(format_version, int)
    ):
        raise ValueError(f"{path}: format_version must be an integer scalar")
    if require_current_format:
        if format_version != PACKED_MOTION_FORMAT_VERSION:
            raise ValueError(
                f"{path}: format_version must be {PACKED_MOTION_FORMAT_VERSION}, "
                f"got {format_version!r}"
            )
        if quaternion_order != PACKED_QUATERNION_ORDER:
            raise ValueError(
                f"{path}: quaternion_order must be {PACKED_QUATERNION_ORDER!r}, "
                f"got {quaternion_order!r}"
            )
        if pipeline_fingerprint is None or not re.fullmatch(
            r"[0-9a-f]{64}", pipeline_fingerprint
        ):
            raise ValueError(f"{path}: missing or malformed pipeline_fingerprint")
        if (
            expected_pipeline_fingerprint is not None
            and pipeline_fingerprint != expected_pipeline_fingerprint
        ):
            raise ValueError(
                f"{path}: pipeline_fingerprint does not match current package code, "
                "robot assets, or IsaacLab environment"
            )
        if reference_root_pos is None or reference_root_pos.shape != (frame_count, 3):
            raise ValueError(
                f"{path}: reference_root_pos must have shape ({frame_count}, 3)"
            )
        if reference_root_quat is None or reference_root_quat.shape != (frame_count, 4):
            raise ValueError(
                f"{path}: reference_root_quat_w must have shape ({frame_count}, 4)"
            )
        if not np.isfinite(reference_root_pos).all() or not np.isfinite(
            reference_root_quat
        ).all():
            raise ValueError(f"{path}: reference root trajectory contains NaN or Inf")
        if not np.allclose(body_pos[:, 0], reference_root_pos, atol=1.0e-4, rtol=0.0):
            max_error = float(np.max(np.abs(body_pos[:, 0] - reference_root_pos)))
            raise ValueError(
                f"{path}: packed pelvis positions disagree with reference root "
                f"(max abs error {max_error:.6g} m)"
            )
        quaternion_error = np.minimum(
            np.linalg.norm(body_quat[:, 0] - reference_root_quat, axis=-1),
            np.linalg.norm(body_quat[:, 0] + reference_root_quat, axis=-1),
        )
        if float(np.max(quaternion_error)) > 1.0e-4:
            raise ValueError(
                f"{path}: packed pelvis rotations disagree with reference root "
                f"(max sign-invariant L2 error {float(np.max(quaternion_error)):.6g})"
            )
        if source_path is None or not source_path.is_file():
            raise ValueError(f"{path}: current-format validation requires its source .pkl")
        expected_source_sha256 = _sha256(source_path)
        if source_sha256 != expected_source_sha256:
            raise ValueError(
                f"{path}: source_sha256 does not match {source_path}; "
                "repackage this motion"
            )

    return {"frames": frame_count, "fps": fps, "joints": 29, "bodies": 30}


def processed_outputs_are_current(
    processed_files: Sequence[Path],
    retargeted_files: Sequence[Path],
    *,
    expected_pipeline_fingerprint: str,
) -> bool:
    """Return whether every expected archive matches its exact retargeted source."""
    if len(processed_files) != len(retargeted_files) or not _files_complete(processed_files):
        return False
    try:
        for processed_file, retargeted_file in zip(
            processed_files, retargeted_files, strict=True
        ):
            validate_processed_motion(
                processed_file,
                source_path=retargeted_file,
                require_current_format=True,
                expected_pipeline_fingerprint=expected_pipeline_fingerprint,
            )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False
    return True


def positive_int(value: str) -> int:
    """Argparse type for resource counts, frame rates, and run lengths."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def scale_track_fps(value: str) -> int:
    """Keep generated files compatible with ScaleTrack's strict loader."""
    parsed = positive_int(value)
    if parsed != 50:
        raise argparse.ArgumentTypeError(
            f"this ScaleTrack workflow only supports 50 FPS, got {parsed}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AMASS 到 ScaleBFM 的数据、训练与回放流水线。",
        epilog="安全默认值：只准备数据，默认不启动训练；传入 --train 才会训练。",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=Path, help="AMASS Stage-II 文件或目录。")
    source_group.add_argument(
        "--public-sample",
        action="store_true",
        help="下载固定版本的官方公开小样，只用于端到端冒烟测试。",
    )
    parser.add_argument("--run-name", required=True, help="输出数据集及训练运行名称。")
    parser.add_argument("--smplx-model", type=Path, required=True, help="SMPL-X 模型目录或中性模型文件。")
    parser.add_argument(
        "--retarget-python",
        type=Path,
        default=Path(sys.executable),
        help="安装了 ScaleRetarget 的 Python 3.11。",
    )
    parser.add_argument("--isaaclab-python", type=Path, required=True, help="现有 IsaacLab 环境的 Python。")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不创建文件或运行命令。")
    parser.add_argument("--retarget-workers", type=positive_int, default=1, help="动作重定向工作进程数。")
    parser.add_argument(
        "--height-adjust-mode", choices=("body_origin", "collision_sole"), default="body_origin",
        help="默认保留旧高度算法；collision_sole 使用碰撞脚底，需新 run-name 和物理验收。",
    )
    parser.add_argument("--package-num-envs", type=positive_int, default=1, help="IsaacLab 打包并行环境数。")
    parser.add_argument(
        "--package-loader-workers",
        type=positive_int,
        default=4,
        help="每批动作文件的并行加载进程数。",
    )
    parser.add_argument(
        "--output-fps",
        type=scale_track_fps,
        default=50,
        help="ScaleTrack 动作数据帧率（当前必须为 50）。",
    )
    parser.add_argument("--device", default="cuda:0", help="IsaacLab 设备，例如 cuda:0。")
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="重跑重定向与打包并覆盖同名数据产物。",
    )
    parser.add_argument(
        "--force-package",
        action="store_true",
        help="只重跑 IsaacLab 打包，复用已有重定向结果。",
    )
    parser.add_argument(
        "--adopt-legacy-retarget",
        action="store_true",
        help="显式认领缺少来源记录的旧 .pkl；无法证明其历史，仅用于受控迁移。",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="数据准备完成后启动 ScaleTrack 训练（默认不启动训练）。",
    )
    parser.add_argument("--train-run-name", help="训练日志目录名；默认是 <run-name>_finetune。")
    parser.add_argument("--train-num-envs", type=positive_int, default=256, help="训练并行环境数。")
    parser.add_argument("--max-iterations", type=positive_int, default=1000, help="训练迭代数。")
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子。")
    parser.add_argument("--base-run", default="humanoid_transformer_m", help="微调起点的运行目录。")
    parser.add_argument("--base-checkpoint", default="model_22200.pt", help="微调起点的检查点文件。")
    parser.add_argument("--play", action="store_true", help="数据准备后打开策略回放；默认打开 GUI。")
    parser.add_argument(
        "--play-run", help="要回放的训练运行目录；默认使用基础模型或本次训练。"
    )
    parser.add_argument(
        "--play-checkpoint",
        help="要回放的检查点；与 --train 联用时默认自动选最新。",
    )
    parser.add_argument("--mode-index", type=int, choices=range(8), default=7, help="BFM 控制模式 0..7。")
    parser.add_argument("--play-num-envs", type=positive_int, default=1, help="回放并行环境数。")
    parser.add_argument(
        "--local-tracking",
        action="store_true",
        help="以当前机器人根位置对齐参考动作。",
    )
    parser.add_argument("--record-video", action="store_true", help="回放时录制视频。")
    parser.add_argument("--video-length", type=positive_int, default=250, help="录制的仿真步数。")
    parser.add_argument("--play-headless", action="store_true", help="回放时不打开 GUI。")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    public_sample_path = (
        repo_root / "ScaleRetarget/dataset/amass_public_sample_source/amass_sample.npz"
    )
    source = public_sample_path if args.public_sample else args.input
    if not args.public_sample and not source.exists():
        raise FileNotFoundError(f"AMASS input does not exist: {source}")
    if not args.smplx_model.exists():
        raise FileNotFoundError(f"SMPL-X model does not exist: {args.smplx_model}")
    validate_python_interpreter(args.retarget_python, label="ScaleRetarget")
    validate_python_interpreter(args.isaaclab_python, label="IsaacLab")
    if not args.run_name or any(part in {"", ".", ".."} for part in Path(args.run_name).parts) or "/" in args.run_name:
        raise ValueError("run-name must be one path-safe name")
    if args.local_tracking and args.mode_index not in {0, 2, 4, 6, 7}:
        raise ValueError("local tracking requires a root-containing mode: 0, 2, 4, 6, or 7")
    if args.force_data and args.adopt_legacy_retarget:
        raise ValueError("--force-data and --adopt-legacy-retarget are mutually exclusive")

    options = PipelineOptions(
        repo_root=repo_root,
        source=source if args.public_sample else source.resolve(),
        run_name=args.run_name,
        smplx_model=args.smplx_model.resolve(),
        # Do not resolve venv interpreter symlinks: executing the base binary would
        # lose the virtual environment's site-packages.
        retarget_python=args.retarget_python.absolute(),
        isaaclab_python=args.isaaclab_python.absolute(),
        output_fps=args.output_fps,
        package_num_envs=args.package_num_envs,
        package_loader_workers=args.package_loader_workers,
        retarget_workers=args.retarget_workers,
        device=args.device,
        train=args.train,
        train_num_envs=args.train_num_envs,
        max_iterations=args.max_iterations,
        seed=args.seed,
        base_run=args.base_run,
        base_checkpoint=args.base_checkpoint,
        training_run_name=args.train_run_name,
        play=args.play,
        play_run=args.play_run,
        play_checkpoint=args.play_checkpoint,
        mode_index=args.mode_index,
        play_num_envs=args.play_num_envs,
        play_headless=args.play_headless,
        local_tracking=args.local_tracking,
        record_video=args.record_video,
        video_length=args.video_length,
        force_data=args.force_data,
        force_package=args.force_package,
        adopt_legacy_retarget=args.adopt_legacy_retarget,
        height_adjust_mode=args.height_adjust_mode,
    )
    commands = build_command_plan(options)
    command_env = os.environ.copy()
    source_paths = [
        str(options.repo_root / "ScaleTrack/source/scaletrack"),
        str(options.repo_root / "ScaleTrack/source/my_rsl_rl"),
    ]
    if command_env.get("PYTHONPATH"):
        source_paths.append(command_env["PYTHONPATH"])
    command_env["PYTHONPATH"] = os.pathsep.join(source_paths)
    command_env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    command_env["PYTHONUNBUFFERED"] = "1"
    if args.dry_run:
        prepared_dir = options.repo_root / "ScaleRetarget/dataset" / options.run_name
        if args.public_sample:
            print(f"[DRY-RUN] download: {PUBLIC_AMASS_SAMPLE_URL} -> {options.source}")
        print(f"[DRY-RUN] prepare: {options.source} -> {prepared_dir}")
        for command in commands:
            print(f"[DRY-RUN] {command.name}: (cd {command.cwd} && {shlex.join(command.argv)})")
        return 0

    checkpoint_root = (
        options.repo_root / "ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp"
    )
    if options.train:
        base_checkpoint_path = checkpoint_root / options.base_run / options.base_checkpoint
        if not base_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Base checkpoint does not exist: {base_checkpoint_path}. "
                "Download the official checkpoint or choose --base-run/--base-checkpoint."
            )
    elif options.play:
        play_run = options.play_run or options.base_run
        play_checkpoint = options.play_checkpoint or options.base_checkpoint
        play_checkpoint_path = checkpoint_root / play_run / play_checkpoint
        if not play_checkpoint_path.is_file():
            raise FileNotFoundError(f"Playback checkpoint does not exist: {play_checkpoint_path}")

    retarget_environment_fingerprint = probe_python_environment(
        options.retarget_python,
        (
            "hydra",
            "joblib",
            "loguru",
            "mink",
            "mujoco",
            "natsort",
            "numpy",
            "omegaconf",
            "qpsolvers",
            "scipy",
            "smplx",
            "torch",
            "yaml",
            "scaleretarget",
        ),
        label="ScaleRetarget",
        cwd=options.repo_root / "ScaleRetarget",
        env=command_env,
    )
    isaaclab_modules, policy_modules = isaaclab_dependency_groups(
        needs_policy=options.train or options.play
    )
    package_environment_fingerprint = probe_python_environment(
        options.isaaclab_python,
        isaaclab_modules,
        tree_modules=("isaaclab", "isaaclab_physx", "scaletrack"),
        label="IsaacLab/ScaleTrack",
        cwd=options.repo_root / "ScaleTrack",
        env=command_env,
    )
    if policy_modules:
        probe_python_environment(
            options.isaaclab_python,
            policy_modules,
            label="ScaleTrack policy",
            cwd=options.repo_root / "ScaleTrack",
            env=command_env,
        )
    retarget_fingerprint = retarget_pipeline_fingerprint(
        options, retarget_environment_fingerprint
    )
    package_fingerprint = package_pipeline_fingerprint(
        options, package_environment_fingerprint
    )

    prepared_dir = options.repo_root / "ScaleRetarget/dataset" / options.run_name
    if args.public_sample:
        print(f"[RUN] download: {PUBLIC_AMASS_SAMPLE_URL} -> {options.source}", flush=True)
        download_verified_file(PUBLIC_AMASS_SAMPLE_URL, options.source, PUBLIC_AMASS_SAMPLE_SHA256)
    print(f"[RUN] prepare: {options.source} -> {prepared_dir}", flush=True)
    prepared_files = prepare_amass_dataset(options.source, prepared_dir, force=options.force_data)
    write_prepared_motion_manifest(prepared_dir, prepared_files)
    retargeted_dir = options.repo_root / "ScaleRetarget/retargeted_dataset" / options.run_name
    processed_dir = options.repo_root / "ScaleRetarget/retargeted_dataset" / f"{options.run_name}_processed"
    retargeted_files = expected_retargeted_files(prepared_files, prepared_dir, retargeted_dir)
    processed_files = expected_processed_files(retargeted_files, retargeted_dir, processed_dir)
    for command in commands:
        previous_retarget_stats: dict[Path, tuple[int, int, int]] = {}
        preserved_retarget_stats: dict[Path, tuple[int, int, int]] = {}
        retarget_pairs_to_record: list[tuple[Path, Path]] = []
        if command.name == "retarget":
            motion_pairs = list(
                zip(prepared_files, retargeted_files, strict=True)
            )
            if options.force_data:
                retarget_pairs_to_record = motion_pairs
                for _prepared_file, retargeted_file in motion_pairs:
                    if retargeted_file.is_symlink():
                        retargeted_file.unlink()
                    elif retargeted_file.exists() and not retargeted_file.is_file():
                        raise RuntimeError(
                            f"Cannot replace non-file retarget output: {retargeted_file}"
                        )
                    elif retargeted_file.is_file():
                        previous_retarget_stats[retargeted_file] = _file_stat_signature(
                            retargeted_file
                        )
            else:
                existing_pairs: list[tuple[Path, Path]] = []
                missing_pairs: list[tuple[Path, Path]] = []
                for prepared_file, retargeted_file in motion_pairs:
                    if retargeted_file.is_symlink():
                        raise RuntimeError(
                            f"Refusing retargeted output symlink {retargeted_file}; "
                            "rerun with --force-data to replace the link safely"
                        )
                    if not retargeted_file.exists():
                        missing_pairs.append((prepared_file, retargeted_file))
                        continue
                    if not retargeted_file.is_file() or retargeted_file.stat().st_size <= 0:
                        raise RuntimeError(
                            f"Cached retargeted output is not a non-empty regular file: "
                            f"{retargeted_file}; rerun with --force-data"
                        )
                    existing_pairs.append((prepared_file, retargeted_file))

                if existing_pairs:
                    existing_prepared = [pair[0] for pair in existing_pairs]
                    existing_retargeted = [pair[1] for pair in existing_pairs]
                    stale_pairs = newer_inputs(existing_prepared, existing_retargeted)
                    if stale_pairs:
                        stale_input, stale_output = stale_pairs[0]
                        raise RuntimeError(
                            f"Prepared input {stale_input} is newer than cached "
                            f"retargeted output {stale_output}; rerun with --force-data "
                            "or use a new run-name"
                        )
                    for retargeted_file in existing_retargeted:
                        try:
                            validate_retargeted_motion(retargeted_file)
                        except Exception as exc:
                            raise RuntimeError(
                                f"Cached retargeted output is invalid: {retargeted_file}; "
                                "rerun with --force-data"
                            ) from exc
                    missing_provenance = verify_retarget_provenance(
                        existing_prepared,
                        existing_retargeted,
                        retarget_fingerprint,
                    )
                    if missing_provenance:
                        missing_set = set(missing_provenance)
                        unverified_pairs = [
                            pair
                            for pair in existing_pairs
                            if _retarget_source_digest_path(pair[1]) in missing_set
                            or _retarget_pipeline_digest_path(pair[1]) in missing_set
                        ]
                        if not options.adopt_legacy_retarget:
                            raise RuntimeError(
                                f"Cached retargeted output {unverified_pairs[0][1]} is "
                                "missing provenance; its source cannot be verified. "
                                "Rerun with --force-data, use a new run-name, or explicitly "
                                "accept this uncertainty with --adopt-legacy-retarget"
                            )
                        record_retarget_provenance(
                            [pair[0] for pair in unverified_pairs],
                            [pair[1] for pair in unverified_pairs],
                            retarget_fingerprint,
                        )
                        print(
                            "[ADOPT-UNVERIFIED] retarget: legacy cache was explicitly "
                            "bound to the current input and pipeline",
                            flush=True,
                        )
                    preserved_retarget_stats = {
                        path: _file_stat_signature(path)
                        for path in existing_retargeted
                    }

                if not missing_pairs:
                    print(
                        f"[SKIP] retarget: {len(retargeted_files)} output(s) "
                        "already complete",
                        flush=True,
                    )
                    continue
                retarget_pairs_to_record = missing_pairs
        if command.name == "package":
            summaries = [validate_retargeted_motion(path) for path in retargeted_files]
            total_frames = sum(int(summary["frames"]) for summary in summaries)
            print(
                f"[OK] retarget: {len(summaries)} motion(s), {total_frames} frame(s)",
                flush=True,
            )
        force_package = options.force_data or options.force_package
        if command.name == "package" and not force_package:
            if processed_outputs_are_current(
                processed_files,
                retargeted_files,
                expected_pipeline_fingerprint=package_fingerprint,
            ):
                print(
                    f"[SKIP] package: {len(processed_files)} current output(s)",
                    flush=True,
                )
                continue
            if _files_complete(processed_files):
                print(
                    "[STALE] package: cached outputs lack current provenance or "
                    "do not match their source; rebuilding",
                    flush=True,
                )
        if command.name == "yaml":
            summaries = [
                validate_processed_motion(
                    processed_path,
                    source_path=retargeted_path,
                    require_current_format=True,
                    expected_pipeline_fingerprint=package_fingerprint,
                )
                for processed_path, retargeted_path in zip(
                    processed_files, retargeted_files, strict=True
                )
            ]
            total_frames = sum(int(summary["frames"]) for summary in summaries)
            print(
                f"[OK] validate: {len(summaries)} motion(s), {total_frames} packed frame(s)",
                flush=True,
            )
        command_argv = command.argv
        if command.name == "package":
            command_argv += ("--pipeline_fingerprint", package_fingerprint)
            command_argv += tuple(
                token
                for motion_path in retargeted_files
                for token in ("--motion_file", str(motion_path))
            )
        elif command.name == "yaml":
            command_argv += tuple(
                token
                for motion_path in processed_files
                for token in ("--motion_file", str(motion_path))
            )
        if command.name == "play" and "__LATEST__" in command_argv:
            load_run = command_argv[command_argv.index("--load_run") + 1]
            run_dir = (
                options.repo_root
                / "ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp"
                / load_run
            )
            command_argv = tuple(
                latest_checkpoint(run_dir).name if token == "__LATEST__" else token
                for token in command_argv
            )
        print(f"[RUN] {command.name}: {shlex.join(command_argv)}", flush=True)
        subprocess.run(command_argv, cwd=command.cwd, env=command_env, check=True)
        if command.name == "retarget":
            if not _files_complete(retargeted_files):
                raise RuntimeError(
                    "ScaleRetarget exited successfully but did not create every expected .pkl"
                )
            unchanged_outputs = [
                path
                for path, previous_stat in previous_retarget_stats.items()
                if _file_stat_signature(path) == previous_stat
            ]
            if unchanged_outputs:
                raise RuntimeError(
                    "ScaleRetarget exited successfully but did not refresh forced output: "
                    f"{unchanged_outputs[0]}"
                )
            changed_preserved_outputs = [
                path
                for path, previous_stat in preserved_retarget_stats.items()
                if _file_stat_signature(path) != previous_stat
            ]
            if changed_preserved_outputs:
                raise RuntimeError(
                    "ScaleRetarget unexpectedly changed a verified cached output: "
                    f"{changed_preserved_outputs[0]}; rerun with --force-data"
                )
            for retargeted_file in retargeted_files:
                validate_retargeted_motion(retargeted_file)
            if retarget_pairs_to_record:
                record_retarget_provenance(
                    [pair[0] for pair in retarget_pairs_to_record],
                    [pair[1] for pair in retarget_pairs_to_record],
                    retarget_fingerprint,
                )
    yaml_path = options.repo_root / "ScaleRetarget/retargeted_dataset" / f"{options.run_name}.yaml"
    print(f"[DONE] data pipeline complete: {yaml_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

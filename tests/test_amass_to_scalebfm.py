from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import hashlib
import zipfile

import numpy as np

from scripts import amass_to_scalebfm


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "amass_to_scalebfm.py"


def write_fake_pipeline_scripts(repo_root: Path) -> None:
    """Create tiny real subprocesses that model the three external stage contracts."""
    retarget_script = repo_root / "ScaleRetarget" / "retarget.py"
    package_script = repo_root / "ScaleTrack" / "scripts/pretrain/data_process/package_motions.py"
    yaml_script = repo_root / "ScaleTrack" / "scripts/pretrain/data_process/create_yaml.py"
    for script in (retarget_script, package_script, yaml_script):
        script.parent.mkdir(parents=True, exist_ok=True)

    # The production entry point probes both interpreters before touching data.
    # These empty packages model dependencies of the tiny fake stage scripts.
    for package_name in ("hydra", "loguru", "mink", "mujoco", "natsort", "omegaconf",
                         "qpsolvers", "smplx", "scaleretarget"):
        package = repo_root / "ScaleRetarget" / package_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
    fake_python_path = repo_root / "ScaleTrack/source/scaletrack"
    for package_name in ("isaaclab", "isaaclab_physx", "scaletrack", "my_rsl_rl"):
        package = fake_python_path / package_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
    for module_name in ("app", "assets", "scene", "sim"):
        (fake_python_path / "isaaclab" / f"{module_name}.py").touch()
    isaaclab_utils = fake_python_path / "isaaclab/utils"
    isaaclab_utils.mkdir()
    (isaaclab_utils / "__init__.py").touch()
    (isaaclab_utils / "math.py").touch()
    fake_robots = fake_python_path / "scaletrack/robots"
    fake_robots.mkdir()
    (fake_robots / "__init__.py").touch()
    (fake_robots / "g1_29dof.py").touch()

    retarget_script.write_text(
        """from pathlib import Path
import pickle
import sys
import numpy as np
args = dict(arg.split('=', 1) for arg in sys.argv[1:] if '=' in arg)
source = Path(args['data_path'])
target = Path(args['output_dir']) / source.name
overwrite = args.get('loader.config.overwrite', 'false').lower() == 'true'
manifest_path = args.get('loader.config.motion_manifest')
motions = (
    [source / line for line in Path(manifest_path).read_text().splitlines()]
    if manifest_path
    else source.rglob('*.npz')
)
for motion in motions:
    output = (target / motion.relative_to(source)).with_suffix('.pkl')
    if output.exists() and not overwrite:
        continue
    output.parent.mkdir(parents=True, exist_ok=True)
    quat = np.zeros((3, 4), dtype=np.float32)
    quat[:, 3] = 1.0
    with output.open('wb') as stream:
        pickle.dump({
            'fps': 30,
            'root_pos': np.zeros((3, 3), dtype=np.float32),
            'root_rot': quat,
            'dof_pos': np.zeros((3, 29), dtype=np.float32),
        }, stream)
""",
        encoding="utf-8",
    )
    package_script.write_text(
        """import argparse
import hashlib
from pathlib import Path
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--motion_file', action='append', dest='motion_files')
parser.add_argument('--pipeline_fingerprint', default='0' * 64)
args, _ = parser.parse_known_args()
source = Path(args.data_dir)
target = Path(args.output_dir)
target.mkdir(parents=True, exist_ok=True)
motions = [Path(path) for path in args.motion_files] if args.motion_files else source.rglob('*.pkl')
for motion in motions:
    relative_output = motion.relative_to(source).with_suffix('.npz')
    output = target / relative_output
    output.parent.mkdir(parents=True, exist_ok=True)
    quat = np.zeros((3, 30, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    np.savez(
        output, fps=50,
        joint_pos=np.zeros((3, 29), dtype=np.float32),
        joint_vel=np.zeros((3, 29), dtype=np.float32),
        body_pos_w=np.zeros((3, 30, 3), dtype=np.float32),
        body_quat_w=quat,
        body_lin_vel_w=np.zeros((3, 30, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((3, 30, 3), dtype=np.float32),
        reference_root_pos=np.zeros((3, 3), dtype=np.float32),
        reference_root_quat_w=quat[:, 0].copy(),
        format_version=np.array(3),
        quaternion_order=np.array('wxyz'),
        source_sha256=np.array(hashlib.sha256(motion.read_bytes()).hexdigest()),
        pipeline_fingerprint=np.array(args.pipeline_fingerprint),
    )
""",
        encoding="utf-8",
    )
    yaml_script.write_text(
        """import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--motion_file', action='append', dest='motion_files')
args = parser.parse_args()
motions = [Path(path) for path in args.motion_files] if args.motion_files else sorted(Path(args.data_dir).rglob('*.npz'))
output = Path(args.output_path)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(''.join(f'{path.stem}: {path}\\n' for path in motions), encoding='utf-8')
""",
        encoding="utf-8",
    )


class CommandLineContractTests(unittest.TestCase):
    def test_help_describes_opt_in_training(self) -> None:
        """Catch a missing CLI or a dangerous default that starts training implicitly."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--train", result.stdout)
        self.assertIn("默认不启动训练", result.stdout)

    def test_non_executable_python_is_rejected_even_for_a_dry_run(self) -> None:
        """Catch an unusable environment before creating or downloading data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk_stageii.npz"
            source.touch()
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            not_python = temp_path / "not-python"
            not_python.write_text("#!/bin/sh\n", encoding="utf-8")
            not_python.chmod(0o644)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "bad_interpreter",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    str(not_python),
                    "--isaaclab-python",
                    sys.executable,
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not executable", result.stdout)

    def test_environment_probe_reports_the_missing_dependency(self) -> None:
        """Catch a valid Python binary whose stage dependencies are incomplete."""
        missing_module = "bfm_dependency_that_must_not_exist"

        with self.assertRaisesRegex(RuntimeError, missing_module):
            amass_to_scalebfm.probe_python_environment(
                Path(sys.executable),
                ("numpy", missing_module),
                label="test environment",
                cwd=REPO_ROOT,
                env=os.environ.copy(),
            )

    def test_environment_probe_returns_a_reproducible_fingerprint(self) -> None:
        """Make dependency versions and imported code part of cache identity."""
        first = amass_to_scalebfm.probe_python_environment(
            Path(sys.executable),
            ("numpy",),
            label="test environment",
            cwd=REPO_ROOT,
            env=os.environ.copy(),
        )
        second = amass_to_scalebfm.probe_python_environment(
            Path(sys.executable),
            ("numpy",),
            label="test environment",
            cwd=REPO_ROOT,
            env=os.environ.copy(),
        )

        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.assertEqual(first, second)

    def test_environment_probe_hashes_internal_package_implementation(self) -> None:
        """Detect editable IsaacLab changes outside its package __init__.py."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "fingerprint_package"
            package_root.mkdir()
            (package_root / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
            implementation = package_root / "internal.py"
            implementation.write_text("VALUE = 1\n", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [temp_dir, env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)

            before = amass_to_scalebfm.probe_python_environment(
                Path(sys.executable),
                ("fingerprint_package",),
                tree_modules=("fingerprint_package",),
                label="test package tree",
                cwd=REPO_ROOT,
                env=env,
            )
            implementation.write_text("VALUE = 2\n", encoding="utf-8")
            after = amass_to_scalebfm.probe_python_environment(
                Path(sys.executable),
                ("fingerprint_package",),
                tree_modules=("fingerprint_package",),
                label="test package tree",
                cwd=REPO_ROOT,
                env=env,
            )

            self.assertNotEqual(before, after)

    def test_policy_dependencies_do_not_change_the_package_cache_identity(self) -> None:
        """Starting playback must not make an unchanged motion archive stale."""
        package_only, no_policy = amass_to_scalebfm.isaaclab_dependency_groups(
            needs_policy=False
        )
        package_for_play, policy = amass_to_scalebfm.isaaclab_dependency_groups(
            needs_policy=True
        )

        self.assertEqual(package_only, package_for_play)
        self.assertNotIn("my_rsl_rl", package_for_play)
        self.assertEqual(no_policy, ())
        self.assertIn("my_rsl_rl", policy)

    def test_default_command_plan_uses_two_environments_without_training(self) -> None:
        """Catch interpreter mixing and accidental training in the safe default plan."""
        self.assertTrue(
            hasattr(amass_to_scalebfm, "PipelineOptions"),
            "pipeline options model is missing",
        )
        options = amass_to_scalebfm.PipelineOptions(
            repo_root=Path("/repo"),
            source=Path("/motions"),
            run_name="demo",
            smplx_model=Path("/models/SMPLX_NEUTRAL.npz"),
            retarget_python=Path("/envs/retarget/bin/python"),
            isaaclab_python=Path("/envs/isaaclab/bin/python"),
        )

        try:
            commands = amass_to_scalebfm.build_command_plan(options)
        except Exception as exc:  # pragma: no cover - produces a readable RED phase
            self.fail(f"command planning failed: {exc}")

        self.assertEqual([command.name for command in commands], ["retarget", "package", "yaml"])
        self.assertEqual(commands[0].argv[0], "/envs/retarget/bin/python")
        self.assertIn("data_path=/repo/ScaleRetarget/dataset/demo", commands[0].argv)
        self.assertEqual(commands[1].argv[0], "/envs/isaaclab/bin/python")
        self.assertIn("--headless", commands[1].argv)
        self.assertNotIn("train.py", " ".join(token for command in commands for token in command.argv))

    def test_train_flag_appends_finetune_from_official_checkpoint(self) -> None:
        """Catch a train flag that is ignored or silently trains from random initialization."""
        options = amass_to_scalebfm.PipelineOptions(
            repo_root=Path("/repo"),
            source=Path("/motions"),
            run_name="demo",
            smplx_model=Path("/models/SMPLX_NEUTRAL.npz"),
            retarget_python=Path("/envs/retarget/bin/python"),
            isaaclab_python=Path("/envs/isaaclab/bin/python"),
            train=True,
        )

        commands = amass_to_scalebfm.build_command_plan(options)

        self.assertEqual(commands[-1].name, "train")
        self.assertIn("/repo/ScaleTrack/scripts/pretrain/rsl_rl/train.py", commands[-1].argv)
        self.assertIn("/repo/ScaleRetarget/retargeted_dataset/demo.yaml", commands[-1].argv)
        self.assertIn("demo_finetune", commands[-1].argv)
        self.assertIn("--resume", commands[-1].argv)
        self.assertIn("humanoid_transformer_m", commands[-1].argv)
        self.assertIn("model_22200.pt", commands[-1].argv)

    def test_play_flag_appends_gui_whole_body_playback(self) -> None:
        """Catch playback being headless or evaluating the wrong conditioning mode."""
        self.assertIn("play", amass_to_scalebfm.PipelineOptions.__dataclass_fields__)
        options = amass_to_scalebfm.PipelineOptions(
            repo_root=Path("/repo"),
            source=Path("/motions"),
            run_name="demo",
            smplx_model=Path("/models/SMPLX_NEUTRAL.npz"),
            retarget_python=Path("/envs/retarget/bin/python"),
            isaaclab_python=Path("/envs/isaaclab/bin/python"),
            play=True,
        )

        commands = amass_to_scalebfm.build_command_plan(options)

        self.assertEqual(commands[-1].name, "play")
        self.assertIn("/repo/ScaleTrack/scripts/pretrain/rsl_rl/play.py", commands[-1].argv)
        self.assertIn("--mode_index", commands[-1].argv)
        self.assertIn("7", commands[-1].argv)
        self.assertNotIn("--headless", commands[-1].argv)
        self.assertIn("--viz", commands[-1].argv)
        viz_index = commands[-1].argv.index("--viz")
        self.assertEqual(commands[-1].argv[viz_index + 1], "kit")
        self.assertIn("humanoid_transformer_m", commands[-1].argv)
        self.assertIn("model_22200.pt", commands[-1].argv)

    def test_dry_run_prints_data_stages_without_writing_or_training(self) -> None:
        """Catch dry-run side effects and regression to implicit training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            fake_repo = temp_path / "repo"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "dry_contract",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(fake_repo),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("[DRY-RUN] prepare", result.stdout)
            self.assertIn("[DRY-RUN] retarget", result.stdout)
            self.assertIn("[DRY-RUN] package", result.stdout)
            self.assertIn("[DRY-RUN] yaml", result.stdout)
            self.assertNotIn("[DRY-RUN] train", result.stdout)
            self.assertFalse(fake_repo.exists())

    def test_public_sample_dry_run_needs_no_input_file(self) -> None:
        """Catch public smoke-test mode still requiring licensed AMASS input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            fake_repo = temp_path / "repo"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--public-sample",
                    "--run-name",
                    "public_contract",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(fake_repo),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("[DRY-RUN] download", result.stdout)
            self.assertIn("amass_sample.npz", result.stdout)
            self.assertFalse(fake_repo.exists())

    def test_training_cli_overrides_reach_the_train_command(self) -> None:
        """Catch user-selected training budgets or checkpoints being ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk_stageii.npz"
            source.touch()
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "train_contract",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(temp_path / "repo"),
                    "--train",
                    "--train-run-name",
                    "custom_finetune",
                    "--train-num-envs",
                    "64",
                    "--max-iterations",
                    "12",
                    "--seed",
                    "9",
                    "--base-run",
                    "base_policy",
                    "--base-checkpoint",
                    "model_10.pt",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            train_line = next(line for line in result.stdout.splitlines() if "[DRY-RUN] train" in line)
            self.assertIn("--run_name custom_finetune", train_line)
            self.assertIn("--num_envs 64", train_line)
            self.assertIn("--max_iterations 12", train_line)
            self.assertIn("--seed 9", train_line)
            self.assertIn("--load_run base_policy", train_line)
            self.assertIn("--checkpoint model_10.pt", train_line)

    def test_playback_cli_options_reach_the_play_command(self) -> None:
        """Catch GUI/headless, mode, and recording choices being dropped by the wrapper."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk_stageii.npz"
            source.touch()
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "play_contract",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(temp_path / "repo"),
                    "--play",
                    "--play-run",
                    "trained_policy",
                    "--play-checkpoint",
                    "model_900.pt",
                    "--mode-index",
                    "4",
                    "--local-tracking",
                    "--record-video",
                    "--video-length",
                    "123",
                    "--play-headless",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            play_line = next(line for line in result.stdout.splitlines() if "[DRY-RUN] play" in line)
            self.assertIn("--load_run trained_policy", play_line)
            self.assertIn("--checkpoint model_900.pt", play_line)
            self.assertIn("--mode_index 4", play_line)
            self.assertIn("--local_tracking", play_line)
            self.assertIn("--video --video_length 123", play_line)
            self.assertIn("--headless", play_line)

    def test_data_parallelism_and_device_overrides_reach_commands(self) -> None:
        """Catch full-dataset tuning options being accepted but not forwarded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk_stageii.npz"
            source.touch()
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "scale_contract",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(temp_path / "repo"),
                    "--retarget-workers",
                    "4",
                    "--package-num-envs",
                    "8",
                    "--package-loader-workers",
                    "3",
                    "--output-fps",
                    "50",
                    "--device",
                    "cuda:1",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            lines = result.stdout.splitlines()
            retarget_line = next(line for line in lines if "[DRY-RUN] retarget" in line)
            package_line = next(line for line in lines if "[DRY-RUN] package" in line)
            self.assertIn("multi_process=True", retarget_line)
            self.assertIn("num_workers=4", retarget_line)
            self.assertIn("--num_envs 8", package_line)
            self.assertIn("--loader_workers 3", package_line)
            self.assertIn("--output_fps 50", package_line)
            self.assertIn("--device cuda:1", package_line)

    def test_workflow_rejects_fps_that_the_motion_loader_cannot_read(self) -> None:
        """Reject non-50-Hz data before spending time in IsaacLab packaging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk_stageii.npz"
            source.touch()
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            fake_repo = temp_path / "repo"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "invalid_fps",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(fake_repo),
                    "--output-fps",
                    "60",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only supports 50 FPS", result.stdout)
            self.assertFalse(fake_repo.exists())

    def test_non_positive_stage_sizes_are_rejected_before_any_output(self) -> None:
        """Catch invalid resource settings before launching either environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk_stageii.npz"
            source.touch()
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            fake_repo = temp_path / "repo"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "invalid_size",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(fake_repo),
                    "--package-num-envs",
                    "0",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("positive integer", result.stdout)
            self.assertFalse(fake_repo.exists())

    def test_force_data_rebuild_disables_stage_skips_and_overwrites_retargeting(self) -> None:
        """Catch a corrupt cached artifact having no supported rebuild path."""
        self.assertIn("force_data", amass_to_scalebfm.PipelineOptions.__dataclass_fields__)
        options = amass_to_scalebfm.PipelineOptions(
            repo_root=Path("/repo"),
            source=Path("/motions"),
            run_name="demo",
            smplx_model=Path("/models/SMPLX_NEUTRAL.npz"),
            retarget_python=Path("/envs/retarget/bin/python"),
            isaaclab_python=Path("/envs/isaaclab/bin/python"),
            force_data=True,
        )

        commands = amass_to_scalebfm.build_command_plan(options)

        self.assertIn("loader.config.overwrite=true", commands[0].argv)

    def test_force_data_rejects_a_retargeter_that_did_not_refresh_outputs(self) -> None:
        """Do not bind new source provenance to a silently reused old trajectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--run-name",
                "force_refresh",
                "--smplx-model",
                str(model),
                "--retarget-python",
                sys.executable,
                "--isaaclab-python",
                sys.executable,
                "--repo-root",
                str(fake_repo),
            ]
            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)
            retarget_script = fake_repo / "ScaleRetarget/retarget.py"
            retarget_script.write_text("# injected successful no-op\n", encoding="utf-8")

            second = subprocess.run(
                [*command, "--force-data"],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(second.returncode, 0)
            self.assertIn("did not refresh", second.stdout)

    def test_force_data_replaces_a_retarget_symlink_without_touching_its_target(self) -> None:
        """Keep ScaleRetarget's direct joblib write away from a symlink victim."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            victim = temp_path / "victim.pkl"
            victim.write_bytes(b"preserve-me")
            output = (
                fake_repo
                / "ScaleRetarget/retargeted_dataset/safe_force/walk_stageii.pkl"
            )
            output.parent.mkdir(parents=True)
            output.symlink_to(victim)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "safe_force",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(fake_repo),
                    "--force-data",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertFalse(output.is_symlink())
            self.assertEqual(victim.read_bytes(), b"preserve-me")

    def test_local_tracking_rejects_a_mode_without_the_root(self) -> None:
        """Catch invalid local tracking before Isaac Sim spends time starting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk_stageii.npz"
            source.touch()
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "invalid_mode",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--play",
                    "--mode-index",
                    "1",
                    "--local-tracking",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("local tracking requires a root-containing mode", result.stdout)

    def test_missing_base_checkpoint_fails_before_creating_data_outputs(self) -> None:
        """Catch a long data run followed by a preventable missing-checkpoint failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "missing_checkpoint",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(fake_repo),
                    "--train",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Base checkpoint does not exist", result.stdout)
            self.assertFalse((fake_repo / "ScaleRetarget/dataset/missing_checkpoint").exists())

    def test_default_cli_executes_the_three_data_stages(self) -> None:
        """Catch a CLI that plans correctly but never executes its data pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "integration",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(fake_repo),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            yaml_path = fake_repo / "ScaleRetarget/retargeted_dataset/integration.yaml"
            packed_path = fake_repo / "ScaleRetarget/retargeted_dataset/integration_processed/walk_stageii.npz"
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue(packed_path.is_file(), result.stdout)
            self.assertTrue(yaml_path.is_file(), result.stdout)
            self.assertIn(str(packed_path), yaml_path.read_text(encoding="utf-8"))
            self.assertNotIn("[RUN] train", result.stdout)
            self.assertIn("[DONE] data pipeline complete", result.stdout)
            self.assertIn(str(yaml_path), result.stdout)

    def test_completed_data_stages_are_skipped_on_rerun(self) -> None:
        """Catch expensive shape optimization and IsaacLab packaging being repeated needlessly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--run-name",
                "resume_contract",
                "--smplx-model",
                str(model),
                "--retarget-python",
                sys.executable,
                "--isaaclab-python",
                sys.executable,
                "--repo-root",
                str(fake_repo),
            ]
            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)

            second = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn("[SKIP] retarget", second.stdout)
            self.assertIn("[SKIP] package", second.stdout)
            self.assertIn("[RUN] yaml", second.stdout)

    def test_partial_resume_never_certifies_an_unverified_existing_retarget(self) -> None:
        """Do not attach current provenance to an old pkl skipped by ScaleRetarget."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source_dir = temp_path / "motions"
            source_dir.mkdir()
            for name in ("a_stageii.npz", "b_stageii.npz"):
                np.savez(
                    source_dir / name,
                    root_orient=np.zeros((3, 3), dtype=np.float32),
                    pose_body=np.zeros((3, 63), dtype=np.float32),
                    trans=np.zeros((3, 3), dtype=np.float32),
                    betas=np.zeros(16, dtype=np.float32),
                    gender=np.array("neutral"),
                    mocap_frame_rate=np.array(60.0),
                )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source_dir),
                "--run-name",
                "partial_provenance",
                "--smplx-model",
                str(model),
                "--retarget-python",
                sys.executable,
                "--isaaclab-python",
                sys.executable,
                "--repo-root",
                str(fake_repo),
            ]
            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)

            retargeted_dir = (
                fake_repo
                / "ScaleRetarget/retargeted_dataset/partial_provenance"
            )
            existing = retargeted_dir / "a_stageii.pkl"
            missing = retargeted_dir / "b_stageii.pkl"
            existing.with_suffix(".pkl.source.sha256").unlink()
            existing.with_suffix(".pkl.pipeline.sha256").unlink()
            missing.unlink()
            missing.with_suffix(".pkl.source.sha256").unlink()
            missing.with_suffix(".pkl.pipeline.sha256").unlink()

            resumed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(resumed.returncode, 0, resumed.stdout)
            self.assertIn("missing provenance", resumed.stdout)
            self.assertFalse(existing.with_suffix(".pkl.source.sha256").exists())

    def test_partial_resume_keeps_verified_outputs_and_builds_missing_ones(self) -> None:
        """Resume a valid mixed cache without overwriting the completed pkl."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source_dir = temp_path / "motions"
            source_dir.mkdir()
            for name in ("a_stageii.npz", "b_stageii.npz"):
                np.savez(
                    source_dir / name,
                    root_orient=np.zeros((3, 3), dtype=np.float32),
                    pose_body=np.zeros((3, 63), dtype=np.float32),
                    trans=np.zeros((3, 3), dtype=np.float32),
                    betas=np.zeros(16, dtype=np.float32),
                    gender=np.array("neutral"),
                    mocap_frame_rate=np.array(60.0),
                )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source_dir),
                "--run-name",
                "partial_valid",
                "--smplx-model",
                str(model),
                "--retarget-python",
                sys.executable,
                "--isaaclab-python",
                sys.executable,
                "--repo-root",
                str(fake_repo),
            ]
            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)

            retargeted_dir = fake_repo / "ScaleRetarget/retargeted_dataset/partial_valid"
            existing = retargeted_dir / "a_stageii.pkl"
            missing = retargeted_dir / "b_stageii.pkl"
            existing_bytes = existing.read_bytes()
            for path in (
                missing,
                missing.with_suffix(".pkl.source.sha256"),
                missing.with_suffix(".pkl.pipeline.sha256"),
            ):
                path.unlink()

            resumed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            self.assertEqual(existing.read_bytes(), existing_bytes)
            self.assertTrue(missing.is_file())
            self.assertTrue(missing.with_suffix(".pkl.source.sha256").is_file())

    def test_legacy_retarget_cache_without_provenance_is_not_implicitly_adopted(self) -> None:
        """Do not claim an unverifiable old .pkl came from the current AMASS input."""
        import joblib

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            output = (
                fake_repo
                / "ScaleRetarget/retargeted_dataset/unverified/walk_stageii.pkl"
            )
            output.parent.mkdir(parents=True)
            root_rot = np.zeros((3, 4), dtype=np.float32)
            root_rot[:, 3] = 1.0
            joblib.dump(
                {
                    "fps": 30,
                    "root_pos": np.zeros((3, 3), dtype=np.float32),
                    "root_rot": root_rot,
                    "dof_pos": np.zeros((3, 29), dtype=np.float32),
                },
                output,
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "unverified",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(fake_repo),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing provenance", result.stdout)
            self.assertIn("--force-data", result.stdout)

            adopted = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--run-name",
                    "unverified",
                    "--smplx-model",
                    str(model),
                    "--retarget-python",
                    sys.executable,
                    "--isaaclab-python",
                    sys.executable,
                    "--repo-root",
                    str(fake_repo),
                    "--adopt-legacy-retarget",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(adopted.returncode, 0, adopted.stdout)
            self.assertIn("[ADOPT-UNVERIFIED]", adopted.stdout)

    def test_yaml_rerun_excludes_orphaned_processed_files(self) -> None:
        """Keep stale artifacts on disk from silently entering a training dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--run-name",
                "exact_yaml",
                "--smplx-model",
                str(model),
                "--retarget-python",
                sys.executable,
                "--isaaclab-python",
                sys.executable,
                "--repo-root",
                str(fake_repo),
            ]
            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)
            stale = (
                fake_repo
                / "ScaleRetarget/retargeted_dataset/exact_yaml_processed/orphan/old.npz"
            )
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"not-a-motion")

            second = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            yaml_path = (
                fake_repo / "ScaleRetarget/retargeted_dataset/exact_yaml.yaml"
            )
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertNotIn(str(stale), yaml_path.read_text(encoding="utf-8"))

    def test_force_package_rebuilds_only_simulator_outputs(self) -> None:
        """Allow packaging fixes without repeating expensive AMASS retargeting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--run-name",
                "repackage_contract",
                "--smplx-model",
                str(model),
                "--retarget-python",
                sys.executable,
                "--isaaclab-python",
                sys.executable,
                "--repo-root",
                str(fake_repo),
            ]
            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)

            second = subprocess.run(
                [*command, "--force-package"],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn("[SKIP] retarget", second.stdout)
            self.assertIn("[RUN] package", second.stdout)
            self.assertNotIn("[SKIP] package", second.stdout)

    def test_newer_native_source_cannot_silently_reuse_old_retargeting(self) -> None:
        """Require explicit force before replacing a stale expensive stage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            fields = {
                "root_orient": np.zeros((3, 3), dtype=np.float32),
                "pose_body": np.zeros((3, 63), dtype=np.float32),
                "trans": np.zeros((3, 3), dtype=np.float32),
                "betas": np.zeros(16, dtype=np.float32),
                "gender": np.array("neutral"),
                "mocap_frame_rate": np.array(60.0),
            }
            np.savez(source, **fields)
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--run-name",
                "stale_retarget",
                "--smplx-model",
                str(model),
                "--retarget-python",
                sys.executable,
                "--isaaclab-python",
                sys.executable,
                "--repo-root",
                str(fake_repo),
            ]
            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)
            retargeted = (
                fake_repo
                / "ScaleRetarget/retargeted_dataset/stale_retarget/walk_stageii.pkl"
            )
            newer_ns = retargeted.stat().st_mtime_ns + 2_000_000_000
            os.utime(source, ns=(newer_ns, newer_ns))

            second = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(second.returncode, 0)
            self.assertIn("--force-data", second.stdout)
            self.assertIn("newer than cached retargeted output", second.stdout)

    def test_changed_native_content_is_detected_even_when_mtime_is_restored(self) -> None:
        """Use content provenance instead of trusting timestamps alone."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            fields = {
                "root_orient": np.zeros((3, 3), dtype=np.float32),
                "pose_body": np.zeros((3, 63), dtype=np.float32),
                "trans": np.zeros((3, 3), dtype=np.float32),
                "betas": np.zeros(16, dtype=np.float32),
                "gender": np.array("neutral"),
                "mocap_frame_rate": np.array(60.0),
            }
            np.savez(source, **fields)
            original_times = source.stat()
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.touch()
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--run-name",
                "hashed_retarget",
                "--smplx-model",
                str(model),
                "--retarget-python",
                sys.executable,
                "--isaaclab-python",
                sys.executable,
                "--repo-root",
                str(fake_repo),
            ]
            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)

            changed_fields = dict(fields)
            changed_fields["trans"] = np.ones((3, 3), dtype=np.float32)
            np.savez(source, **changed_fields)
            os.utime(
                source,
                ns=(original_times.st_atime_ns, original_times.st_mtime_ns),
            )
            second = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(second.returncode, 0)
            self.assertIn("content no longer matches", second.stdout)
            self.assertIn("--force-data", second.stdout)

    def test_changed_retarget_model_invalidates_the_cached_trajectory(self) -> None:
        """Include the SMPL-X model and retarget implementation in provenance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.write_bytes(b"model-v1")
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--run-name",
                "model_provenance",
                "--smplx-model",
                str(model),
                "--retarget-python",
                sys.executable,
                "--isaaclab-python",
                sys.executable,
                "--repo-root",
                str(fake_repo),
            ]
            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)
            model.write_bytes(b"model-v2")

            second = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(second.returncode, 0)
            self.assertIn("retarget pipeline fingerprint", second.stdout)
            self.assertIn("--force-data", second.stdout)

    def test_changed_package_code_rebuilds_the_cached_archive(self) -> None:
        """Bind processed FK data to package code, robot assets, and IsaacLab."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            write_fake_pipeline_scripts(fake_repo)
            source = temp_path / "walk_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 63), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )
            model = temp_path / "SMPLX_NEUTRAL.npz"
            model.write_bytes(b"model")
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--run-name",
                "package_provenance",
                "--smplx-model",
                str(model),
                "--retarget-python",
                sys.executable,
                "--isaaclab-python",
                sys.executable,
                "--repo-root",
                str(fake_repo),
            ]
            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)
            package_script = (
                fake_repo
                / "ScaleTrack/scripts/pretrain/data_process/package_motions.py"
            )
            package_script.write_text(
                package_script.read_text(encoding="utf-8") + "\n# semantic change\n",
                encoding="utf-8",
            )

            second = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn("[STALE] package", second.stdout)
            self.assertIn("[RUN] package", second.stdout)


class AmassPreparationTests(unittest.TestCase):
    def test_download_verifies_the_pinned_sample_checksum(self) -> None:
        """Catch a partial or silently changed public sample before NumPy opens it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "remote.npz"
            source.write_bytes(b"official-public-sample")
            destination = temp_path / "downloads" / "sample.npz"
            checksum = hashlib.sha256(b"official-public-sample").hexdigest()

            self.assertTrue(
                hasattr(amass_to_scalebfm, "download_verified_file"),
                "verified downloader is missing",
            )
            result = amass_to_scalebfm.download_verified_file(
                source.as_uri(), destination, checksum
            )

            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), b"official-public-sample")
            self.assertFalse(destination.with_suffix(".npz.part").exists())

    def test_download_does_not_follow_a_predictable_partial_symlink(self) -> None:
        """Prevent a stale .part symlink from redirecting download writes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "remote.npz"
            payload = b"verified-payload"
            source.write_bytes(payload)
            destination = temp_path / "downloads" / "sample.npz"
            destination.parent.mkdir()
            victim = temp_path / "victim.txt"
            victim.write_bytes(b"preserve-me")
            destination.with_suffix(".npz.part").symlink_to(victim)

            result = amass_to_scalebfm.download_verified_file(
                source.as_uri(), destination, hashlib.sha256(payload).hexdigest()
            )

            self.assertEqual(result.read_bytes(), payload)
            self.assertEqual(victim.read_bytes(), b"preserve-me")

    def test_download_rejects_an_existing_destination_symlink(self) -> None:
        """Do not accept a cached public sample redirected outside its data root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "remote.npz"
            payload = b"verified-payload"
            source.write_bytes(payload)
            victim = temp_path / "victim.npz"
            victim.write_bytes(payload)
            destination = temp_path / "downloads/sample.npz"
            destination.parent.mkdir()
            destination.symlink_to(victim)

            with self.assertRaisesRegex(ValueError, "symlink"):
                amass_to_scalebfm.download_verified_file(
                    source.as_uri(),
                    destination,
                    hashlib.sha256(payload).hexdigest(),
                )

            self.assertTrue(destination.is_symlink())
            self.assertEqual(victim.read_bytes(), payload)

    def test_empty_input_directory_is_rejected(self) -> None:
        """Catch an empty dataset launching the expensive retargeter with zero motions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "empty"
            source.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "No AMASS .npz motions"):
                amass_to_scalebfm.prepare_amass_dataset(source, Path(temp_dir) / "prepared")

    def test_legacy_public_sample_is_converted_to_stageii_fields(self) -> None:
        """Catch incorrect SMPL pose slicing or the wrong AMASS frame-rate key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk.npz"
            prepared_dir = temp_path / "prepared"
            poses = np.arange(3 * 156, dtype=np.float32).reshape(3, 156)
            np.savez(
                source,
                poses=poses,
                trans=np.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    dtype=np.float32,
                ),
                betas=np.arange(16, dtype=np.float32),
                gender=np.array("female"),
                mocap_framerate=np.array(120.0),
            )

            self.assertTrue(
                hasattr(amass_to_scalebfm, "prepare_amass_dataset"),
                "AMASS preparation entry point is missing",
            )
            prepared_files = amass_to_scalebfm.prepare_amass_dataset(source, prepared_dir)

            self.assertEqual(prepared_files, [prepared_dir / "walk_stageii.npz"])
            with np.load(prepared_files[0], allow_pickle=False) as converted:
                self.assertEqual(
                    set(converted.files),
                    {"root_orient", "pose_body", "trans", "betas", "gender", "mocap_frame_rate"},
                )
                np.testing.assert_array_equal(converted["root_orient"], poses[:, :3])
                np.testing.assert_array_equal(converted["pose_body"], poses[:, 3:66])
                self.assertEqual(float(converted["mocap_frame_rate"]), 120.0)

    def test_legacy_sample_is_validated_before_conversion(self) -> None:
        """Reject a legacy sample that would immediately fail the Stage-II contract."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "short.npz"
            np.savez(
                source,
                poses=np.zeros((2, 156), dtype=np.float32),
                trans=np.zeros((2, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_framerate=np.array(60.0),
            )

            with self.assertRaisesRegex(ValueError, "at least 3 frames"):
                amass_to_scalebfm.prepare_amass_dataset(
                    source, temp_path / "prepared"
                )

    def test_native_stageii_directory_keeps_hierarchy_and_skips_stagei(self) -> None:
        """Catch flattening collisions and accidental ingestion of AMASS shape files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "amass"
            nested_dir = source_dir / "ACCAD" / "subject"
            nested_dir.mkdir(parents=True)
            motion = nested_dir / "walk_stageii.npz"
            stagei = nested_dir / "walk_stagei.npz"
            fields = {
                "root_orient": np.zeros((3, 3), dtype=np.float32),
                "pose_body": np.zeros((3, 63), dtype=np.float32),
                "trans": np.zeros((3, 3), dtype=np.float32),
                "betas": np.zeros(16, dtype=np.float32),
                "gender": np.array("neutral"),
                "mocap_frame_rate": np.array(60.0),
            }
            np.savez(motion, **fields)
            np.savez(stagei, **fields)
            prepared_dir = temp_path / "prepared"

            try:
                prepared_files = amass_to_scalebfm.prepare_amass_dataset(source_dir, prepared_dir)
            except Exception as exc:  # pragma: no cover - turns a missing behavior into a useful failure
                self.fail(f"native Stage-II preparation failed: {exc}")

            expected = prepared_dir / "ACCAD" / "subject" / "walk_stageii.npz"
            self.assertEqual(prepared_files, [expected])
            self.assertTrue(expected.is_symlink())
            self.assertEqual(expected.resolve(), motion.resolve())

    def test_reusing_run_name_with_a_different_native_source_is_rejected(self) -> None:
        """Catch stale prepared symlinks silently pointing at an earlier dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            prepared_dir = temp_path / "prepared"
            fields = {
                "root_orient": np.zeros((3, 3), dtype=np.float32),
                "pose_body": np.zeros((3, 63), dtype=np.float32),
                "trans": np.zeros((3, 3), dtype=np.float32),
                "betas": np.zeros(16, dtype=np.float32),
                "gender": np.array("neutral"),
                "mocap_frame_rate": np.array(60.0),
            }
            first_dir = temp_path / "first"
            second_dir = temp_path / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "same_stageii.npz"
            second = second_dir / "same_stageii.npz"
            np.savez(first, **fields)
            np.savez(second, **fields)
            amass_to_scalebfm.prepare_amass_dataset(first, prepared_dir)

            with self.assertRaisesRegex(FileExistsError, "different source"):
                amass_to_scalebfm.prepare_amass_dataset(second, prepared_dir)

    def test_force_never_replaces_a_native_input_that_is_already_prepared(self) -> None:
        """Catch --force-data deleting its own source and creating a self-link."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_dir = Path(temp_dir) / "prepared"
            prepared_dir.mkdir()
            source = prepared_dir / "walk_stageii.npz"
            fields = {
                "root_orient": np.zeros((3, 3), dtype=np.float32),
                "pose_body": np.zeros((3, 63), dtype=np.float32),
                "trans": np.zeros((3, 3), dtype=np.float32),
                "betas": np.zeros(16, dtype=np.float32),
                "gender": np.array("neutral"),
                "mocap_frame_rate": np.array(60.0),
            }
            np.savez(source, **fields)

            prepared_files = amass_to_scalebfm.prepare_amass_dataset(
                source, prepared_dir, force=True
            )

            self.assertEqual(prepared_files, [source])
            self.assertTrue(source.is_file())
            self.assertFalse(source.is_symlink())
            with np.load(source, allow_pickle=False) as preserved:
                np.testing.assert_array_equal(preserved["pose_body"], fields["pose_body"])

    def test_legacy_conversion_never_writes_through_an_existing_symlink(self) -> None:
        """Catch generated compatibility data overwriting a symlink target outside its run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk.npz"
            poses = np.zeros((3, 156), dtype=np.float32)
            np.savez(
                source,
                poses=poses,
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_framerate=np.array(60.0),
            )
            prepared_dir = temp_path / "prepared"
            prepared_dir.mkdir()
            victim = temp_path / "victim.txt"
            victim.write_text("keep me", encoding="utf-8")
            (prepared_dir / "walk_stageii.npz").symlink_to(victim)

            with self.assertRaisesRegex(FileExistsError, "generated output"):
                amass_to_scalebfm.prepare_amass_dataset(source, prepared_dir)

            self.assertEqual(victim.read_text(encoding="utf-8"), "keep me")

    def test_failed_force_conversion_preserves_the_previous_output(self) -> None:
        """Publish converted Stage-II files atomically after a successful write."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk.npz"
            np.savez(
                source,
                poses=np.zeros((3, 156), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_framerate=np.array(60.0),
            )
            prepared_dir = temp_path / "prepared"
            output = prepared_dir / "walk_stageii.npz"
            amass_to_scalebfm.prepare_amass_dataset(source, prepared_dir)
            previous_bytes = output.read_bytes()

            def partial_write_then_fail(stream, **_arrays):
                stream.write(b"partial")
                raise RuntimeError("injected write failure")

            with mock.patch.object(
                amass_to_scalebfm.np,
                "savez_compressed",
                side_effect=partial_write_then_fail,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected write failure"):
                    amass_to_scalebfm.prepare_amass_dataset(
                        source, prepared_dir, force=True
                    )

            self.assertEqual(output.read_bytes(), previous_bytes)

    def test_changed_legacy_source_requires_an_explicit_rebuild(self) -> None:
        """Do not silently reuse converted data after its legacy source changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk.npz"

            def write_source(value: float) -> None:
                poses = np.full((3, 156), value, dtype=np.float32)
                np.savez(
                    source,
                    poses=poses,
                    trans=np.zeros((3, 3), dtype=np.float32),
                    betas=np.zeros(16, dtype=np.float32),
                    gender=np.array("neutral"),
                    mocap_framerate=np.array(60.0),
                )

            write_source(0.0)
            prepared_dir = temp_path / "prepared"
            output = amass_to_scalebfm.prepare_amass_dataset(source, prepared_dir)[0]
            write_source(1.0)

            with self.assertRaisesRegex(ValueError, "legacy source changed"):
                amass_to_scalebfm.prepare_amass_dataset(source, prepared_dir)

            with np.load(output, allow_pickle=False) as preserved:
                self.assertEqual(float(preserved["pose_body"][0, 0]), 0.0)

    def test_malformed_stageii_pose_is_rejected_before_retargeting(self) -> None:
        """Catch malformed pose dimensions before an expensive shape-optimization run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "broken_stageii.npz"
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=np.zeros((3, 60), dtype=np.float32),
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )

            with self.assertRaisesRegex(ValueError, r"pose_body.*\(N, 63\)"):
                amass_to_scalebfm.prepare_amass_dataset(source, temp_path / "prepared")

    def test_unknown_npz_schema_has_an_actionable_error(self) -> None:
        """Catch opaque KeyError failures for SMPL+H or unrelated NumPy archives."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "unknown.npz"
            np.savez(source, xyz=np.zeros((3, 3), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "expected AMASS Stage-II fields"):
                amass_to_scalebfm.prepare_amass_dataset(source, Path(temp_dir) / "prepared")

    def test_non_finite_stageii_values_are_rejected(self) -> None:
        """Catch NaN trajectories before they poison retargeting and PPO training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "nan_stageii.npz"
            pose_body = np.zeros((3, 63), dtype=np.float32)
            pose_body[1, 10] = np.nan
            np.savez(
                source,
                root_orient=np.zeros((3, 3), dtype=np.float32),
                pose_body=pose_body,
                trans=np.zeros((3, 3), dtype=np.float32),
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(60.0),
            )

            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                amass_to_scalebfm.prepare_amass_dataset(source, Path(temp_dir) / "prepared")

    def test_empty_betas_and_unknown_gender_are_rejected_early(self) -> None:
        """Validate SMPL-X metadata before expensive shape optimization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base_fields = {
                "root_orient": np.zeros((3, 3), dtype=np.float32),
                "pose_body": np.zeros((3, 63), dtype=np.float32),
                "trans": np.zeros((3, 3), dtype=np.float32),
                "mocap_frame_rate": np.array(60.0),
            }
            empty_betas = temp_path / "empty_betas_stageii.npz"
            np.savez(
                empty_betas,
                **base_fields,
                betas=np.array([], dtype=np.float32),
                gender=np.array("neutral"),
            )
            unknown_gender = temp_path / "unknown_gender_stageii.npz"
            np.savez(
                unknown_gender,
                **base_fields,
                betas=np.zeros(16, dtype=np.float32),
                gender=np.array("robot"),
            )

            with self.assertRaisesRegex(ValueError, "betas must be a non-empty"):
                amass_to_scalebfm.prepare_amass_dataset(
                    empty_betas, temp_path / "prepared-empty"
                )
            with self.assertRaisesRegex(ValueError, "gender must be one of"):
                amass_to_scalebfm.prepare_amass_dataset(
                    unknown_gender, temp_path / "prepared-gender"
                )


class ArtifactValidationTests(unittest.TestCase):
    def test_corrupt_cached_zip_is_treated_as_stale(self) -> None:
        """Let the wrapper rebuild a truncated archive instead of crashing in resume."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            processed = temp_path / "walk.npz"
            processed.write_bytes(b"not-empty")
            source = temp_path / "walk.pkl"
            source.write_bytes(b"retargeted")
            with mock.patch.object(
                amass_to_scalebfm,
                "validate_processed_motion",
                side_effect=zipfile.BadZipFile("truncated"),
            ):
                current = amass_to_scalebfm.processed_outputs_are_current(
                    [processed],
                    [source],
                    expected_pipeline_fingerprint="a" * 64,
                )

            self.assertFalse(current)

    def test_processed_paths_preserve_hierarchy_without_flattening_collisions(self) -> None:
        """Keep distinct AMASS paths distinct after IsaacLab packaging."""
        retargeted_dir = Path("/data/retargeted")
        processed_dir = Path("/data/processed")
        inputs = [
            retargeted_dir / "a_b/c.pkl",
            retargeted_dir / "a/b_c.pkl",
        ]

        outputs = amass_to_scalebfm.expected_processed_files(
            inputs, retargeted_dir, processed_dir
        )

        self.assertEqual(
            outputs,
            [processed_dir / "a_b/c.npz", processed_dir / "a/b_c.npz"],
        )
        self.assertEqual(len(set(outputs)), len(outputs))

    def test_latest_checkpoint_uses_numeric_iteration_order(self) -> None:
        """Catch lexicographic selection choosing model_9.pt after model_100.pt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "model_9.pt").touch()
            (run_dir / "model_100.pt").touch()
            (run_dir / "model_bad.pt").touch()

            self.assertTrue(
                hasattr(amass_to_scalebfm, "latest_checkpoint"),
                "latest-checkpoint resolver is missing",
            )
            checkpoint = amass_to_scalebfm.latest_checkpoint(run_dir)

            self.assertEqual(checkpoint.name, "model_100.pt")

    def test_retargeted_motion_reports_frames_and_xyzw_quaternions(self) -> None:
        """Catch invalid G1 joint counts and malformed retargeted root rotations."""
        import joblib

        with tempfile.TemporaryDirectory() as temp_dir:
            motion = Path(temp_dir) / "walk.pkl"
            root_rot = np.zeros((4, 4), dtype=np.float32)
            root_rot[:, 3] = 1.0
            joblib.dump(
                {
                    "fps": 30,
                    "root_pos": np.zeros((4, 3), dtype=np.float32),
                    "root_rot": root_rot,
                    "dof_pos": np.zeros((4, 29), dtype=np.float32),
                },
                motion,
            )

            self.assertTrue(
                hasattr(amass_to_scalebfm, "validate_retargeted_motion"),
                "retargeted-motion validator is missing",
            )
            summary = amass_to_scalebfm.validate_retargeted_motion(motion)

            self.assertEqual(summary, {"frames": 4, "fps": 30.0, "dofs": 29, "quat_order": "xyzw"})

    def test_processed_motion_reports_frames_and_schema(self) -> None:
        """Catch packed files that cannot satisfy ScaleTrack's 29-DoF/30-body contract."""
        with tempfile.TemporaryDirectory() as temp_dir:
            motion = Path(temp_dir) / "walk.npz"
            quat = np.zeros((4, 30, 4), dtype=np.float32)
            quat[..., 0] = 1.0
            np.savez(
                motion,
                fps=50,
                joint_pos=np.zeros((4, 29), dtype=np.float32),
                joint_vel=np.zeros((4, 29), dtype=np.float32),
                body_pos_w=np.zeros((4, 30, 3), dtype=np.float32),
                body_quat_w=quat,
                body_lin_vel_w=np.zeros((4, 30, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((4, 30, 3), dtype=np.float32),
            )

            self.assertTrue(
                hasattr(amass_to_scalebfm, "validate_processed_motion"),
                "processed-motion validator is missing",
            )
            summary = amass_to_scalebfm.validate_processed_motion(motion)

            self.assertEqual(summary, {"frames": 4, "fps": 50.0, "joints": 29, "bodies": 30})

    def test_processed_motion_rejects_a_non_50_hz_archive(self) -> None:
        """Match validation to the hard-coded 50-Hz training loader contract."""
        with tempfile.TemporaryDirectory() as temp_dir:
            motion = Path(temp_dir) / "walk.npz"
            quat = np.zeros((4, 30, 4), dtype=np.float32)
            quat[..., 0] = 1.0
            np.savez(
                motion,
                fps=60,
                joint_pos=np.zeros((4, 29), dtype=np.float32),
                joint_vel=np.zeros((4, 29), dtype=np.float32),
                body_pos_w=np.zeros((4, 30, 3), dtype=np.float32),
                body_quat_w=quat,
                body_lin_vel_w=np.zeros((4, 30, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((4, 30, 3), dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "exactly 50 FPS"):
                amass_to_scalebfm.validate_processed_motion(motion)

    def test_current_processed_format_requires_matching_source_provenance(self) -> None:
        """Do not resume a packed artifact made from stale retargeted input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk.pkl"
            source.write_bytes(b"current-retargeted-motion")
            motion = temp_path / "walk.npz"
            quat = np.zeros((4, 30, 4), dtype=np.float32)
            quat[..., 0] = 1.0
            fields = {
                "fps": np.array(50),
                "joint_pos": np.zeros((4, 29), dtype=np.float32),
                "joint_vel": np.zeros((4, 29), dtype=np.float32),
                "body_pos_w": np.zeros((4, 30, 3), dtype=np.float32),
                "body_quat_w": quat,
                "body_lin_vel_w": np.zeros((4, 30, 3), dtype=np.float32),
                "body_ang_vel_w": np.zeros((4, 30, 3), dtype=np.float32),
                "reference_root_pos": np.zeros((4, 3), dtype=np.float32),
                "reference_root_quat_w": quat[:, 0].copy(),
                "format_version": np.array(3),
                "quaternion_order": np.array("wxyz"),
                "source_sha256": np.array(hashlib.sha256(source.read_bytes()).hexdigest()),
                "pipeline_fingerprint": np.array("a" * 64),
            }
            np.savez(motion, **fields)

            summary = amass_to_scalebfm.validate_processed_motion(
                motion,
                source_path=source,
                require_current_format=True,
            )
            self.assertEqual(summary["frames"], 4)

            source.write_bytes(b"changed-retargeted-motion")
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                amass_to_scalebfm.validate_processed_motion(
                    motion,
                    source_path=source,
                    require_current_format=True,
                )

    def test_current_processed_format_rejects_fractional_version(self) -> None:
        """Do not truncate a malformed format version and reuse the archive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk.pkl"
            source.write_bytes(b"retargeted-motion")
            motion = temp_path / "walk.npz"
            quat = np.zeros((4, 30, 4), dtype=np.float32)
            quat[..., 0] = 1.0
            np.savez(
                motion,
                fps=np.array(50),
                joint_pos=np.zeros((4, 29), dtype=np.float32),
                joint_vel=np.zeros((4, 29), dtype=np.float32),
                body_pos_w=np.zeros((4, 30, 3), dtype=np.float32),
                body_quat_w=quat,
                body_lin_vel_w=np.zeros((4, 30, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((4, 30, 3), dtype=np.float32),
                reference_root_pos=np.zeros((4, 3), dtype=np.float32),
                reference_root_quat_w=quat[:, 0].copy(),
                format_version=np.array(3.7),
                quaternion_order=np.array("wxyz"),
                source_sha256=np.array(hashlib.sha256(source.read_bytes()).hexdigest()),
                pipeline_fingerprint=np.array("a" * 64),
            )

            with self.assertRaisesRegex(ValueError, "format_version.*integer scalar"):
                amass_to_scalebfm.validate_processed_motion(
                    motion,
                    source_path=source,
                    require_current_format=True,
                    expected_pipeline_fingerprint="a" * 64,
                )

    def test_current_archive_rejects_fk_that_disagrees_with_reference_root(self) -> None:
        """Detect quaternion/FK regressions that still have valid shapes and norms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "walk.pkl"
            source.write_bytes(b"retargeted-motion")
            motion = temp_path / "walk.npz"
            quat = np.zeros((4, 30, 4), dtype=np.float32)
            quat[..., 0] = 1.0
            reference_root_pos = np.zeros((4, 3), dtype=np.float32)
            reference_root_pos[:, 0] = 0.25
            np.savez(
                motion,
                fps=50,
                joint_pos=np.zeros((4, 29), dtype=np.float32),
                joint_vel=np.zeros((4, 29), dtype=np.float32),
                body_pos_w=np.zeros((4, 30, 3), dtype=np.float32),
                body_quat_w=quat,
                body_lin_vel_w=np.zeros((4, 30, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((4, 30, 3), dtype=np.float32),
                reference_root_pos=reference_root_pos,
                reference_root_quat_w=quat[:, 0].copy(),
                format_version=np.array(3),
                quaternion_order=np.array("wxyz"),
                source_sha256=np.array(hashlib.sha256(source.read_bytes()).hexdigest()),
                pipeline_fingerprint=np.array("a" * 64),
            )

            with self.assertRaisesRegex(ValueError, "pelvis positions disagree"):
                amass_to_scalebfm.validate_processed_motion(
                    motion,
                    source_path=source,
                    require_current_format=True,
                )


if __name__ == "__main__":
    unittest.main()

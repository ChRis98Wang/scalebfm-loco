import argparse
import os
from pathlib import Path
import tempfile

import yaml


def create_yaml(
    data_dir: str,
    output_path: str,
    motion_files: list[str] | None = None,
) -> Path:
    """Create one YAML file containing every processed motion in data_dir."""
    data_path = Path(data_dir).resolve()
    if not data_path.is_dir():
        raise NotADirectoryError(f"Motion directory does not exist: {data_path}")

    if motion_files is None:
        selected_files = sorted(data_path.rglob("*.npz"))
    else:
        selected_files = [Path(path).resolve() for path in motion_files]
        for motion_file in selected_files:
            if not motion_file.is_file() or motion_file.suffix != ".npz":
                raise FileNotFoundError(f"Processed motion does not exist: {motion_file}")
            try:
                motion_file.relative_to(data_path)
            except ValueError as exc:
                raise ValueError(
                    f"Processed motion is outside data_dir: {motion_file}"
                ) from exc
        selected_files.sort()
    if not selected_files:
        raise FileNotFoundError(f"No processed .npz motion files found under: {data_path}")

    motions = {}
    for motion_file in selected_files:
        motion_name = motion_file.relative_to(data_path).with_suffix("").as_posix()
        if motion_name in motions:
            raise ValueError(
                f"Duplicate motion name '{motion_name}' found in:\n"
                f"  {motions[motion_name]}\n"
                f"  {motion_file}"
            )
        motions[motion_name] = str(motion_file)

    yaml_path = Path(output_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=yaml_path.parent,
            prefix=f".{yaml_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as yaml_file:
            temporary_path = Path(yaml_file.name)
            yaml.safe_dump(motions, yaml_file, sort_keys=False)
            yaml_file.flush()
            os.fsync(yaml_file.fileno())
        temporary_path.replace(yaml_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Saved {len(motions)} motions to: {yaml_path}")
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one YAML file for all processed motion files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing processed .npz motions.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="source/scaletrack/data/yaml_files/motions.yaml",
        help="Path of the YAML file to create.",
    )
    parser.add_argument(
        "--motion_file",
        action="append",
        dest="motion_files",
        help="Exact processed motion to include; repeat to exclude stale files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args_cli = parse_args()
    create_yaml(args_cli.data_dir, args_cli.output_path, args_cli.motion_files)

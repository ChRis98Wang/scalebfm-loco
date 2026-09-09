"""Content snapshots for evaluation inputs, without importing Torch or Kit."""

import hashlib
from importlib.util import find_spec
import json
from pathlib import Path


def locate_package_sources(names):
    """Locate all regular or namespace package portions without importing Kit."""
    roots = {}
    for name in names:
        spec = find_spec(name)
        locations = spec.submodule_search_locations if spec is not None else None
        # PEP 660 editable namespace finders append a virtual path-hook marker.
        # It is not a filesystem source tree, unlike the package portions.
        locations = [Path(path).resolve() for path in locations or () if Path(path).is_dir()]
        if not locations:
            raise ValueError(f"Cannot locate Python package sources for {name}")
        roots.update({f"{name}/{index}": path for index, path in enumerate(locations)})
    return roots


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _group(paths):
    files = {name: _file_record(path) for name, path in sorted(paths.items())}
    # The content digest is portable; absolute paths remain available for audits.
    hashes = {name: record["sha256"] for name, record in files.items()}
    encoded = json.dumps(hashes, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def snapshot_inputs(checkpoint_path, motion_file, code_roots):
    """Hash every referenced payload and actual Python source, including dirty files.

    Relative motion paths intentionally follow the working directory, matching
    ScaleTrack's motion loader. This is a content audit, not a dependency lock or
    a snapshot of simulator binaries, robot assets, or GPU/driver state.
    """
    import yaml

    with Path(motion_file).open(encoding="utf-8") as stream:
        index = yaml.safe_load(stream)
    if not isinstance(index, dict) or not index:
        raise ValueError("motion_file must contain a nonempty name-to-path mapping")
    if any(not isinstance(name, str) or not isinstance(path, str) for name, path in index.items()):
        raise ValueError("Motion names and paths must be strings")
    sources = {}
    for name, root in sorted(code_roots.items()):
        root = Path(root).resolve(strict=True)
        paths = sorted(root.rglob("*.py"))
        if not paths:
            raise ValueError(f"No Python sources found under {root}")
        sources.update({f"{name}/{path.relative_to(root).as_posix()}": path for path in paths})
    return {
        "checkpoint": _file_record(checkpoint_path),
        "motion_index": _file_record(motion_file),
        "motions": _group(index),
        "python_sources": _group(sources),
    }


def verify_unchanged(before, after):
    changed = [key for key in before.keys() | after.keys() if before.get(key) != after.get(key)]
    if changed:
        raise RuntimeError(f"Evaluation inputs changed during the run: {', '.join(sorted(changed))}")

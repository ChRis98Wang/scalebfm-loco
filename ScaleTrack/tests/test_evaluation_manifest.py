"""Changing motion bytes or uncommitted runtime code must invalidate an audit."""

import importlib.util
from importlib.machinery import ModuleSpec
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl/evaluation_manifest.py"


def manifest_module():
    assert MODULE_PATH.is_file(), "evaluation input snapshot implementation is missing"
    spec = importlib.util.spec_from_file_location("bfm_evaluation_manifest", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    checkpoint = tmp_path / "trusted.pt"
    checkpoint.write_bytes(b"checkpoint")
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"abc")
    index = tmp_path / "validation.yaml"
    index.write_text(json.dumps({"A/walk": str(motion)}), encoding="utf-8")
    code = tmp_path / "source"
    code.mkdir()
    (code / "task.py").write_text("gravity = -9.81\n", encoding="utf-8")
    return checkpoint, index, {"task": code}


def test_snapshot_hashes_motion_payloads_and_actual_python_sources(inputs):
    module = manifest_module()
    result = module.snapshot_inputs(*inputs)
    motion = result["motions"]["files"]["A/walk"]
    assert motion["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert motion["size_bytes"] == 3
    assert set(result["python_sources"]["files"]) == {"task/task.py"}
    module.verify_unchanged(result, module.snapshot_inputs(*inputs))


@pytest.mark.parametrize("change", ["motion", "source", "new_source", "checkpoint", "index"])
def test_mutation_is_rejected_even_when_git_head_is_unchanged(inputs, change):
    module = manifest_module()
    before = module.snapshot_inputs(*inputs)
    checkpoint, index, roots = inputs
    targets = {
        "motion": index.parent / "motion.npz",
        "source": roots["task"] / "task.py",
        "new_source": roots["task"] / "untracked.py",
        "checkpoint": checkpoint,
        "index": index,
    }
    with targets[change].open("ab") as stream:
        stream.write(b"\n ")
    with pytest.raises(RuntimeError, match="changed"):
        module.verify_unchanged(before, module.snapshot_inputs(*inputs))


def test_missing_motion_cannot_be_silently_left_out_of_manifest(inputs):
    module = manifest_module()
    _, index, _ = inputs
    (index.parent / "motion.npz").unlink()
    with pytest.raises(FileNotFoundError):
        module.snapshot_inputs(*inputs)


def test_relative_motion_paths_follow_the_loader_working_directory(inputs, monkeypatch):
    module = manifest_module()
    _, index, _ = inputs
    index.write_text('{"A/walk": "motion.npz"}', encoding="utf-8")
    monkeypatch.chdir(index.parent)
    manifest = module.snapshot_inputs(*inputs)
    assert manifest["motions"]["files"]["A/walk"]["path"] == str(index.parent / "motion.npz")


def test_source_discovery_ignores_generated_bytecode(inputs):
    module = manifest_module()
    before = module.snapshot_inputs(*inputs)
    cache = inputs[2]["task"] / "__pycache__"
    cache.mkdir()
    (cache / "task.cpython-312.pyc").write_bytes(b"generated cache")
    module.verify_unchanged(before, module.snapshot_inputs(*inputs))


def test_source_locator_handles_namespace_packages_without_an_init_file(tmp_path, monkeypatch):
    module = manifest_module()
    namespace = tmp_path / "bfm_test_namespace"
    namespace.mkdir()
    (namespace / "policy.py").write_text("hidden_size = 256\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert hasattr(module, "locate_package_sources"), "runtime source location must support namespace packages"
    roots = module.locate_package_sources(["bfm_test_namespace"])
    assert roots == {"bfm_test_namespace/0": namespace}


def test_source_locator_fails_closed_for_missing_packages():
    module = manifest_module()
    assert hasattr(module, "locate_package_sources"), "runtime package source preflight is missing"
    with pytest.raises(ValueError, match="bfm_nonexistent_eval_package"):
        module.locate_package_sources(["bfm_nonexistent_eval_package"])


def test_editable_namespace_virtual_hook_is_not_a_source_directory(tmp_path, monkeypatch):
    module = manifest_module()
    source = tmp_path / "policy"
    source.mkdir()
    spec = ModuleSpec("policy", loader=None, is_package=True)
    spec.submodule_search_locations = [str(source), "__editable__.policy.finder.__path_hook__"]
    monkeypatch.setattr(module, "find_spec", lambda name: spec)
    assert module.locate_package_sources(["policy"]) == {"policy/0": source}


def test_only_virtual_namespace_locations_cannot_pass_preflight(monkeypatch):
    module = manifest_module()
    spec = ModuleSpec("policy", loader=None, is_package=True)
    spec.submodule_search_locations = ["__editable__.policy.finder.__path_hook__"]
    monkeypatch.setattr(module, "find_spec", lambda name: spec)
    with pytest.raises(ValueError, match="policy"):
        module.locate_package_sources(["policy"])

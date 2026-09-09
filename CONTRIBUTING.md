# Contributing

This fork preserves the upstream ScaleRetarget / ScaleTrack / ScaleBridge layout.
Keep changes scoped and retain upstream authorship and third-party notices.

## Environments

- Retargeting: separate Python 3.11 environment and personally obtained motion/body-model inputs.
- Local training/playback: existing Python 3.12, IsaacLab 3.0 / Isaac Sim 6 development build.
- Upstream documented stack: Python 3.11, Isaac Sim 5.1, pinned IsaacLab revision;
  see `ScaleTrack/README.md`. That stack has not been re-verified for this fork's
  latest changes. Do not treat these combinations as interchangeable.

For the 2026-09-07 evaluation, IsaacLab's checkout `VERSION` is `3.0.0` at
`ffff603eafc6b74264a5261cc0183d6a65390d78`, while its Python extension/package
metadata is `isaaclab==6.1.14`. Isaac Sim package metadata is `6.0.1.0`.
These are distinct version fields, not interchangeable release labels; the
evaluation JSON records package versions and hashes the actual Python sources.

Do not reinstall or upgrade a working IsaacLab environment merely to run CPU tests.

## Tests

From the repository root, using the existing IsaacLab Python:

```bash
"$BFM_ISAAC_PYTHON" -m pytest ScaleTrack/tests tests/test_gui_launcher.py tests/test_ci_workflow.py tests/test_mask_report_compare.py -q
bash -n scripts/open_motion_browser.sh
git diff --check
```

These are not the entire repository's tests: retargeting tests need the separate
retarget environment and dependencies. Native Kit UI and physics validation are
integration checks, not evidence supplied by the CPU suite alone.

For the simulator-independent CI suite, use a separate Python 3.11 or 3.12
virtual environment (not the existing simulator or retarget environment):

```bash
python -m pip install -r requirements-ci.txt
python -m pytest tests/test_ci_workflow.py tests/test_gui_launcher.py ScaleTrack/tests/test_evaluation_cli.py ScaleTrack/tests/test_evaluation_manifest.py -q
python -m pytest tests/test_mask_report_compare.py -q
```

`.github/workflows/cpu-tests.yml` runs this explicit subset on both versions with
read-only permissions, full-SHA pinned actions, and a ten-minute job limit. It
does not install IsaacLab, fetch motion data/weights, or upload local artifacts.
The same subset was locally validated in an isolated Python 3.11 environment;
the GitHub-hosted workflow has not run yet. The report-comparison tests use
synthetic temporary JSON, not licensed clips or checkpoints. This is not complete
project coverage.

Real Kit/physics checks require the existing simulator, and robot playback also
requires trusted local data and weights. See
[target-object validation](docs/BFM_TARGET_OBJECT_VALIDATION.md) for their scope.
`ScaleTrack/tests/playback_robot_gui_smoke.py` takes the same CLI arguments as
`play.py` with `--motion_menu --target_object --num_envs 1 --mode_index 7 --viz kit`;
it exercises the actual player and closes automatically after 130 policy steps.
Still run it inside a bounded process group for setup-failure protection.

## Experiment discipline

- Preserve the training/validation split; do not move validation failures into training.
- Record exact checkpoint, index, seed, mode, horizon and simulator version.
- Compare policies under the same protocol. A successful GUI launch is not a quality evaluation.
- Write experiments to new paths under `logs/`; do not overwrite existing runs.
- Bound GPU jobs and clean their own process groups. Never use broad `pkill python`.
- Use regression tests for metric counting, failure paths and changes at simulator boundaries.

Before publishing, complete [the release checklist](docs/RELEASE_CHECKLIST.md).

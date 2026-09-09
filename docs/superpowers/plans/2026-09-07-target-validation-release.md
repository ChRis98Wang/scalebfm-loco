# Target Validation and CPU CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the approved physical-target GUI milestone with independent runtime evidence and portable CPU CI.

**Architecture:** Keep the object opt-in and reuse the existing policy observations and rewards. Keep simulator-independent CI separate from licensed motion data, checkpoints, and real Kit/contact integration checks.

**Tech Stack:** Existing IsaacLab/Kit, Python, pytest, PyYAML, GitHub Actions.

**Spec:** User-approved scope: physical configurable box, motion/reference-mode switching, reset-target-only, actual object status, separate object evaluation, and source-release preparation. The implementation and limits are documented in `docs/BFM_TARGET_OBJECT_VALIDATION.md`.

## Global Constraints

- Reuse existing IsaacLab; do not reinstall or upgrade it.
- Preserve the current dirty worktree, datasets, checkpoints, and old evaluation reports.
- No residual simulator processes: run bounded owned systemd units and verify cleanup.
- No publication, commits, pushes, license grants, or licensed-data CI uploads.
- Engineering contact parameters are not measured material calibration; object presence is not learned manipulation.

---

### Task 1: Simulator-independent CI

**Files:** Create `tests/test_ci_workflow.py`, `requirements-ci.txt`, and `.github/workflows/cpu-tests.yml`; update `CONTRIBUTING.md` and `docs/RELEASE_CHECKLIST.md`.

**Interfaces:** CI consumes only public source and synthetic fixtures. It runs launcher/CLI/manifest tests with Python 3.11 and 3.12, pytest and PyYAML; it produces only pass/fail, with no artifact upload or simulator installation.

- [x] Add regression tests requiring a read-only, bounded CPU workflow; full-SHA pinned actions; no persisted checkout credentials; and explicit simulator-free test paths.
- [x] Run `python3 -m pytest tests/test_ci_workflow.py -q`; require failure because the workflow is absent.
- [x] Add dependencies `pytest==8.4.2` and `PyYAML==6.0.3`; add workflow triggers `push`, `pull_request`, `workflow_dispatch`, permission `contents: read`, matrix `['3.11', '3.12']`, and `timeout-minutes: 10`.
- [x] The workflow runs the following exact command, then `bash -n scripts/open_motion_browser.sh` and `git diff --check`:

```bash
python -m pytest tests/test_ci_workflow.py tests/test_gui_launcher.py ScaleTrack/tests/test_evaluation_cli.py ScaleTrack/tests/test_evaluation_manifest.py -q
```

- [x] Run that command in an isolated temporary Python 3.11 environment with only `requirements-ci.txt`. Record local validation without claiming a GitHub-hosted run happened.
- [x] Update the release checklist: CI configuration and local validation done; hosted run, licenses, clean simulator installation, and publication still pending.

### Task 2: Real integration and handoff

**Files:** Existing `play.py`, `playback_ui.py`, `target_object.py`, `evaluate.py`, physics/UI smoke tests; create `configs/target_objects/box_engineering.json` and `docs/BFM_TARGET_OBJECT_VALIDATION.md`; update README and motion browser instructions.

**Interfaces:** `--target_object` enables the box, `--target_config` loads local physical parameters. `--target-object` is the shell-launcher spelling. GUI mode selection queues a change until Apply; target reset changes no robot state. Evaluator schema 3 records `scene_variant`, effective parameters and target diagnostics without claiming task success.

- [x] Confirm real contact smoke exits zero and reports resting height, frictional sliding, and subset reset; confirm real Kit controls smoke exits zero.
- [x] Run the real checkpoint evaluator with `--target_object --mode_index 7 --num_envs 128 --max_steps 10`, a new report path, and the unchanged 962-clip validation index. Check finite diagnostics and verified unchanged inputs; label it a short integration smoke, not performance evidence.
- [x] Run `bash scripts/open_motion_browser.sh --target-object`; inspect robot, target, ground and menu in the real unlocked desktop. Exercise mode Apply, clip switching, target reset and Exit; inspect logs and process cleanup.
- [x] Add a portable JSON profile matching the verified defaults. Explain SI units, contact/visual-material distinction, parameter-combine behavior, unknown real-world calibration and unimplemented object-conditioned control in the guide.
- [x] Re-run `"$BFM_ISAAC_PYTHON" -m pytest ScaleTrack/tests tests/test_gui_launcher.py tests/test_ci_workflow.py -q`, shell syntax, and `git diff --check`. Check owned units and GPU processes before the final report.
- [x] Handoff the existing native GUI launch command and validation guide. Do not claim full BFM reproduction or autonomous manipulation.

Execution outcome: 105 scoped local tests passed; the overlapping isolated Python
3.11 CI subset passed 26 tests. Real Kit, contact, robot/menu and evaluator smoke
checks passed and their owned processes exited. GUI automation uses the real
production player and policy with scripted UI callbacks, not desktop input.
Source review found no significant blocker. Hosted CI and public release remain
outside the completed milestone; no commits or uploads were made.

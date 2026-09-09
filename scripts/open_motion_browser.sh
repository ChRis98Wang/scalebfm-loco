#!/usr/bin/env bash
# Launch the existing IsaacLab player, with one bounded, self-cleaning cgroup.
set -euo pipefail

bfm_dry_run=false
bfm_target_object=false
bfm_online_targets=false
for bfm_arg in "$@"; do
  case "$bfm_arg" in
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/open_motion_browser.sh [--dry-run] [--target-object] [--online-targets]' \
        'Open the native IsaacLab GUI and motion menu; no installation or training.' \
        'BFM_ISAAC_PYTHON: absolute path to your existing IsaacLab Python.' \
        'BFM_LOAD_RUN / BFM_CHECKPOINT: run directory name and checkpoint filename.' \
        'BFM_INITIAL_MOTION: initial clip name (optional).' \
        'BFM_VALIDATION_INDEX / BFM_TRAIN_INDEX: local YAML motion indexes.' \
        'BFM_MODE_INDEX: initial reference mode (0=Pelvis-1, 1=UMI-2, 2=VR-3, 7=WholeBody-14).' \
        '--target-object: add a physical box; BFM_TARGET_CONFIG may select a local physics JSON.' \
        '--online-targets: single-environment Kit targets (modes 0/1/2, default VR-3); no box controls.' \
        '--dry-run prints the command without starting any process or checking local assets.' \
        'Exit using Exit player. Emergency stop: systemctl --user stop bfm-motion-browser.service' \
        'The GUI is limited to 30 minutes; the transient service does not auto-restart.'
      exit 0 ;;
    --dry-run) bfm_dry_run=true ;;
    --target-object) bfm_target_object=true ;;
    --online-targets) bfm_online_targets=true ;;
    *) printf 'Unknown argument: %s\n' "$bfm_arg" >&2; exit 2 ;;
  esac
done
if "$bfm_online_targets" && "$bfm_target_object"; then
  printf '%s\n' '--online-targets cannot combine with --target-object.' >&2
  exit 2
fi

bfm_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
bfm_python="${BFM_ISAAC_PYTHON:-${HOME}/isaaclab_ws/env_isaaclab_sim6_newton/bin/python}"
bfm_run="${BFM_LOAD_RUN:-amass_full_v1_finetune_20260906}"
bfm_checkpoint="${BFM_CHECKPOINT:-model_23197.pt}"
bfm_initial="${BFM_INITIAL_MOTION:-ACCAD/Male1Walking_c3d/Walk_B10_-_Walk_turn_left_45_stageii}"
bfm_validation="${BFM_VALIDATION_INDEX:-${bfm_root}/ScaleRetarget/retargeted_dataset/amass_full_v1_validation.yaml}"
bfm_train="${BFM_TRAIN_INDEX:-${bfm_root}/ScaleRetarget/retargeted_dataset/amass_full_v1_train.yaml}"
bfm_mode="${BFM_MODE_INDEX:-7}"
if "$bfm_online_targets"; then
  bfm_mode="${BFM_MODE_INDEX:-2}"
  case "$bfm_mode" in
    0|1|2) ;;
    *) printf '%s\n' 'Online targets require BFM_MODE_INDEX=0, 1, or 2.' >&2; exit 2 ;;
  esac
fi

bfm_command=(
  systemd-run --user --unit=bfm-motion-browser --collect --quiet --wait --pipe
  --property=Type=exec "--property=WorkingDirectory=${bfm_root}/ScaleTrack"
  --property=RuntimeMaxSec=30min --property=TimeoutStopSec=20s
  --property=KillMode=control-group --property=KillSignal=SIGINT
  --property=SendSIGKILL=yes --property=Restart=no
  --property=MemoryMax=16G --property=TasksMax=256
  "--setenv=DISPLAY=${DISPLAY:-}" --setenv=OMNI_KIT_ACCEPT_EULA=YES
  --setenv=PYTHONUNBUFFERED=1 --setenv=OMP_NUM_THREADS=1
  --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1
)
[[ -z "${XAUTHORITY:-}" ]] || bfm_command+=("--setenv=XAUTHORITY=${XAUTHORITY}")
[[ -z "${XDG_RUNTIME_DIR:-}" ]] || bfm_command+=("--setenv=XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}")
bfm_command+=(
  "$bfm_python" -u "${bfm_root}/ScaleTrack/scripts/pretrain/rsl_rl/play.py"
  --task G1-BFM-Transformer-Tracking --load_run "$bfm_run" --checkpoint "$bfm_checkpoint"
  --motion_file "$bfm_validation" --additional_motion_file "$bfm_train"
  --initial_motion "$bfm_initial" --motion_menu --num_envs 1 --mode_index "$bfm_mode"
  --device cuda:0 --viz kit --logger tensorboard
  --kit_args=--/app/window/title=BFM_Motion_Browser
)
if "$bfm_online_targets"; then
  bfm_command+=(--online_targets)
fi
if "$bfm_target_object"; then
  bfm_command+=(--target_object)
  if [[ -n "${BFM_TARGET_CONFIG:-}" ]]; then
    bfm_target_config="$BFM_TARGET_CONFIG"
    # The service runs in ScaleTrack; preserve relative paths from the caller.
    [[ "$bfm_target_config" == /* ]] || bfm_target_config="${PWD}/${bfm_target_config}"
    bfm_command+=(--target_config "$bfm_target_config")
  fi
fi

if "$bfm_dry_run"; then
  printf '%q ' "${bfm_command[@]}"
  printf '\n'
  exit 0
fi
if [[ -z "${DISPLAY:-}" ]]; then
  printf 'DISPLAY is unset. Run this command in your desktop terminal.\n' >&2
  exit 1
fi
if [[ "$bfm_python" != /* || ! -x "$bfm_python" ]]; then
  printf 'Set BFM_ISAAC_PYTHON to the absolute path of your existing IsaacLab Python.\n' >&2
  exit 1
fi
for bfm_file in "$bfm_validation" "$bfm_train" "${bfm_root}/ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/${bfm_run}/${bfm_checkpoint}"; do
  if [[ ! -f "$bfm_file" ]]; then
    printf 'Required local file is missing: %s\n' "$bfm_file" >&2
    exit 1
  fi
done
printf 'Opening IsaacLab GUI with motion switching. Close with Exit player (30 minute limit).\n'
exec "${bfm_command[@]}"

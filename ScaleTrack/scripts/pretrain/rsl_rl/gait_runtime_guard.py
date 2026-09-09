"""Fail-closed guards against resets and direct simulator state writes during gait playback."""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def forbid_state_writes(raw_env, command):
    """Reject reset/resample/state-write calls while leaving normal action targets untouched.

    The guard is instance-local.  Inherited methods are restored by deleting the
    temporary instance override; pre-existing instance overrides are restored
    verbatim.
    """
    robot = getattr(command, "robot", None)
    scene = getattr(raw_env, "scene", None)
    required = (
        (raw_env, "reset", "env.reset"),
        (raw_env, "_reset_idx", "env._reset_idx"),
        (scene, "reset", "scene.reset"),
        (command, "_resample_command", "command._resample_command"),
        (robot, "write_root_pose_to_sim", "robot.write_root_pose_to_sim"),
        (robot, "write_root_velocity_to_sim", "robot.write_root_velocity_to_sim"),
        (robot, "write_root_state_to_sim", "robot.write_root_state_to_sim"),
        (robot, "write_joint_state_to_sim", "robot.write_joint_state_to_sim"),
    )
    optional = (
        (getattr(raw_env, "command_manager", None), "reset", "command_manager.reset"),
        (getattr(raw_env, "event_manager", None), "reset", "event_manager.reset"),
        (robot, "reset", "robot.reset"),
    )

    missing = [label for owner, name, label in required if owner is None or not callable(getattr(owner, name, None))]
    if missing:
        raise AttributeError(f"Cannot guard missing required state-write methods: {missing}")

    targets = list(required)
    targets.extend(
        (owner, name, label)
        for owner, name, label in optional
        if owner is not None and callable(getattr(owner, name, None))
    )
    counters = {label: 0 for _, _, label in targets}
    installed = []

    def make_guard(label):
        def forbidden(*_args, **_kwargs):
            counters[label] += 1
            raise RuntimeError(f"Forbidden gait-runtime state write: {label}")

        return forbidden

    try:
        for owner, name, label in targets:
            namespace = getattr(owner, "__dict__", None)
            had_instance_override = namespace is not None and name in namespace
            previous = namespace[name] if had_instance_override else None
            setattr(owner, name, make_guard(label))
            installed.append((owner, name, had_instance_override, previous))
        yield counters
    finally:
        restore_errors = []
        for owner, name, had_instance_override, previous in reversed(installed):
            try:
                if had_instance_override:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except BaseException as error:  # attempt every restoration before surfacing corruption
                restore_errors.append((name, error))
        if restore_errors:
            names = [name for name, _ in restore_errors]
            raise RuntimeError(f"Failed to restore gait-runtime guards: {names}") from restore_errors[0][1]

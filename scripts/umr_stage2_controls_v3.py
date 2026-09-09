"""Versioned Stage-II-only experimental controls; no runtime imported here.

The caller supplies the SHA-verified SMPL-X UMR subclass, the pinned Mink
module, and the independently validated output-frame RateLimit class. This
module never starts Stage I, a simulator, training or file writes.
"""
from __future__ import annotations

import hashlib
import numpy as np

SCHEMA = "bfm.umr_stage2_controls/3"
ARMS = ("control", "rate_only", "wrist_only", "both")
WRIST_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
WRIST_OFFSETS = np.stack((np.eye(3), np.diag([-1., -1., 1.])))


def _array_sha256(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(str(value.shape).encode("ascii") + b"\0" + value.tobytes()).hexdigest()


def _rotations(value):
    value = np.asarray(value, dtype=np.float64)
    if (value.ndim != 4 or value.shape[1:] != (55, 3, 3) or not len(value)
            or not np.isfinite(value).all()
            or not np.allclose(value.swapaxes(-1, -2) @ value, np.eye(3), rtol=0, atol=2e-5)
            or not np.allclose(np.linalg.det(value), 1., rtol=0, atol=2e-5)):
        raise ValueError("Require finite prepared (frames,55,3,3) proper world rotations")
    return value.copy()


def controlled_retargeter_class(base, mink, RateLimit):
    """Return a narrow hook subclass; control delegates unchanged stock solving.

    RateLimit(model) must implement begin_frame(frame, previous_q_or_None) and
    finish_frame(q_copy). It is absent during bootstrap and holds a fixed
    previous-output anchor across all six IK iterations of each output frame.
    """

    class ControlledRetargeter(base):
        def __init__(self, *args, arm="control", **kwargs):
            if arm not in ARMS:
                raise ValueError("Unknown frozen v3 arm")
            super().__init__(*args, **kwargs)
            if self.iterations != 6 or self.dt != .02:
                raise ValueError("v3 requires six inner iterations and dt=0.02")
            self.arm = arm
            self._human_rotations = _rotations(self.human.source["joint_rotations"])
            self._running = False
            self._used = False
            self._phase = "NEW"
            self._next_frame = 0
            self._previous_output = None
            self._wrist_tasks = []
            self._output_rate_limit = None
            self.v3_audit = {
                "schema": SCHEMA, "arm": arm, "result": "NOT_RUN",
                "base_task_count": len(self.tasks), "base_limit_count": len(self.limits),
                "warmup": None, "frames": [], "wrist_tasks": [], "wrist_targets": [],
                "source_rotation_float64_sha256": _array_sha256(self._human_rotations),
                "warmup_iterations": 60, "iterations_per_output_frame": 6, "output_fps": 50.,
                "iteration_count_definition": "Configured maxima; upstream NoSolutionFound may break an inner loop early and is reported separately",
                "rate_enabled": arm in ("rate_only", "both"),
                "rate_rad_per_second": 12. if arm in ("rate_only", "both") else None,
                "rate_limit_absent_during_warmup": True, "postprocessing_performed": False,
                "warmup_failures": 0, "output_failures": 0, "total_observed_failures": 0,
            }
            if arm in ("wrist_only", "both"):
                for name in WRIST_NAMES:
                    task = mink.FrameTask(frame_name=name, frame_type="body",
                                          position_cost=0., orientation_cost=10., lm_damping=1.)
                    cost = np.asarray(task.cost, dtype=np.float64)
                    if (cost.shape != (6,) or not np.array_equal(cost, [0., 0., 0., 10., 10., 10.])
                            or task.lm_damping != 1. or task.gain != 1.):
                        raise ValueError("Mink actual wrist task cost/gain/damping differs from fixed protocol")
                    self._wrist_tasks.append(task)
                    self.tasks.append(task)
                    self.v3_audit["wrist_tasks"].append({
                        "frame_name": name, "frame_type": "body", "cost": cost.tolist(),
                        "gain": float(task.gain), "lm_damping": float(task.lm_damping),
                        "target_translation": [0., 0., 0.],
                        "interpretation": "world orientation soft cost; 10 is squared in the QP",
                    })
            if arm in ("rate_only", "both"):
                self._output_rate_limit = RateLimit(self.robot.model)

        def _set_wrist_targets(self, frame, *, warmup):
            if not self._wrist_tasks:
                return
            targets = self._human_rotations[frame, [20, 21]] @ WRIST_OFFSETS
            actual_targets = []
            for task, rotation in zip(self._wrist_tasks, targets):
                # These are already world matrices. No pelvis/heading/canonical
                # rotation is applied. Translation cost is zero, never a hard
                # full-frame constraint; an identity translation is sufficient.
                target = mink.SE3.from_rotation(mink.SO3.from_matrix(rotation))
                actual = np.asarray(target.as_matrix(), dtype=np.float64)
                if (actual.shape != (4, 4) or not np.isfinite(actual).all()
                        or not np.allclose(actual[:3, :3], rotation, rtol=0, atol=2e-5)
                        or not np.array_equal(actual[:3, 3], np.zeros(3))
                        or not np.array_equal(actual[3], [0., 0., 0., 1.])):
                    raise ValueError("Mink SE3 target changed wrist orientation or zero translation")
                task.set_target(target)
                actual_targets.append(actual)
            self.v3_audit["wrist_targets"].append({
                "frame": frame, "warmup": warmup,
                "requested_rotation_float64_sha256": _array_sha256(targets),
                "actual_se3_float64_sha256": _array_sha256(np.stack(actual_targets)),
            })

        @staticmethod
        def _observation(out, frame, warmup):
            failures = out["failures"]
            qpos = np.asarray(out["qpos"])
            if (not isinstance(failures, (int, np.integer)) or isinstance(failures, (bool, np.bool_))
                    or failures < 0 or qpos.shape != (36,) or not np.isfinite(qpos).all()
                    or not np.isclose(np.linalg.norm(qpos[3:7]), 1., rtol=0, atol=1e-5)):
                raise ValueError("Invalid upstream frame output / failure count")
            return {"frame": frame, "warmup": warmup, "failures": int(failures),
                    "qpos_float64_sha256": _array_sha256(qpos)}, qpos.copy()

        def solve_frame(self, frame, iterations=None):
            if (not self._running or not isinstance(frame, (int, np.integer))
                    or isinstance(frame, (bool, np.bool_))):
                raise ValueError("solve_frame may only be called by the bounded sequential run")
            frame = int(frame)
            warmup = self._phase == "AWAIT_WARMUP"
            if warmup:
                if frame != 0 or type(iterations) is not int or iterations != 60:
                    raise ValueError("Expected exactly one frame-zero 60-iteration bootstrap")
                if self._output_rate_limit is not None and any(x is self._output_rate_limit for x in self.limits):
                    raise ValueError("Rate constraint must be absent during bootstrap")
            elif (self._phase != "OUTPUT" or iterations is not None
                  or frame != self._next_frame or frame >= len(self._human_rotations)):
                raise ValueError("Repeated, skipped or malformed output-frame call")
            self._set_wrist_targets(frame, warmup=warmup)
            if not warmup and self._output_rate_limit is not None:
                anchor = None if self._previous_output is None else self._previous_output.copy()
                self._output_rate_limit.begin_frame(frame, anchor)
            # Do not catch NoSolutionFound here: pinned base already counts its
            # last-valid-iteration fallback. Expose that count including warmup.
            out = super().solve_frame(frame, iterations=iterations)
            record, qpos = self._observation(out, frame, warmup)
            if warmup:
                self.v3_audit["warmup"] = record
                self.v3_audit["warmup_failures"] = record["failures"]
                self.v3_audit["total_observed_failures"] += record["failures"]
                self._phase = "OUTPUT"
                if self._output_rate_limit is not None:
                    self.limits.append(self._output_rate_limit)
            else:
                self.v3_audit["frames"].append(record)
                self.v3_audit["output_failures"] += record["failures"]
                self.v3_audit["total_observed_failures"] += record["failures"]
                if self._output_rate_limit is not None:
                    # The independent limit verifies the output delta, with its
                    # fixed tolerance; never clip or silently repair a failure.
                    try:
                        self._output_rate_limit.finish_frame(qpos.copy())
                    except Exception as error:
                        record["rate_validation_error"] = f"{type(error).__name__}: {error}"
                        raise
                self._previous_output = qpos.copy()
                self._next_frame += 1
            return out

        def run(self, frame_indices, fps, warmup_iterations=60, progress=True):
            indices = np.asarray(frame_indices)
            if (self._used or self._running or indices.ndim != 1 or indices.dtype.kind not in "iu"
                    or not np.array_equal(indices, np.arange(len(self._human_rotations)))
                    or not isinstance(fps, (float, int, np.floating, np.integer))
                    or isinstance(fps, (bool, np.bool_)) or fps != 50.
                    or type(warmup_iterations) is not int or warmup_iterations != 60):
                raise ValueError("Require one run of all contiguous source frames at 50 Hz, with bootstrap60")
            self._used = True
            self._running = True
            self._phase = "AWAIT_WARMUP"
            self.v3_audit["result"] = "RUNNING"
            try:
                result = super().run(frame_indices, fps, warmup_iterations=60, progress=progress)
                if (self._phase != "OUTPUT" or self._next_frame != len(indices)
                        or self.v3_audit["warmup"] is None
                        or len(self.v3_audit["frames"]) != len(indices)
                        or not np.array_equal(result.frame_indices, indices)
                        or np.asarray(result.qpos).shape != (len(indices), 36) or result.fps != 50.):
                    raise ValueError("Upstream run did not produce exactly the audited frame sequence")
                for qpos, record in zip(result.qpos, self.v3_audit["frames"]):
                    if _array_sha256(qpos) != record["qpos_float64_sha256"]:
                        raise ValueError("Upstream run changed qpos after solve_frame (postprocessing forbidden)")
                self.v3_audit["result"] = "COMPLETE_NOT_QUALITY_ACCEPTED"
                self._phase = "COMPLETE"
                return result
            except Exception as error:
                self.v3_audit["result"] = "ERROR"
                self.v3_audit["error"] = f"{type(error).__name__}: {error}"
                self._phase = "ERROR"
                raise
            finally:
                self._running = False

        def lock_ankle_roll(self, *args, **kwargs):
            raise ValueError("v3 forbids ankle locking and all post-solve qpos modifications")

    return ControlledRetargeter

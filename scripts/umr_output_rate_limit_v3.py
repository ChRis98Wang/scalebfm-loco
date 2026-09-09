"""Pure-NumPy, previous-OUTPUT-frame G1 rate box for injected Mink Limits.

Mink 1.1.1 build_ik optimizes delta_q, not velocity; solve_ik returns delta_q/dt.
This is intentionally different from resetting a VelocityLimit at every inner
IK iteration. No import of Mink, MuJoCo, Isaac, or any physics/training runtime.
There is no clipping, rescaling, angle wrapping, or automatic output mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np

OUTPUT_DT = 0.02
MAX_RATE_RAD_S = 12.0
MAX_OUTPUT_STEP_RAD = MAX_RATE_RAD_S * OUTPUT_DT
STEP_TOLERANCE_RAD = 2e-6
RATE_WITH_TOLERANCE_RAD_S = MAX_RATE_RAD_S + STEP_TOLERANCE_RAD / OUTPUT_DT
SCHEMA = "bfm.umr_output_frame_rate_limit_v3/1"
JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"Expected integer {label}")
    return int(value)


def _positive_dt(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("Mink integration dt must be a finite positive number")


@dataclass(frozen=True)
class HingeLayout:
    nq: int
    nv: int
    joint_names: tuple[str, ...]
    joint_ids: tuple[int, ...]
    qpos_addresses: tuple[int, ...]
    dof_addresses: tuple[int, ...]

    def __post_init__(self):
        _integer(self.nq, "qpos dimension")
        _integer(self.nv, "tangent dimension")
        for values in (self.joint_names, self.joint_ids, self.qpos_addresses, self.dof_addresses):
            if not isinstance(values, tuple):
                raise ValueError("Hinge layout fields must be immutable tuples")
        if (self.nq != 36 or self.nv != 35 or self.joint_names != JOINT_NAMES
                or len(self.joint_ids) != 29 or set(self.joint_ids) != set(range(1, 30))
                or len(self.qpos_addresses) != 29 or set(self.qpos_addresses) != set(range(7, 36))
                or len(self.dof_addresses) != 29 or set(self.dof_addresses) != set(range(6, 35))
                or any(q != d + 1 for q, d in zip(self.qpos_addresses, self.dof_addresses))):
            raise ValueError("Expected exact named 29-hinge G1 free-base layout")
        for values in (self.joint_ids, self.qpos_addresses, self.dof_addresses):
            for value in values:
                _integer(value, "joint ID/address")


def validate_hinge_layout(model):
    """Read model metadata only. MuJoCo enums: FREE=0, HINGE=3.

    Named lookup, reverse lookup, joint types and complete qpos/dof address
    coverage must agree. No reliance on positional order of the named hinges.
    """
    if tuple(_integer(getattr(model, k), k) for k in ("nq", "nv", "njnt")) != (36, 35, 30):
        raise ValueError("Model must have one free joint plus exactly 29 hinges")
    fields = {}
    for name in ("jnt_type", "jnt_qposadr", "jnt_dofadr"):
        value = np.asarray(getattr(model, name))
        if value.shape != (30,) or value.dtype.kind not in "iu":
            raise ValueError(f"Invalid model joint metadata: {name}")
        fields[name] = value
    if (fields["jnt_type"][0] != 0 or not np.all(fields["jnt_type"][1:] == 3)
            or fields["jnt_qposadr"][0] != 0 or fields["jnt_dofadr"][0] != 0):
        raise ValueError("Expected one leading free joint and 29 scalar hinge joints")
    ids = []
    for name in JOINT_NAMES:
        try:
            joint = model.joint(name)
            jid = _integer(joint.id, "named joint ID")
            if not 1 <= jid < 30 or joint.name != name or model.joint(jid).name != name:
                raise ValueError("Named lookup/reverse lookup disagree")
        except (KeyError, AttributeError, IndexError) as exc:
            raise ValueError(f"Missing/inconsistent G1 hinge name: {name}") from exc
        ids.append(jid)
    return HingeLayout(36, 35, JOINT_NAMES, tuple(ids),
                       tuple(int(fields["jnt_qposadr"][i]) for i in ids),
                       tuple(int(fields["jnt_dofadr"][i]) for i in ids))


def checked_qpos(value, layout):
    raw = np.asarray(value)
    if raw.shape != (layout.nq,) or raw.dtype.kind not in "fiu":
        raise ValueError("Expected one numeric full G1 qpos vector")
    q = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(q).all():
        raise ValueError("Nonfinite output/current/anchor qpos")
    if abs(float(np.linalg.norm(q[3:7])) - 1.) > 1e-5:
        raise ValueError("G1 free-base quaternion must be unit wxyz")
    return q


def output_box_inequalities(current_qpos, previous_output_qpos, layout):
    """Return float64 G,h for G delta_q <= h about one fixed output anchor.

    E selects the 29 hinge tangent DOFs; the base remains unconstrained here.
    G=[E;-E], h=[anchor+0.24-current; current-anchor+0.24]. The 0.24 rad
    budget is fixed to 12 rad/s * 0.02 s OUTPUT time, never inner solver dt.
    Negative h is valid when current iterate is outside the output-rate box.
    """
    current = checked_qpos(current_qpos, layout)
    anchor = checked_qpos(previous_output_qpos, layout)
    qids = np.asarray(layout.qpos_addresses)
    E = np.eye(layout.nv, dtype=np.float64)[np.asarray(layout.dof_addresses)]
    offset = current[qids] - anchor[qids]
    G = np.vstack((E, -E))
    h = np.concatenate((MAX_OUTPUT_STEP_RAD - offset, MAX_OUTPUT_STEP_RAD + offset))
    if not np.isfinite(h).all():
        raise ValueError("Nonfinite output-rate box bounds")
    return G, h


def _interval_report(current_qpos, previous_output_qpos, layout, frame_index):
    current, previous = checked_qpos(current_qpos, layout), checked_qpos(previous_output_qpos, layout)
    steps = np.abs(current[np.asarray(layout.qpos_addresses)] - previous[np.asarray(layout.qpos_addresses)])
    rates = steps / OUTPUT_DT
    if not np.isfinite(rates).all():
        raise ValueError("Nonfinite output-frame rates")
    exact = rates > MAX_RATE_RAD_S
    beyond_tolerance = rates > RATE_WITH_TOLERANCE_RAD_S
    # Acceptance is defined in radians, not a second floating-point conversion.
    accepted = not bool(np.any(steps > MAX_OUTPUT_STEP_RAD + STEP_TOLERANCE_RAD))
    return {"frame_index": frame_index, "anchor_frame_index": frame_index - 1,
            "bootstrap": False, "interval_count": 1, "rate_limit_applicable": True,
            "max_step_rad": float(steps.max()), "max_rate_rad_s": float(rates.max()),
            "exact_exceedance_joint_count": int(exact.sum()),
            "tolerance_exceedance_joint_count": int(beyond_tolerance.sum()),
            "accepted_with_fixed_tolerance": accepted,
            "per_joint_abs_step_rad": dict(zip(JOINT_NAMES, map(float, steps))),
            "per_joint_rate_rad_s": dict(zip(JOINT_NAMES, map(float, rates)))}


def interval_statistics(qposes, layout, *, frame_indices=None):
    """Audit ALL T-1 output intervals, including output frame 0 -> 1.

    No decimation, angle wrapping, bootstrap-interval skipping, or repair. With
    one frame, counts are zero and the rate metric is explicitly not applicable.
    """
    values = np.asarray(qposes)
    if values.ndim != 2 or values.shape[1] != layout.nq or values.shape[0] < 1:
        raise ValueError("Need at least one full output qpos frame")
    q = np.stack([checked_qpos(row, layout) for row in values])
    frames = len(q)
    if frame_indices is not None:
        indices = np.asarray(frame_indices)
        if indices.dtype.kind not in "iu" or not np.array_equal(indices, np.arange(frames)):
            raise ValueError("Output frame indices must be consecutive 0..T-1")
    intervals = [_interval_report(q[i], q[i-1], layout, i) for i in range(1, frames)]
    per_joint = {}
    for name in JOINT_NAMES:
        steps = [r["per_joint_abs_step_rad"][name] for r in intervals]
        rates = [r["per_joint_rate_rad_s"][name] for r in intervals]
        per_joint[name] = {"max_step_rad": max(steps, default=None), "max_rate_rad_s": max(rates, default=None),
                           "exact_exceedance_count": sum(v > MAX_RATE_RAD_S for v in rates),
                           "tolerance_exceedance_count": sum(v > RATE_WITH_TOLERANCE_RAD_S for v in rates)}
    return {"schema": SCHEMA, "output_dt": OUTPUT_DT, "max_rate_rad_s": MAX_RATE_RAD_S,
            "max_output_step_rad": MAX_OUTPUT_STEP_RAD, "step_tolerance_rad": STEP_TOLERANCE_RAD,
            "rate_threshold_with_tolerance_rad_s": RATE_WITH_TOLERANCE_RAD_S,
            "frame_count": frames, "interval_count": frames - 1,
            "bootstrap_frame_index": 0, "bootstrap_has_previous_output": False,
            "first_interval_included": frames > 1, "rate_limit_applicable": frames > 1,
            "max_observed_step_rad": max((r["max_step_rad"] for r in intervals), default=None),
            "max_observed_rate_rad_s": max((r["max_rate_rad_s"] for r in intervals), default=None),
            "exact_exceedance_joint_intervals": sum(r["exact_exceedance_joint_count"] for r in intervals),
            "tolerance_exceedance_joint_intervals": sum(r["tolerance_exceedance_joint_count"] for r in intervals),
            "exact_exceedance_frame_indices": [r["frame_index"] for r in intervals if r["exact_exceedance_joint_count"]],
            "tolerance_exceedance_frame_indices": [r["frame_index"] for r in intervals if r["tolerance_exceedance_joint_count"]],
            "accepted_with_fixed_tolerance": all(r["accepted_with_fixed_tolerance"] for r in intervals),
            "per_joint": per_joint, "intervals": intervals,
            "interpretation": "Output hinge rates only; not a physical-tracking or quality gate; no mutation"}


class OutputRateViolation(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"Output frame {report['frame_index']} exceeds the fixed 12 rad/s + 2e-6 rad interval tolerance")


def output_rate_limit_class(limit_base, constraint_type):
    """Inject mink.Limit and mink.limits.Constraint without runtime imports.

    begin_frame(i, previous_output_qpos) must happen exactly once per OUTPUT
    frame. The first call is explicitly begin_frame(0, None). finish_frame(q)
    checks the output, saves an independent copy and closes the frame. Warmup
    should not attach this limit. Disabled control arms can audit their complete
    outputs with interval_statistics without attaching/finishing this limit.
    """
    class OutputFrameRateLimit(limit_base):
        def __init__(self, model):
            self.model = model
            self.layout = validate_hinge_layout(model)
            self._next_frame = 0
            self._active_frame = None
            self._anchor = None
            self._previous_output = None
            self._history = []

        def _check_model(self):
            if validate_hinge_layout(self.model) != self.layout:
                raise ValueError("Named hinge model layout changed after rate-limit construction")

        @property
        def anchor_qpos(self):
            return None if self._anchor is None else self._anchor.copy()

        @property
        def completed_frames(self):
            return self._next_frame

        def begin_frame(self, frame_index, previous_output_qpos):
            self._check_model()
            frame_index = _integer(frame_index, "output frame index")
            if self._active_frame is not None:
                raise ValueError("Output frame already prepared; inner iterations cannot reset the anchor")
            if frame_index != self._next_frame:
                raise ValueError("Output frames must start at zero and advance exactly once")
            if frame_index == 0:
                if previous_output_qpos is not None:
                    raise ValueError("Bootstrap output frame zero has no previous-output anchor")
                anchor = None
            else:
                if previous_output_qpos is None:
                    raise ValueError("Missing explicit previous-output anchor")
                anchor = checked_qpos(previous_output_qpos, self.layout)
                if not np.array_equal(anchor, self._previous_output):
                    raise ValueError("Anchor does not equal the last completed output frame")
                anchor.setflags(write=False)
            self._anchor, self._active_frame = anchor, frame_index

        def compute_qp_inequalities(self, configuration, dt):
            _positive_dt(dt)
            self._check_model()
            if configuration.model is not self.model:
                raise ValueError("Configuration is not bound to this named G1 model")
            if self._active_frame is None:
                raise ValueError("Output rate limit was not prepared for this output frame")
            current = checked_qpos(configuration.q, self.layout)
            if self._active_frame == 0:
                return constraint_type()  # Explicit bootstrap, not missing-state fallback.
            G, h = output_box_inequalities(current, self._anchor, self.layout)
            return constraint_type(G=G, h=h)

        def finish_frame(self, output_qpos):
            self._check_model()
            if self._active_frame is None:
                raise ValueError("Cannot finish an unprepared/already completed output frame")
            current = checked_qpos(output_qpos, self.layout)
            if self._active_frame == 0:
                report = {"frame_index": 0, "anchor_frame_index": None, "bootstrap": True,
                          "interval_count": 0, "rate_limit_applicable": False,
                          "max_step_rad": None, "max_rate_rad_s": None,
                          "exact_exceedance_joint_count": 0, "tolerance_exceedance_joint_count": 0,
                          "accepted_with_fixed_tolerance": True}
            else:
                report = _interval_report(current, self._anchor, self.layout, self._active_frame)
                if not report["accepted_with_fixed_tolerance"]:
                    raise OutputRateViolation(report)  # Keep state/anchor; never clip or advance.
            self._previous_output = current.copy()
            self._history.append(current.copy())
            self._next_frame += 1
            self._active_frame = self._anchor = None
            return report

        def statistics(self):
            if self._active_frame is not None:
                raise ValueError("Cannot report a completed sequence with an unfinished output frame")
            if not self._history:
                raise ValueError("No completed output frames")
            return interval_statistics(np.stack(self._history), self.layout)

    OutputFrameRateLimit.__name__ = "OutputFrameRateLimit"
    return OutputFrameRateLimit

"""Opt-in root-XY-relative tracking which retains world-height feedback.

The upstream reference_forcing mode replaces all of root XYZ. Here only XY is
replaced: target Z remains relative to measured robot Z. No robot state or
reference trajectory is teleported. The legacy environment is unchanged.
"""
from scalebridge.env.motion_tracking import MotionTrackingEnv


class HeightAwareMotionTrackingEnv(MotionTrackingEnv):
    def _gather_reference_state(self):
        actual_z = self.state_buffer["root_pos_buffer"][:, -1, 2].clone()
        super()._gather_reference_state()
        if self.reference_forcing:
            self.state_buffer["root_pos_buffer"][:, -1, 2] = actual_z

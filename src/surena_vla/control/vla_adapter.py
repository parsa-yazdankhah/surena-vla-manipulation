"""OpenVLA/MiniVLA delta-action bridge from the validated controller."""

from __future__ import annotations

import queue

import mink
import numpy as np

from .ik import SurenaIK

class OpenVLABridge:
    """
    Integrates OpenVLA 7-D delta-action vectors into IK targets.

    Action convention:
        [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]

    The seventh value is forwarded exactly as received. For the local
    Bridge/RLDS MiniVLA checkpoint it is continuous in [0, 1], with 0=close and
    1=open; sticky-hand normalization is handled separately.
    """

    DEFAULT_POS_SCALE = 0.01
    DEFAULT_ROT_SCALE = 0.05
    MAX_POS_DELTA     = 0.05
    MAX_ROT_DELTA     = 0.20

    def __init__(self, ik: SurenaIK,
                 pos_scale: float = DEFAULT_POS_SCALE,
                 rot_scale: float = DEFAULT_ROT_SCALE):
        self.ik        = ik
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale

        pos, _ = self.ik.get_eef_pose()
        self.target_pos = pos.copy()
        self.target_so3 = self._current_eef_so3()

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._last_action = np.zeros(7)

    def _current_eef_so3(self):
        return self.ik.current_eef_so3()

    def send_action(self, action: np.ndarray):
        arr = np.asarray(action, dtype=float).copy()
        if arr.shape[0] < 7:
            raise ValueError(f"OpenVLA action must have at least 7 elements, got {arr.shape}")
        try:
            self._queue.put_nowait(arr)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(arr)

    def control_callback(self, verbose: bool = False, **ik_kwargs) -> dict:
        while not self._queue.empty():
            try:
                self._last_action = self._queue.get_nowait()
            except queue.Empty:
                break

        a = self._last_action
        d_pos = np.clip(a[:3] * self.pos_scale, -self.MAX_POS_DELTA, self.MAX_POS_DELTA)
        d_rpy = np.clip(a[3:6] * self.rot_scale, -self.MAX_ROT_DELTA, self.MAX_ROT_DELTA)

        self.target_pos += d_pos
        d_so3 = mink.SO3.from_rpy_radians(*d_rpy)
        self.target_so3 = d_so3 @ self.target_so3

        info = self.ik.move_eef_to(self.target_pos, self.target_so3, verbose=verbose, **ik_kwargs)
        info["gripper"] = float(a[6])
        return info

    def reset_to_current(self):
        """Re-seed the integration state from the actual current EEF pose."""
        pos, _ = self.ik.get_eef_pose()
        self.target_pos = pos.copy()
        self.target_so3 = self._current_eef_so3()
        print(f"[VLA] Reset to current EEF pose: {self.target_pos.round(4)}")

# Model-neutral compatibility name for new package users.
DeltaActionAdapter = OpenVLABridge

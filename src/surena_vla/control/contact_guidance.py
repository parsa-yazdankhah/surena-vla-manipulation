"""Reusable contact-aware action guidance for articulated fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

try:
    import mujoco
except ImportError:  # Keep the geometry helpers testable without simulation extras.
    mujoco = None


def hinge_tangent_delta(
    point: np.ndarray,
    hinge: np.ndarray,
    axis: np.ndarray,
    joint_direction: float,
    step: float,
) -> np.ndarray:
    """Return a Cartesian step tangent to a revolute joint's motion."""
    axis = np.asarray(axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    radial = np.asarray(point, dtype=float) - np.asarray(hinge, dtype=float)
    radial -= axis * float(np.dot(radial, axis))
    tangent = np.cross(axis, radial) * float(np.sign(joint_direction))
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        return np.zeros(3)
    return tangent * (float(step) / norm)


def hinge_arc_delta(
    point: np.ndarray,
    hinge: np.ndarray,
    axis: np.ndarray,
    joint_direction: float,
    arc_step: float,
) -> np.ndarray:
    """Return a finite hinge-arc step while preserving hinge radius exactly."""
    axis = np.asarray(axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    radial = np.asarray(point, dtype=float) - np.asarray(hinge, dtype=float)
    axial = axis * float(np.dot(radial, axis))
    planar = radial - axial
    radius = float(np.linalg.norm(planar))
    if radius <= 1e-12:
        return np.zeros(3)
    angle = float(np.sign(joint_direction)) * min(float(arc_step) / radius, 0.12)
    rotated = (planar * np.cos(angle) + np.cross(axis, planar) * np.sin(angle) +
               axis * float(np.dot(axis, planar)) * (1.0 - np.cos(angle)))
    return rotated - planar


def clamp_target_lag(target: np.ndarray, actual: np.ndarray, max_lag: float) -> np.ndarray:
    """Clamp an integrated Cartesian target to a ball around actual EEF pose."""
    delta = np.asarray(target, dtype=float) - np.asarray(actual, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= float(max_lag) or distance <= 1e-12:
        return np.asarray(target, dtype=float).copy()
    return np.asarray(actual, dtype=float) + delta * (float(max_lag) / distance)


@dataclass
class ContactGuidanceConfig:
    joint_names: tuple[str, ...]
    fixture_body_names: tuple[str, ...]
    target_q: float
    tangent_step: float = 0.012
    max_target_lag: float = 0.015
    contact_preload: float = 0.002
    progress_epsilon: float = 2e-4
    stall_steps: int = 5
    recovery_retract_step: float = 0.010
    recovery_retract_steps: int = 2
    recovery_reacquire_step: float = 0.005
    recovery_reacquire_steps: int = 2

    @classmethod
    def from_mapping(cls, value: Mapping) -> "ContactGuidanceConfig":
        return cls(
            joint_names=tuple(value["joint_names"]),
            fixture_body_names=tuple(value["fixture_body_names"]),
            target_q=float(value["target_q"]),
            tangent_step=float(value.get("tangent_step", 0.012)),
            max_target_lag=float(value.get("max_target_lag", 0.015)),
            contact_preload=float(value.get("contact_preload", 0.002)),
            progress_epsilon=float(value.get("progress_epsilon", 2e-4)),
            stall_steps=int(value.get("stall_steps", 5)),
            recovery_retract_step=float(value.get("recovery_retract_step", 0.010)),
            recovery_retract_steps=int(value.get("recovery_retract_steps", 2)),
            recovery_reacquire_step=float(value.get("recovery_reacquire_step", 0.005)),
            recovery_reacquire_steps=int(value.get("recovery_reacquire_steps", 2)),
        )


class HingeContactGuidance:
    """State machine that turns a free-space VLA approach into hinge following.

    The VLA remains responsible for finding the fixture. Once physical contact
    or joint progress is observed, Cartesian motion follows the revolute joint
    tangent. Target anti-windup is active throughout the episode.
    """

    def __init__(self, model, data, eef_site_name: str, config: Mapping):
        if mujoco is None:
            raise RuntimeError("HingeContactGuidance requires the mujoco package")
        self.model = model
        self.data = data
        self.config = ContactGuidanceConfig.from_mapping(config)
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, eef_site_name)
        if self.site_id < 0:
            raise RuntimeError(f"EEF site not found: {eef_site_name!r}")
        self.joint_id = self._resolve_joint(self.config.joint_names)
        self.fixture_body_id = self._resolve_body(self.config.fixture_body_names)
        self.qadr = int(model.jnt_qposadr[self.joint_id])
        self.joint_body_id = int(model.jnt_bodyid[self.joint_id])
        self.initial_q = self.joint_q
        self.previous_q = self.initial_q
        self.active = False
        self.stall_count = 0
        self.recovery_remaining = 0
        self.reacquire_remaining = 0
        self.recovery_direction: np.ndarray | None = None
        self.last_mode = "approach"
        self.approach_recovery_remaining = 0

    def _resolve_joint(self, names: Sequence[str]) -> int:
        for name in names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                return int(jid)
        raise RuntimeError(f"Articulation joint not found; tried {list(names)!r}")

    def _resolve_body(self, names: Sequence[str]) -> int:
        for name in names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return int(bid)
        raise RuntimeError(f"Fixture body not found; tried {list(names)!r}")

    @property
    def joint_q(self) -> float:
        return float(self.data.qpos[self.qadr])

    def _hinge_world(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = np.asarray(self.data.xmat[self.joint_body_id], dtype=float).reshape(3, 3)
        hinge = (np.asarray(self.data.xpos[self.joint_body_id], dtype=float) +
                 rotation @ np.asarray(self.model.jnt_pos[self.joint_id], dtype=float))
        axis = rotation @ np.asarray(self.model.jnt_axis[self.joint_id], dtype=float)
        return hinge, axis

    def _eef_pos(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self.site_id], dtype=float).copy()

    def _contact_state(self) -> tuple[str | None, np.ndarray | None]:
        """Classify live hand contact as door or fixed-fixture contact."""
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            n1 = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b1) or "").lower()
            n2 = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2) or "").lower()
            gname1 = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or "").lower()
            gname2 = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or "").lower()
            hand1 = any(token in n1 or token in gname1 for token in ("r_hand", "hand_roll", "palm", "wrist_contact"))
            hand2 = any(token in n2 or token in gname2 for token in ("r_hand", "hand_roll", "palm", "wrist_contact"))
            door1 = self._is_joint_descendant(b1)
            door2 = self._is_joint_descendant(b2)
            fixture1 = self._is_fixture_descendant(b1)
            fixture2 = self._is_fixture_descendant(b2)
            if not ((hand1 and fixture2) or (hand2 and fixture1)):
                continue
            normal = np.asarray(contact.frame[:3], dtype=float)
            # MuJoCo's contact normal points from geom 1 toward geom 2.
            retract = -normal if hand1 else normal
            norm = float(np.linalg.norm(retract))
            if norm > 1e-12:
                kind = "door" if ((hand1 and door2) or (hand2 and door1)) else "fixture"
                return kind, retract / norm
        return None, None

    def _hand_retract_direction(self) -> np.ndarray | None:
        kind, retract = self._contact_state()
        return retract if kind == "door" else None

    def _is_joint_descendant(self, body_id: int) -> bool:
        bid = int(body_id)
        while bid > 0:
            if bid == self.joint_body_id:
                return True
            bid = int(self.model.body_parentid[bid])
        return False

    def _is_fixture_descendant(self, body_id: int) -> bool:
        bid = int(body_id)
        while bid > 0:
            if bid == self.fixture_body_id:
                return True
            bid = int(self.model.body_parentid[bid])
        return False

    def prepare_action(self, ctrl, exec_action: np.ndarray,
                       previous_ik_info: Mapping | None) -> tuple[np.ndarray, str]:
        """Apply anti-windup and, after contact, hinge/recovery guidance."""
        actual = self._eef_pos()
        ctrl.vla.target_pos = clamp_target_lag(
            ctrl.vla.target_pos, actual, self.config.max_target_lag)

        joint_progress = abs(self.joint_q - self.initial_q)
        contact_kind, contact_retract = self._contact_state()
        if contact_kind == "door" or joint_progress >= self.config.progress_epsilon:
            self.active = True
        if not self.active:
            if contact_kind == "fixture" and contact_retract is not None:
                self.approach_recovery_remaining = self.config.recovery_retract_steps
                self.recovery_direction = contact_retract.copy()
            if self.approach_recovery_remaining > 0 and self.recovery_direction is not None:
                ctrl.vla.target_pos = actual.copy()
                action = np.asarray(exec_action, dtype=float).copy()
                delta = self.recovery_direction * self.config.recovery_retract_step
                action[:3] = delta / float(ctrl.vla.pos_scale)
                self.approach_recovery_remaining -= 1
                self.last_mode = "approach_retract"
                return action, self.last_mode
            self.last_mode = "approach"
            return np.asarray(exec_action, dtype=float).copy(), self.last_mode

        # Contact mode is referenced to the live pose every step; unreachable
        # deltas therefore cannot accumulate behind the fixture.
        ctrl.vla.target_pos = actual.copy()
        action = np.asarray(exec_action, dtype=float).copy()
        hinge, axis = self._hinge_world()
        direction = self.config.target_q - self.joint_q
        delta = hinge_arc_delta(
            actual, hinge, axis, direction, self.config.tangent_step)
        # Maintain gentle pressure into the moving door. This is deliberately
        # much smaller than the arc step and uses the measured contact normal.
        if contact_kind == "door" and contact_retract is not None:
            self.recovery_direction = contact_retract.copy()
        if self.recovery_direction is not None:
            delta = delta - self.recovery_direction * self.config.contact_preload

        if self.recovery_remaining > 0:
            retract = self._hand_retract_direction()
            if retract is None:
                # Radially outward is a safe deterministic fallback when the
                # contact vanished before recovery was sampled.
                retract = actual - hinge
                retract -= axis * float(np.dot(retract, axis))
                retract /= max(float(np.linalg.norm(retract)), 1e-12)
            self.recovery_direction = retract.copy()
            delta = retract * self.config.recovery_retract_step
            self.recovery_remaining -= 1
            self.last_mode = "recover_retract"
        elif self.reacquire_remaining > 0:
            # Move back toward the door while already advancing along its arc.
            # The tangential component avoids returning to exactly the same
            # jammed contact point.
            retract = self.recovery_direction
            if retract is not None:
                delta = delta - retract * self.config.recovery_reacquire_step
            self.reacquire_remaining -= 1
            self.last_mode = "recover_reacquire"
        else:
            self.last_mode = "hinge_follow"

        action[:3] = delta / float(ctrl.vla.pos_scale)
        return action, self.last_mode

    def observe_execution(self) -> None:
        """Update progress and arm a short recovery after sustained stalling."""
        q = self.joint_q
        desired = float(np.sign(self.config.target_q - q))
        progress = desired * (q - self.previous_q)
        self.previous_q = q
        if not self.active or self.recovery_remaining > 0 or self.reacquire_remaining > 0:
            return
        if progress >= self.config.progress_epsilon:
            self.stall_count = 0
        else:
            self.stall_count += 1
        if self.stall_count >= self.config.stall_steps:
            self.recovery_remaining = self.config.recovery_retract_steps
            self.reacquire_remaining = self.config.recovery_reacquire_steps
            self.recovery_direction = self._hand_retract_direction()
            self.stall_count = 0

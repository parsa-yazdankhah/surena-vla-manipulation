"""Robust command-level sticky hand for SURENA's fixed palm.

Command qualification is deliberately independent of MuJoCo.  The VLA value
remains continuous; only :class:`HandIntent` and physical state transitions are
discrete.  ``update_command`` is called at controller rate, while
``enforce_attachment`` may be called after every physics step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import logging
import math
from typing import Sequence

import numpy as np

from surena_vla.gripper_command import (
    CommandSample, GripperCommandConfig, GripperCommandProcessor, HandIntent,
)

LOG = logging.getLogger(__name__)


class StickyGripperState(str, Enum):
    OPEN = "OPEN"
    SEEKING = "SEEKING"
    ATTACHED = "ATTACHED"
    RELEASE_PENDING = "RELEASE_PENDING"


@dataclass(frozen=True)
class StickyGripperConfig:
    object_name_filter: str | None = None
    object_name_exclude: tuple[str, ...] = (
        "robot", "surena", "table", "cabinet", "fixture", "world", "arena",
    )
    attach_distance: float = 0.09
    candidate_dwell_ticks: int = 2
    close_threshold: float = 0.65
    release_threshold: float = 0.35
    close_dwell_ticks: int = 2
    release_dwell_ticks: int = 2
    filter_alpha: float | None = None
    normalization_scale: float = -1.0
    normalization_offset: float = 1.0
    verbose: bool = False

    def __post_init__(self):
        if not math.isfinite(self.attach_distance) or self.attach_distance <= 0:
            raise ValueError("attach_distance must be finite and positive")
        if self.candidate_dwell_ticks < 1:
            raise ValueError("candidate_dwell_ticks must be at least 1")
        # Reuse all command invariant validation.
        self.command_config()

    def command_config(self) -> GripperCommandConfig:
        return GripperCommandConfig(
            close_threshold=self.close_threshold,
            release_threshold=self.release_threshold,
            close_dwell_ticks=self.close_dwell_ticks,
            release_dwell_ticks=self.release_dwell_ticks,
            filter_alpha=self.filter_alpha,
            normalization_scale=self.normalization_scale,
            normalization_offset=self.normalization_offset,
        )


class StickyGripper:
    """Kinematically attach at most one freejoint body to SURENA's palm."""

    def __init__(self, env, ctrl, *, config: StickyGripperConfig | None = None,
                 object_name_filter: str | None = None,
                 object_name_exclude: Sequence[str] | None = None,
                 attach_distance: float = 0.09, close_threshold: float = 0.65,
                 release_threshold: float = 0.35, close_dwell_ticks: int = 2,
                 release_dwell_ticks: int = 2, candidate_dwell_ticks: int = 2,
                 filter_alpha: float | None = None,
                 normalization_scale: float = -1.0,
                 normalization_offset: float = 1.0, verbose: bool = False):
        if config is None:
            kwargs = dict(object_name_filter=object_name_filter,
                          attach_distance=attach_distance,
                          close_threshold=close_threshold,
                          release_threshold=release_threshold,
                          close_dwell_ticks=close_dwell_ticks,
                          release_dwell_ticks=release_dwell_ticks,
                          candidate_dwell_ticks=candidate_dwell_ticks,
                          filter_alpha=filter_alpha,
                          normalization_scale=normalization_scale,
                          normalization_offset=normalization_offset,
                          verbose=verbose)
            if object_name_exclude is not None:
                kwargs["object_name_exclude"] = tuple(object_name_exclude)
            config = StickyGripperConfig(**kwargs)
        self.settings = config
        self.processor = GripperCommandProcessor(config.command_config())
        self.env, self.ctrl = env, ctrl
        self.model, self.data = ctrl.model, ctrl.data
        self._candidate_metadata: list[dict] = []
        self.reset()
        self._cache_candidates()

    # Legacy-readable configuration attributes.
    def __getattr__(self, name):
        if name in StickyGripperConfig.__dataclass_fields__:
            return getattr(self.settings, name)
        raise AttributeError(name)

    @property
    def attached(self) -> bool:
        return self.state in (StickyGripperState.ATTACHED,
                              StickyGripperState.RELEASE_PENDING)

    def config(self) -> dict:
        cfg = asdict(self.settings)
        cfg["object_name_exclude"] = tuple(cfg["object_name_exclude"])
        return cfg

    def reset(self) -> None:
        self.processor.reset()
        self.state = StickyGripperState.OPEN
        self.previous_state = StickyGripperState.OPEN
        self.transition = None
        self.transition_reason = None
        self.attached_body_id = self.attached_body_name = None
        self.attached_joint_id = self.attached_qadr = self.attached_dadr = None
        self.R_eef_obj = self.p_eef_obj = None
        self.candidate_name = None
        self.candidate_distance = None
        self.candidate_counter = 0

    def rebind(self, env, ctrl):
        self.env, self.ctrl = env, ctrl
        self.model, self.data = ctrl.model, ctrl.data
        self.reset()
        self._cache_candidates()
        return self

    @staticmethod
    def _mujoco():
        import mujoco
        return mujoco

    @staticmethod
    def quat_to_mat(q_wxyz):
        mujoco = StickyGripper._mujoco()
        q = np.asarray(q_wxyz, dtype=float)
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("quaternion must have finite non-zero norm")
        out = np.zeros(9)
        mujoco.mju_quat2Mat(out, q / norm)
        return out.reshape(3, 3)

    @staticmethod
    def mat_to_quat(rotation):
        mujoco = StickyGripper._mujoco()
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(rotation, dtype=float).reshape(-1))
        norm = np.linalg.norm(q)
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError("rotation produced an invalid quaternion")
        return q / norm

    def get_eef_pose_mat(self):
        p, q = self.ctrl.get_eef_pose()
        return p, self.quat_to_mat(q), q

    def get_body_pose_mat(self, body_id):
        p = self.data.xpos[body_id].copy()
        rotation = self.data.xmat[body_id].reshape(3, 3).copy()
        return p, rotation, self.mat_to_quat(rotation)

    def _cache_candidates(self) -> None:
        mujoco = self._mujoco()
        rows = []
        excludes = tuple(x.lower() for x in self.settings.object_name_exclude)
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name:
                continue
            lowered = name.lower()
            if self.object_name_filter is not None and self.object_name_filter not in name:
                continue
            if any(token in lowered for token in excludes):
                continue
            jadr, jnum = int(self.model.body_jntadr[body_id]), int(self.model.body_jntnum[body_id])
            free_joints = [jid for jid in range(jadr, jadr + jnum)
                           if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE]
            if len(free_joints) != 1:
                continue
            jid = free_joints[0]
            rows.append(dict(body_id=body_id, body_name=name, joint_id=jid,
                             qadr=int(self.model.jnt_qposadr[jid]),
                             dadr=int(self.model.jnt_dofadr[jid])))
        self._candidate_metadata = sorted(rows, key=lambda row: row["body_name"])

    def freejoint_bodies(self) -> list[dict]:
        return [dict(row) for row in self._candidate_metadata]

    def _candidate_distance(self, body_id: int, eef_position: np.ndarray) -> float:
        """Body-origin distance; encapsulated for a future geometry-aware metric."""
        return float(np.linalg.norm(self.data.xpos[body_id] - eef_position))

    def list_candidates(self) -> list[dict]:
        p_eef, _, _ = self.get_eef_pose_mat()
        rows = []
        for item in self._candidate_metadata:
            pos = self.data.xpos[item["body_id"]].copy()
            rows.append({**item, "dist_to_eef": self._candidate_distance(item["body_id"], p_eef),
                         "pos": pos})
        return sorted(rows, key=lambda row: (round(row["dist_to_eef"], 12), row["body_name"]))

    def print_candidates(self, max_rows: int = 20):
        rows = self.list_candidates()
        print(f"Found {len(rows)} freejoint candidate bodies.")
        for row in rows[:max_rows]:
            print(f"body_id={row['body_id']:3d} | qadr={row['qadr']:3d} | "
                  f"dist={row['dist_to_eef']:.4f} | name={row['body_name']} | "
                  f"pos={np.round(row['pos'], 4)}")
        return rows

    def nearest_attachable_body(self):
        rows = self.list_candidates()
        return rows[0] if rows and rows[0]["dist_to_eef"] <= self.attach_distance else None

    def _set_state(self, state: StickyGripperState, reason: str):
        if state is self.state:
            return
        old = self.state
        self.previous_state = old
        self.state = state
        self.transition = f"{old.value} -> {state.value}"
        self.transition_reason = reason
        if self.verbose:
            LOG.info("StickyGripper %s (%s)", self.transition, reason)

    def attach(self, body_info=None) -> bool:
        if self.attached:
            return True
        body_info = body_info or self.nearest_attachable_body()
        if body_info is None:
            return False
        p_eef, rotation_eef, _ = self.get_eef_pose_mat()
        p_obj, rotation_obj, _ = self.get_body_pose_mat(body_info["body_id"])
        self.R_eef_obj = rotation_eef.T @ rotation_obj
        self.p_eef_obj = rotation_eef.T @ (p_obj - p_eef)
        self.attached_body_id = body_info["body_id"]
        self.attached_body_name = body_info["body_name"]
        self.attached_joint_id = body_info["joint_id"]
        self.attached_qadr = body_info["qadr"]
        self.attached_dadr = body_info["dadr"]
        self.zero_object_velocity()
        self._set_state(StickyGripperState.ATTACHED, "candidate proximity qualified")
        return True

    def release(self, reason: str = "explicit release") -> bool:
        if not self.attached:
            if self.state is not StickyGripperState.OPEN:
                self._set_state(StickyGripperState.OPEN, reason)
            return False
        self.attached_body_id = self.attached_body_name = None
        self.attached_joint_id = self.attached_qadr = self.attached_dadr = None
        self.R_eef_obj = self.p_eef_obj = None
        self.candidate_name = self.candidate_distance = None
        self.candidate_counter = 0
        self._set_state(StickyGripperState.OPEN, reason)
        return True

    def disable(self) -> None:
        self.release("sticky gripper disabled")
        self.reset()

    def zero_object_velocity(self):
        if self.attached_dadr is not None:
            self.data.qvel[self.attached_dadr:self.attached_dadr + 6] = 0.0

    def enforce_attachment(self):
        if not self.attached:
            return
        mujoco = self._mujoco()
        p_eef, rotation_eef, _ = self.get_eef_pose_mat()
        p_obj = p_eef + rotation_eef @ self.p_eef_obj
        q_obj = self.mat_to_quat(rotation_eef @ self.R_eef_obj)
        qadr = self.attached_qadr
        self.data.qpos[qadr:qadr + 3] = p_obj
        self.data.qpos[qadr + 3:qadr + 7] = q_obj
        self.zero_object_velocity()
        mujoco.mj_forward(self.model, self.data)

    def update_command(self, gripper_action) -> dict:
        """Advance transitions exactly once for one controller/VLA tick."""
        self.transition = self.transition_reason = None
        sample = self.processor.update(gripper_action)
        if not sample.valid:
            if self.verbose:
                LOG.warning("StickyGripper invalid command: %s", sample.error)
            return self.status(sample)

        intent = sample.qualified_intent
        if self.state is StickyGripperState.OPEN and intent is HandIntent.CLOSE:
            self._set_state(StickyGripperState.SEEKING, "close intent qualified")

        if self.state is StickyGripperState.SEEKING:
            if intent is HandIntent.OPEN:
                self.candidate_name = self.candidate_distance = None
                self.candidate_counter = 0
                self._set_state(StickyGripperState.OPEN, "open intent qualified")
            else:
                candidate = self.nearest_attachable_body()
                if candidate is None:
                    self.candidate_name = self.candidate_distance = None
                    self.candidate_counter = 0
                else:
                    name = candidate["body_name"]
                    if name == self.candidate_name:
                        self.candidate_counter += 1
                    else:
                        self.candidate_name = name
                        self.candidate_counter = 1
                    self.candidate_distance = candidate["dist_to_eef"]
                    if self.candidate_counter >= self.settings.candidate_dwell_ticks:
                        self.attach(candidate)

        elif self.state is StickyGripperState.ATTACHED:
            if intent is HandIntent.OPEN:
                self.release("open intent qualified")
            elif (self.processor.release_counter > 0 and
                  sample.filtered_command <= self.settings.release_threshold):
                self._set_state(StickyGripperState.RELEASE_PENDING,
                                "release dwell started")

        elif self.state is StickyGripperState.RELEASE_PENDING:
            if intent is HandIntent.OPEN:
                self.release("release dwell qualified")
            elif sample.filtered_command > self.settings.release_threshold:
                self._set_state(StickyGripperState.ATTACHED, "release request reversed")

        return self.status(sample)

    def update(self, gripper_action) -> dict:
        """Backward-compatible alias for controller-rate ``update_command``."""
        return self.update_command(gripper_action)

    def status(self, sample: CommandSample | None = None) -> dict:
        if sample is None:
            sample = self.processor._sample(True, None, False)
        return {
            "state": self.state.value,
            "previous_state": self.previous_state.value,
            "transition": self.transition,
            "transition_reason": self.transition_reason,
            "raw_command": sample.raw_command,
            "normalized_command": sample.normalized_command,
            "filtered_command": sample.filtered_command,
            "qualified_intent": sample.qualified_intent.value,
            "close_counter": sample.close_counter,
            "release_counter": sample.release_counter,
            "candidate_counter": self.candidate_counter,
            "candidate_name": self.candidate_name,
            "candidate_distance": self.candidate_distance,
            "command_valid": sample.valid,
            "command_error": sample.error,
            "attached": self.attached,
            "body_name": self.attached_body_name,
        }

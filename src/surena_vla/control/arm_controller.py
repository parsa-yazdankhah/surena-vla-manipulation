"""High-level SURENA arm controller from the validated implementation."""

from __future__ import annotations

import mink
import mujoco
import numpy as np

from .constants import ARM_INDICES, GAZEBO_INDEX_MAP_BARE
from .execution import get_loop_params, render_frame, smoothstep
from .ik import SurenaIK
from .joint_bridge import GazeboStyleController
from .sticky_gripper import StickyGripper
from .vla_adapter import OpenVLABridge
from .mujoco_utils import clamp_joints

class SurenaArmController:
    """
    Single entry point that builds the full Phase 1→3 controller stack and
    optional sticky gripper on top of any MuJoCo (model, data) pair.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 prefix: str = "robot0_", apply_home: bool = False):
        self.model  = model
        self.data   = data
        self.prefix = prefix

        self.bridge = GazeboStyleController(model, data, prefix=prefix,
                                            apply_home=apply_home)
        self.ik     = SurenaIK(model, data, self.bridge, prefix=prefix)
        self.vla    = OpenVLABridge(self.ik)
        self.sticky: StickyGripper | None = None
        self._sticky_env = None

        # Dynamic execution configuration.
        # Without these, Surena's arm can drift under gravity even while
        # holding a fixed joint target. Enable with:
        #     ctrl.configure_vla_execution()
        self._tracking_cfg: dict | None = None
        self._gravity_comp_enabled: bool = False
        self._gravity_comp_strength: float = 1.0
        self._arm_dof_ids_cache: np.ndarray | None = None

    # ── Phase 1 helpers ───────────────────────────────────────────────

    def set_joint_pose(self, arm_q_7: np.ndarray):
        self.bridge.set_joint_pose(arm_q_7)

    def get_arm_qpos(self) -> np.ndarray:
        return self.bridge.get_arm_qpos()

    # ── Phase 2 helpers ───────────────────────────────────────────────

    def solve_ik_nearest(self, pos: np.ndarray, so3: mink.SO3,
                         verbose: bool = True, **kwargs) -> dict:
        return self.ik.solve_nearest_ik(pos, so3, verbose=verbose, **kwargs)

    def move_eef_to(self, pos: np.ndarray, so3: mink.SO3,
                    verbose: bool = True,
                    teleport: bool = False,
                    robust: bool = True,
                    **kwargs) -> dict:
        return self.ik.move_eef_to(pos, so3, verbose=verbose,
                                   teleport=teleport, robust=robust, **kwargs)

    def get_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.ik.get_eef_pose()

    def current_eef_so3(self):
        return self.ik.current_eef_so3()

    # ── Sticky gripper helpers ────────────────────────────────────────

    def enable_sticky_gripper(self, env,
                              object_name_filter: str | None = None,
                              attach_distance: float = 0.09,
                              close_threshold: float = 0.5,
                              release_threshold: float = 0.5,
                              verbose: bool = True) -> StickyGripper:
        self._sticky_env = env
        self.sticky = StickyGripper(
            env=env,
            ctrl=self,
            object_name_filter=object_name_filter,
            attach_distance=attach_distance,
            close_threshold=close_threshold,
            release_threshold=release_threshold,
            verbose=verbose,
        )
        return self.sticky

    def disable_sticky_gripper(self):
        if self.sticky is not None:
            self.sticky.release()
        self.sticky = None
        self._sticky_env = None

    def sticky_update(self, gripper_action: float) -> dict:
        if self.sticky is None:
            return {"attached": False, "body_name": None}
        return self.sticky.update(gripper_action)

    def sticky_enforce(self):
        if self.sticky is not None:
            self.sticky.enforce_attachment()

    def print_gripper_candidates(self, max_rows: int = 20):
        if self.sticky is None:
            raise RuntimeError("Sticky gripper is not enabled. Call ctrl.enable_sticky_gripper(...).")
        return self.sticky.print_candidates(max_rows=max_rows)

    def get_sticky_object_pos(self, default_nan: bool = False):
        if self.sticky is None or self.sticky.attached_body_name is None:
            if default_nan:
                return np.array([np.nan, np.nan, np.nan])
            return None
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.sticky.attached_body_name)
        if bid < 0:
            if default_nan:
                return np.array([np.nan, np.nan, np.nan])
            return None
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[bid].copy()

    # ── Dynamic tracking / gravity compensation helpers ───────────────

    def get_arm_dof_ids(self) -> np.ndarray:
        """Return MuJoCo DOF ids corresponding to the 7 Surena arm joints."""
        if self._arm_dof_ids_cache is not None:
            return self._arm_dof_ids_cache.copy()

        dof_ids = []
        for gidx in ARM_INDICES:
            qadr = self.bridge._qadr[gidx]
            jid = None
            for j in range(self.model.njnt):
                if self.model.jnt_qposadr[j] == qadr:
                    jid = j
                    break
            if jid is None:
                raise RuntimeError(f"Could not find joint id for qadr={qadr}")
            dof_ids.append(self.model.jnt_dofadr[jid])

        self._arm_dof_ids_cache = np.asarray(dof_ids, dtype=int)
        return self._arm_dof_ids_cache.copy()

    def tune_tracking(self,
                      kp_major: float = 1200.0,
                      kp_wrist: float = 600.0,
                      damping_major: float = 45.0,
                      damping_wrist: float = 25.0,
                      disable_force_limits: bool = True,
                      verbose: bool = True):
        """
        Strengthen Surena's position actuators for dynamic tracking.

        This is required for reliable VLA evaluation: with the default weak
        gains, the arm can drift centimeters under gravity even when holding
        the current joint pose.
        """
        self._tracking_cfg = dict(
            kp_major=float(kp_major),
            kp_wrist=float(kp_wrist),
            damping_major=float(damping_major),
            damping_wrist=float(damping_wrist),
            disable_force_limits=bool(disable_force_limits),
        )

        wrist_gidx = {23, 24, 25}

        for gidx in ARM_INDICES:
            aid = self.bridge._act_id[gidx]
            qadr = self.bridge._qadr[gidx]

            jid = None
            for j in range(self.model.njnt):
                if self.model.jnt_qposadr[j] == qadr:
                    jid = j
                    break
            if jid is None:
                raise RuntimeError(f"Could not find joint id for qadr={qadr}")

            dadr = self.model.jnt_dofadr[jid]
            is_wrist = gidx in wrist_gidx
            kp = float(kp_wrist if is_wrist else kp_major)
            damping = float(damping_wrist if is_wrist else damping_major)

            # MuJoCo position actuator convention:
            #   force = kp * (ctrl - qpos)
            self.model.actuator_gainprm[aid, 0] = kp
            self.model.actuator_biasprm[aid, 1] = -kp
            self.model.dof_damping[dadr] = damping

            if disable_force_limits and hasattr(self.model, "actuator_forcelimited"):
                self.model.actuator_forcelimited[aid] = 0

        mujoco.mj_forward(self.model, self.data)

        if verbose:
            print("[TrackingTune] Applied stronger arm servo gains.")
            print(f"  major kp={kp_major}, wrist kp={kp_wrist}")
            print(f"  major damping={damping_major}, wrist damping={damping_wrist}")

        return self

    def enable_gravity_compensation(self, strength: float = 1.0, verbose: bool = True):
        """
        Enable arm-only gravity/bias compensation during raw stepping.

        The compensation writes qfrc_applied on the 7 arm DOFs before each
        MuJoCo step, using qfrc_bias. It is not teleporting; it only adds
        joint torques to cancel gravity/dynamic bias.
        """
        self._gravity_comp_enabled = True
        self._gravity_comp_strength = float(strength)
        self.get_arm_dof_ids()  # validate/cache ids now
        if verbose:
            print(f"[GravityComp] enabled | strength={self._gravity_comp_strength}")
        return self

    def disable_gravity_compensation(self, verbose: bool = True):
        self._gravity_comp_enabled = False
        self.data.qfrc_applied[:] = 0.0
        if verbose:
            print("[GravityComp] disabled")
        return self

    def apply_gravity_compensation(self, strength: float | None = None):
        """Apply one-step arm-only gravity/bias compensation to qfrc_applied."""
        if strength is None:
            strength = self._gravity_comp_strength
        arm_dofs = self.get_arm_dof_ids()
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[arm_dofs] = float(strength) * self.data.qfrc_bias[arm_dofs]

    def _step_once(self, env=None, gripper_action: float | None = None):
        """
        One raw MuJoCo step with Surena bridge, optional sticky gripper, and
        optional gravity compensation. Used by all notebook execution helpers.
        """
        if gripper_action is not None:
            self.sticky_update(gripper_action)

        if self._gravity_comp_enabled:
            mujoco.mj_forward(self.model, self.data)
            self.apply_gravity_compensation()

        mujoco.mj_step(self.model, self.data)

        # Avoid stale applied forces on the next step.
        self.data.qfrc_applied[:] = 0.0

        if gripper_action is not None:
            self.sticky_enforce()

    def configure_vla_execution(self,
                                kp_major: float = 1200.0,
                                kp_wrist: float = 600.0,
                                damping_major: float = 45.0,
                                damping_wrist: float = 25.0,
                                gravity_comp: bool = True,
                                gravity_strength: float = 1.0,
                                verbose: bool = True):
        """
        Recommended Surena execution setup for VLA/IK tests.

        Call once after controller creation/rebind:
            ctrl.configure_vla_execution()
        """
        self.tune_tracking(
            kp_major=kp_major,
            kp_wrist=kp_wrist,
            damping_major=damping_major,
            damping_wrist=damping_wrist,
            verbose=verbose,
        )
        if gravity_comp:
            self.enable_gravity_compensation(strength=gravity_strength, verbose=verbose)
        return self

    # ── Phase 3 helpers ───────────────────────────────────────────────

    def apply_vla_action(self, action_7d: np.ndarray,
                         verbose: bool = False,
                         update_sticky: bool = True,
                         **ik_kwargs) -> dict:
        self.vla.send_action(action_7d)
        info = self.vla.control_callback(verbose=verbose, **ik_kwargs)
        if update_sticky:
            info["sticky_gripper"] = self.sticky_update(action_7d[6])
        return info

    def reset_vla(self):
        self.vla.reset_to_current()

    # ── Execution helper ──────────────────────────────────────────────

    def execute_joint_target(self, q_goal, env,
                             ctrl_ticks: int = 80,
                             steps_per_ctrl: int | None = None,
                             record: bool = True,
                             render_stride: int = 5,
                             camera_name: str = "agentview",
                             gripper_action: float | None = None,
                             hold_ticks: int = 10):
        mj_model = env.sim.model._model
        mj_data = env.sim.data._data
        _, _, _, derived_steps_per_ctrl = get_loop_params(env, verbose=False)
        if steps_per_ctrl is None:
            steps_per_ctrl = derived_steps_per_ctrl

        q_start = self.get_arm_qpos().copy()
        q_goal = clamp_joints(np.asarray(q_goal, dtype=float).copy())

        frames = []
        eef_log = []
        obj_log = []

        total_ticks = int(ctrl_ticks)
        for tick in range(total_ticks):
            a = smoothstep((tick + 1) / max(total_ticks, 1))
            q_cmd = clamp_joints((1.0 - a) * q_start + a * q_goal)
            self.bridge.publish_arm_qpos(q_cmd)
            self.bridge.control_callback()

            for s in range(steps_per_ctrl):
                self._step_once(env=env, gripper_action=gripper_action)

                global_step = tick * steps_per_ctrl + s
                if record and (global_step % render_stride == 0):
                    frames.append(render_frame(env, camera_name=camera_name))
                    eef_log.append(self.get_eef_pose()[0].copy())
                    if gripper_action is not None:
                        obj_log.append(self.get_sticky_object_pos(default_nan=True))

        # Hold final target.
        self.bridge.publish_arm_qpos(q_goal)
        self.bridge.control_callback()
        for s in range(steps_per_ctrl * int(hold_ticks)):
            self._step_once(env=env, gripper_action=gripper_action)
            if record and (s % render_stride == 0):
                frames.append(render_frame(env, camera_name=camera_name))
                eef_log.append(self.get_eef_pose()[0].copy())
                if gripper_action is not None:
                    obj_log.append(self.get_sticky_object_pos(default_nan=True))

        env.sim.forward()
        if gripper_action is None:
            return frames, np.asarray(eef_log)
        return frames, np.asarray(eef_log), np.asarray(obj_log)

    # ── Robosuite Controller Reconstructor ────────────────────────────

    def rebind(self, env):
        """
        Re-point all controllers to the current sim's model/data.
        Must be called after every env.reset().
        """
        sticky_cfg = None
        if self.sticky is not None:
            sticky_cfg = self.sticky.config()

        tracking_cfg = self._tracking_cfg.copy() if self._tracking_cfg is not None else None
        gravity_enabled = bool(self._gravity_comp_enabled)
        gravity_strength = float(self._gravity_comp_strength)

        mj_model = env.sim.model._model
        mj_data = env.sim.data._data

        self.model = mj_model
        self.data = mj_data

        self.bridge = GazeboStyleController(mj_model, mj_data, prefix=self.prefix, apply_home=False)
        self.ik = SurenaIK(mj_model, mj_data, self.bridge, prefix=self.prefix)
        self.vla = OpenVLABridge(self.ik)

        self._arm_dof_ids_cache = None
        self._tracking_cfg = tracking_cfg
        self._gravity_comp_enabled = gravity_enabled
        self._gravity_comp_strength = gravity_strength

        # Restore damping + kp using names, not raw actuator indices.
        ARM_JOINT_DAMPING = 10.0
        ARM_KP_BY_GIDX = {
            12: 200,
            13: 200,
            14: 200,
            15: 200,
            23: 100,
            24: 100,
            25: 100,
        }

        for gidx in ARM_INDICES:
            jname = self.prefix + GAZEBO_INDEX_MAP_BARE[gidx][1]
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"Joint not found during rebind: {jname}")
            dadr = mj_model.jnt_dofadr[jid]
            mj_model.dof_damping[dadr] = ARM_JOINT_DAMPING

            aname = self.prefix + GAZEBO_INDEX_MAP_BARE[gidx][0]
            aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise RuntimeError(f"Actuator not found during rebind: {aname}")
            kp = ARM_KP_BY_GIDX[gidx]
            mj_model.actuator_gainprm[aid, 0] = kp
            mj_model.actuator_biasprm[aid, 1] = -kp

        # Re-apply user-selected strong tracking after reset/rebind.
        if tracking_cfg is not None:
            self.tune_tracking(**tracking_cfg, verbose=False)
        if gravity_enabled:
            self.enable_gravity_compensation(strength=gravity_strength, verbose=False)

        mujoco.mj_forward(mj_model, mj_data)
        self.reset_vla()

        self.sticky = None
        self._sticky_env = None
        if sticky_cfg is not None:
            self.enable_sticky_gripper(env=env, **sticky_cfg)

        return self

"""Robust multi-seed nearest-possible IK from the validated controller."""

from __future__ import annotations

import mink
import mujoco
import numpy as np

from .constants import (
    ARM_INDICES, EEF_SITE_BARE, GAZEBO_INDEX_MAP_BARE, HOME_QPOS, MAX_REACH
)
from .joint_bridge import GazeboStyleController
from .mujoco_utils import clamp_joints, mat_to_quat, _mj_id

class SurenaIK:
    """
    mink-based differential IK for Surena's 7-DOF right arm.

    This version includes robust nearest-possible IK. It first tries exact
    full-pose IK, then relaxed-orientation and position-only alternatives. It
    returns the best reachable joint target instead of failing uselessly when a
    6D EEF target is outside the local reachable set.
    """

    IK_DT     = 0.005
    MAX_ITERS = 300
    POS_TOL   = 1e-3
    ROT_TOL   = 1e-2

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 bridge: GazeboStyleController, prefix: str = ""):
        self.model  = model
        self.data   = data
        self.bridge = bridge
        self.prefix = prefix

        eef_site = prefix + EEF_SITE_BARE

        self.configuration = mink.Configuration(model)
        self.configuration.update(data.qpos)

        self.eef_task = mink.FrameTask(
            frame_name=eef_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.3,
            lm_damping=1e-4,
        )

        self.posture_task = mink.PostureTask(model, cost=1e-3)
        self.posture_task.set_target_from_configuration(self.configuration)

        self.limits = [mink.ConfigurationLimit(model=model)]
        self._site_id = _mj_id(model, mujoco.mjtObj.mjOBJ_SITE, eef_site)

    # ── Basic IK ─────────────────────────────────────────────────────

    def _run_ik(self, target_se3: mink.SE3, q_init=None):
        if q_init is not None:
            self.configuration.update(q_init)
        self.eef_task.set_target(target_se3)
        pe = re = float("inf")
        for _ in range(self.MAX_ITERS):
            err = self.eef_task.compute_error(self.configuration)
            pe  = np.linalg.norm(err[:3])
            re  = np.linalg.norm(err[3:])
            if pe < self.POS_TOL and re < self.ROT_TOL:
                return self.configuration.q.copy(), True, pe, re
            vel = mink.solve_ik(
                self.configuration, [self.eef_task, self.posture_task],
                dt=self.IK_DT, solver="daqp", limits=self.limits)
            self.configuration.integrate_inplace(vel, self.IK_DT)
        return self.configuration.q.copy(), False, pe, re

    def _arm_q_from_full_q(self, q_full: np.ndarray) -> np.ndarray:
        vals = []
        for gidx in ARM_INDICES:
            jname = self.prefix + GAZEBO_INDEX_MAP_BARE[gidx][1]
            jid = _mj_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qadr = self.model.jnt_qposadr[jid]
            vals.append(q_full[qadr])
        return clamp_joints(np.asarray(vals, dtype=float))

    def _full_q_with_arm_q(self, arm_q: np.ndarray, base_q=None) -> np.ndarray:
        q = self.data.qpos.copy() if base_q is None else np.asarray(base_q).copy()
        arm_q = clamp_joints(np.asarray(arm_q, dtype=float))
        for slot, gidx in enumerate(ARM_INDICES):
            jname = self.prefix + GAZEBO_INDEX_MAP_BARE[gidx][1]
            jid = _mj_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qadr = self.model.jnt_qposadr[jid]
            q[qadr] = arm_q[slot]
        return q

    def _evaluate_pose_error(self, q_full: np.ndarray,
                             target_pos: np.ndarray,
                             target_so3: mink.SO3) -> tuple[float, float]:
        conf = mink.Configuration(self.model)
        conf.update(q_full)
        eval_task = mink.FrameTask(
            frame_name=self.prefix + EEF_SITE_BARE,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-4,
        )
        eval_task.set_target(mink.SE3.from_rotation_and_translation(target_so3, target_pos))
        err = eval_task.compute_error(conf)
        return float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))

    def _solve_one_candidate(self, target_pos: np.ndarray, target_so3: mink.SO3,
                             q_seed: np.ndarray,
                             position_cost: float = 1.0,
                             orientation_cost: float = 0.3,
                             posture_cost: float = 1e-3,
                             max_iters: int = 500,
                             pos_tol: float | None = None,
                             rot_tol: float | None = None):
        pos_tol = self.POS_TOL if pos_tol is None else pos_tol
        rot_tol = self.ROT_TOL if rot_tol is None else rot_tol

        conf = mink.Configuration(self.model)
        conf.update(q_seed)

        eef_task = mink.FrameTask(
            frame_name=self.prefix + EEF_SITE_BARE,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1e-4,
        )
        eef_task.set_target(mink.SE3.from_rotation_and_translation(target_so3, target_pos))

        posture_task = mink.PostureTask(self.model, cost=posture_cost)
        posture_task.set_target_from_configuration(conf)

        limits = [mink.ConfigurationLimit(model=self.model)]
        last_pe = float("inf")
        last_re = float("inf")
        ok = False

        for _ in range(max_iters):
            err = eef_task.compute_error(conf)
            last_pe = float(np.linalg.norm(err[:3]))
            last_re = float(np.linalg.norm(err[3:]))
            orientation_ok = True if orientation_cost == 0 else (last_re < rot_tol)
            if last_pe < pos_tol and orientation_ok:
                ok = True
                break
            vel = mink.solve_ik(
                conf,
                [eef_task, posture_task],
                dt=self.IK_DT,
                solver="daqp",
                limits=limits,
            )
            conf.integrate_inplace(vel, self.IK_DT)

        return conf.q.copy(), ok, last_pe, last_re

    def _make_seeds(self, n_random: int = 10,
                    seed_std: float = 0.35,
                    rng_seed: int = 7) -> list[np.ndarray]:
        q_current = self.data.qpos.copy()
        current_arm = self._arm_q_from_full_q(q_current)
        seeds = [q_current.copy()]
        seeds.append(self._full_q_with_arm_q(HOME_QPOS, base_q=q_current))

        for j in range(len(current_arm)):
            for sign in [-1.0, 1.0]:
                q_arm = current_arm.copy()
                q_arm[j] += sign * 0.35
                seeds.append(self._full_q_with_arm_q(clamp_joints(q_arm), base_q=q_current))

        rng = np.random.default_rng(rng_seed)
        for _ in range(n_random):
            q_arm = current_arm + rng.normal(0.0, seed_std, size=len(current_arm))
            seeds.append(self._full_q_with_arm_q(clamp_joints(q_arm), base_q=q_current))

        return seeds

    def _project_target_to_soft_workspace(self, target_pos: np.ndarray,
                                          max_reach: float = MAX_REACH):
        shoulder_name = self.prefix + "r_arm_pitch"
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, shoulder_name)
        if sid < 0:
            return np.asarray(target_pos).copy(), False, None, None

        mujoco.mj_forward(self.model, self.data)
        shoulder = self.data.xpos[sid].copy()
        vec = np.asarray(target_pos, dtype=float) - shoulder
        dist = float(np.linalg.norm(vec))
        if dist <= max_reach or dist < 1e-9:
            return np.asarray(target_pos).copy(), False, shoulder, dist

        projected = shoulder + vec / dist * max_reach
        return projected, True, shoulder, dist

    def solve_nearest_ik(self, pos: np.ndarray, so3: mink.SO3,
                         n_random_seeds: int = 10,
                         seed_std: float = 0.35,
                         max_reach: float = MAX_REACH,
                         verbose: bool = False) -> dict:
        """
        Robust nearest-possible IK search.

        Returns a dict containing at least:
          q_sol, q_goal, ok, status, pos_err, rot_err, mode, target_variant
        """
        original_target = np.asarray(pos, dtype=float).copy()
        projected_target, was_projected, shoulder, shoulder_dist = \
            self._project_target_to_soft_workspace(original_target, max_reach=max_reach)

        target_variants = [("original", original_target)]
        if was_projected:
            target_variants.append(("workspace_projected", projected_target))

        mode_variants = [
            # mode_name, position_cost, orientation_cost, rot_score_weight
            ("full_pose",           1.0, 0.30, 1.00),
            ("relaxed_orientation", 1.0, 0.05, 0.25),
            ("position_only",       1.0, 0.00, 0.05),
        ]

        seeds = self._make_seeds(n_random=n_random_seeds, seed_std=seed_std)
        q_current = self.data.qpos.copy()
        current_arm = self._arm_q_from_full_q(q_current)

        candidates = []
        for target_name, target_for_solver in target_variants:
            for mode_name, pcost, ocost, rot_w in mode_variants:
                for seed_i, q_seed in enumerate(seeds):
                    try:
                        q_sol, solver_ok, solver_pe, solver_re = self._solve_one_candidate(
                            target_for_solver,
                            so3,
                            q_seed,
                            position_cost=pcost,
                            orientation_cost=ocost,
                            posture_cost=1e-3,
                            max_iters=max(500, self.MAX_ITERS),
                            pos_tol=self.POS_TOL,
                            rot_tol=self.ROT_TOL,
                        )
                        pe, re = self._evaluate_pose_error(q_sol, original_target, so3)
                        q_goal = self._arm_q_from_full_q(q_sol)
                        dq = float(np.linalg.norm(q_goal - current_arm))

                        # Position dominates. Orientation and joint motion regularize.
                        score = pe + rot_w * 0.08 * re + 0.002 * dq

                        candidates.append({
                            "q_sol": q_sol,
                            "q_goal": q_goal,
                            "ok": bool(pe < self.POS_TOL and re < self.ROT_TOL),
                            "solver_ok": bool(solver_ok),
                            "status": "ok" if (pe < self.POS_TOL and re < self.ROT_TOL)
                                      else "best_effort_nearest",
                            "pos_err": float(pe),
                            "rot_err": float(re),
                            "score": float(score),
                            "mode": mode_name,
                            "target_variant": target_name,
                            "seed_i": seed_i,
                            "solver_pos_err": float(solver_pe),
                            "solver_rot_err": float(solver_re),
                        })
                    except Exception as e:
                        if verbose:
                            print(
                                f"[RobustIK candidate failed] target={target_name}, "
                                f"mode={mode_name}, seed={seed_i}: {e}"
                            )

        if not candidates:
            raise RuntimeError("All robust IK candidates failed.")

        best = min(candidates, key=lambda c: c["score"])

        if verbose:
            print(
                "[RobustIK] "
                f"status={best['status']} | mode={best['mode']} | "
                f"target={best['target_variant']} | seed={best['seed_i']} | "
                f"pos_err={best['pos_err']:.4f} m | "
                f"rot_err={best['rot_err']:.4f} rad | score={best['score']:.5f}"
            )
            if was_projected:
                print(
                    "[RobustIK] requested target outside soft reach: "
                    f"dist_from_shoulder={shoulder_dist:.3f} m, max_reach={max_reach:.3f} m"
                )

        return best

    def _apply(self, q_sol: np.ndarray, teleport: bool = False):
        """
        Apply an IK full-q solution through the Surena bridge.

        teleport=False:
            Only command actuator targets through the bridge.
        teleport=True:
            Also directly writes qpos. Use only for diagnostics.
        """
        q_goal = self._arm_q_from_full_q(q_sol)

        if teleport:
            for slot, gidx in enumerate(ARM_INDICES):
                jname = self.prefix + GAZEBO_INDEX_MAP_BARE[gidx][1]
                jid = _mj_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                qadr = self.model.jnt_qposadr[jid]
                self.data.qpos[qadr] = q_goal[slot]
            mujoco.mj_forward(self.model, self.data)

        self.bridge.publish_arm_qpos(q_goal)
        self.bridge.control_callback()
        return q_goal

    def move_eef_to(self, pos: np.ndarray, so3: mink.SO3,
                    verbose: bool = False,
                    teleport: bool = False,
                    robust: bool = True,
                    **robust_kwargs) -> dict:
        if robust:
            out = self.solve_nearest_ik(pos, so3, verbose=verbose, **robust_kwargs)
            self._apply(out["q_sol"], teleport=teleport)
            return out

        target = mink.SE3.from_rotation_and_translation(so3, np.asarray(pos, dtype=float))
        q_sol, ok, pe, re = self._run_ik(target, q_init=self.configuration.q.copy())
        q_goal = self._apply(q_sol, teleport=teleport)
        return {
            "q_sol": q_sol,
            "q_goal": q_goal,
            "status": "ok" if ok else "best_effort",
            "ok": bool(ok),
            "pos_err": float(pe),
            "rot_err": float(re),
            "mode": "basic_full_pose",
            "target_variant": "original",
        }

    def get_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (pos [3], quat [4] wxyz) of the EEF site in world frame."""
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self._site_id].copy()
        mat = self.data.site_xmat[self._site_id].reshape(3, 3).copy()
        return pos, mat_to_quat(mat)

    def current_eef_so3(self):
        """Current EEF orientation as mink.SO3, built directly from site_xmat."""
        mujoco.mj_kinematics(self.model, self.data)
        R = self.data.site_xmat[self._site_id].reshape(3, 3).copy()
        if hasattr(mink.SO3, "from_matrix"):
            return mink.SO3.from_matrix(R)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        try:
            return mink.SO3(wxyz=quat)
        except Exception:
            return mink.SO3.from_wxyz(quat)

"""Deterministic hierarchical adaptive IK for the SURENA right arm."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence

import mink
import mujoco
import numpy as np

from .constants import ARM_INDICES, EEF_SITE_BARE, GAZEBO_INDEX_MAP_BARE, HOME_QPOS, MAX_REACH
from .joint_bridge import GazeboStyleController
from .mujoco_utils import mat_to_quat, _mj_id
from .robust_ik import (
    IKCandidate, IKStage, RobustIKConfig, collision_penalty,
    deduplicate_candidates, joint_limit_proximity_cost,
    elbow_out_penalty, local_acceleration_cost, normalized_candidate_score,
    select_best_candidate, singularity_penalty,
)

LOG = logging.getLogger(__name__)


class SurenaIK:
    """Mink differential IK with bounded, explicit fallback stages."""

    IK_DT = 0.005

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 bridge: GazeboStyleController, prefix: str = "",
                 config: RobustIKConfig | None = None):
        self.model, self.data, self.bridge, self.prefix = model, data, bridge, prefix
        self.config = config or RobustIKConfig()
        self._accepted_history: list[np.ndarray] = []
        self._last_commanded: np.ndarray | None = None
        self._build_model_cache()

    def _build_model_cache(self) -> None:
        self._site_id = _mj_id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                               self.prefix + EEF_SITE_BARE)
        self._joint_ids, self._qadr, self._dof_ids = [], [], []
        lower, upper = [], []
        for gidx in ARM_INDICES:
            name = self.prefix + GAZEBO_INDEX_MAP_BARE[gidx][1]
            jid = _mj_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_ids.append(jid)
            self._qadr.append(int(self.model.jnt_qposadr[jid]))
            self._dof_ids.append(int(self.model.jnt_dofadr[jid]))
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
            else:
                lo, hi = -math.pi, math.pi
            lower.append(float(lo)); upper.append(float(hi))
        self._qadr = np.asarray(self._qadr, dtype=int)
        self._dof_ids = np.asarray(self._dof_ids, dtype=int)
        self.joint_lower = np.asarray(lower)
        self.joint_upper = np.asarray(upper)
        self._available_solvers = self._discover_solvers()
        self._collision_pairs = self._build_collision_metadata()
        self.configuration = mink.Configuration(self.model)
        self.configuration.update(self.data.qpos.copy())
        self.limits = [mink.ConfigurationLimit(model=self.model)]

    def reset(self) -> None:
        self._accepted_history.clear()
        self._last_commanded = None
        self.configuration.update(self.data.qpos.copy())

    @staticmethod
    def _discover_solvers() -> tuple[str, ...]:
        try:
            import qpsolvers
            return tuple(sorted(qpsolvers.available_solvers))
        except Exception:
            return ()

    @property
    def available_solvers(self) -> tuple[str, ...]:
        return self._available_solvers

    def _build_collision_metadata(self) -> dict:
        """Cache robot and intentional-contact geoms for contact classification."""
        self._intentional_contact_body_names: tuple[str, ...] = ()
        self._intentional_fixed_geoms: frozenset[int] = frozenset()
        robot, intentional = set(), set()
        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            name = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            if any(token in name for token in ("robot", "r_arm", "r_elbow", "r_forearm", "r_hand")):
                robot.add(gid)
            if any(token in name for token in ("hand", "eef", "palm")):
                intentional.add(gid)
        return {"robot": frozenset(robot), "intentional": frozenset(intentional)}

    @property
    def intentional_contact_body_names(self) -> tuple[str, ...]:
        """Configured body-name candidates, retained so they can survive a rebind."""
        return self._intentional_contact_body_names

    def set_intentional_contact_bodies(self, body_names: Sequence[str]) -> None:
        """Allow soft palm/hand contact with task-designated body hierarchies.

        This deliberately operates at body (including descendant-body) scope,
        because fixture assets do not expose stable, sufficiently fine geom names.
        Such contacts still contribute collision cost; only hard rejection is
        suppressed. Unresolved candidate names are silently ignored.
        """
        self._intentional_contact_body_names = tuple(str(name) for name in body_names)
        body_ids = set()
        for name in self._intentional_contact_body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                body_ids.add(int(bid))
                break

        geoms = set()
        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            while bid > 0:
                if bid in body_ids:
                    geoms.add(gid)
                    break
                bid = int(self.model.body_parentid[bid])
        self._intentional_fixed_geoms = frozenset(geoms)

    def clear_intentional_contact_bodies(self) -> None:
        """Clear the task body-level soft-contact allowance."""
        self._intentional_contact_body_names = ()
        self._intentional_fixed_geoms = frozenset()

    def _current_full_q(self) -> np.ndarray:
        return self.data.qpos.copy()

    def _arm_q_from_full_q(self, q_full: np.ndarray) -> np.ndarray:
        return np.asarray(q_full, dtype=float)[self._qadr].copy()

    def _full_q_with_arm_q(self, arm_q: np.ndarray, base_q=None) -> np.ndarray:
        q = self._current_full_q() if base_q is None else np.asarray(base_q, dtype=float).copy()
        q[self._qadr] = np.asarray(arm_q, dtype=float)
        return q

    def _evaluate_pose_error(self, q_full, target_pos, target_so3) -> tuple[float, float]:
        conf = mink.Configuration(self.model)
        conf.update(np.asarray(q_full, dtype=float))
        task = mink.FrameTask(frame_name=self.prefix + EEF_SITE_BARE,
                              frame_type="site", position_cost=1.0,
                              orientation_cost=1.0, lm_damping=1e-4)
        task.set_target(mink.SE3.from_rotation_and_translation(target_so3, target_pos))
        err = task.compute_error(conf)
        return float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))

    def _solve_attempt(self, target_pos, target_so3, q_seed, *, solver,
                       orientation_cost, orientation_tolerance,
                       posture_cost=2e-3) -> tuple[np.ndarray, bool, int, str | None]:
        conf = mink.Configuration(self.model)
        conf.update(np.asarray(q_seed, dtype=float))
        task = mink.FrameTask(frame_name=self.prefix + EEF_SITE_BARE,
                              frame_type="site", position_cost=1.0,
                              orientation_cost=orientation_cost, lm_damping=1e-4)
        task.set_target(mink.SE3.from_rotation_and_translation(target_so3, target_pos))
        posture = mink.PostureTask(self.model, cost=posture_cost)
        posture_target = conf.q.copy()
        posture_target[self._qadr[1]] = np.clip(
            self.config.elbow_out_preferred_roll,
            self.joint_lower[1], self.joint_upper[1])
        posture.set_target(posture_target)
        for iteration in range(1, self.config.max_iterations + 1):
            err = task.compute_error(conf)
            pe, re = np.linalg.norm(err[:3]), np.linalg.norm(err[3:])
            if pe <= self.config.strict_position_tolerance and (
                    orientation_cost == 0 or re <= orientation_tolerance):
                return conf.q.copy(), True, iteration, None
            try:
                velocity = mink.solve_ik(conf, [task, posture], dt=self.IK_DT,
                                         solver=solver, limits=self.limits)
            except Exception as exc:
                return conf.q.copy(), False, iteration, f"{type(exc).__name__}: {exc}"
            if not np.isfinite(velocity).all():
                return conf.q.copy(), False, iteration, "solver returned non-finite velocity"
            conf.integrate_inplace(velocity, self.IK_DT)
        return conf.q.copy(), False, self.config.max_iterations, "iteration limit"

    def _jacobian_sigma(self, scratch: mujoco.MjData) -> float:
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self._site_id)
        jac = np.vstack((jacp[:, self._dof_ids], jacr[:, self._dof_ids]))
        values = np.linalg.svd(jac, compute_uv=False)
        return float(values[-1]) if len(values) else 0.0

    def _collision_metrics(self, scratch: mujoco.MjData) -> tuple[float | None, bool]:
        robot = self._collision_pairs["robot"]
        intentional = self._collision_pairs["intentional"]
        distances = []
        hard = False
        for index in range(scratch.ncon):
            contact = scratch.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if not ({g1, g2} & robot):
                continue
            # Ignore palm contact only when the other body is a movable
            # freejoint object; hand contact with a table/fixture remains risk.
            intentional_contact = bool({g1, g2} & intentional)
            other = None
            if intentional_contact:
                other = g2 if g1 in intentional else g1
                body = int(self.model.geom_bodyid[other])
                jadr, jnum = int(self.model.body_jntadr[body]), int(self.model.body_jntnum[body])
                movable = any(self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
                              for jid in range(jadr, jadr + jnum))
                if movable:
                    continue
            intentional_fixed = bool(
                intentional_contact and other in self._intentional_fixed_geoms
            )
            # Expected palm-to-task contact must remain physically active in
            # MuJoCo, but it must not make every useful contact posture lose
            # the IK candidate competition. Non-designated chassis contact is
            # still scored and hard-rejected normally.
            if intentional_fixed:
                continue
            distance = float(contact.dist)
            distances.append(distance)
            hard |= distance < self.config.collision_critical_distance
        return (min(distances) if distances else None), hard

    def _acceptance(self, candidate: IKCandidate) -> bool:
        if not candidate.feasible:
            return False
        if candidate.stage in (IKStage.FULL_POSE_PRIMARY, IKStage.FULL_POSE_ALTERNATES):
            return (candidate.position_error <= self.config.strict_position_tolerance and
                    candidate.orientation_error <= self.config.strict_orientation_tolerance)
        if candidate.stage is IKStage.RELAXED_ORIENTATION:
            tolerance = candidate.orientation_relaxation or self.config.orientation_relaxation_tolerances[-1]
            return (candidate.position_error <= self.config.relaxed_position_tolerance and
                    candidate.orientation_error <= tolerance)
        if candidate.stage is IKStage.POSITION_DOMINANT:
            return candidate.position_error <= self.config.position_dominant_tolerance
        return candidate.feasible

    def _evaluate_candidate(self, candidate, target_pos, target_so3,
                            current_arm) -> IKCandidate:
        violations = []
        q = np.asarray(candidate.q, dtype=float)
        if q.shape != current_arm.shape or not np.isfinite(q).all():
            candidate.hard_constraint_violations = ("invalid_joint_vector",)
            candidate.rejection_reason = "invalid joint vector"
            return candidate
        tolerance = self.config.joint_limit_hard_tolerance
        if np.any(q < self.joint_lower - tolerance) or np.any(q > self.joint_upper + tolerance):
            violations.append("joint_limit")
        candidate.joint_displacement = float(np.linalg.norm(q - current_arm))
        if np.max(np.abs(q - current_arm)) > self.config.maximum_joint_step:
            violations.append("maximum_joint_step")
        candidate.joint_acceleration_cost = local_acceleration_cost(
            q,
            self._accepted_history[-1] if self._accepted_history else None,
            self._accepted_history[-2] if len(self._accepted_history) > 1 else None,
        )
        candidate.joint_limit_cost, _ = joint_limit_proximity_cost(
            q, self.joint_lower, self.joint_upper, self.config.joint_limit_soft_margin)
        candidate.elbow_out_cost = elbow_out_penalty(
            q[1], self.config.elbow_out_soft_boundary,
            self.config.elbow_out_scale)
        q_full = self._full_q_with_arm_q(q, base_q=candidate.q_full)
        candidate.q_full = q_full
        try:
            candidate.position_error, candidate.orientation_error = self._evaluate_pose_error(
                q_full, target_pos, target_so3)
            scratch = mujoco.MjData(self.model)
            scratch.qpos[:] = q_full
            mujoco.mj_forward(self.model, scratch)
            sigma = self._jacobian_sigma(scratch)
            candidate.minimum_singular_value = sigma
            candidate.singularity_cost = singularity_penalty(
                sigma, self.config.singularity_warning_sigma)
            if sigma < self.config.singularity_critical_sigma:
                violations.append("critical_singularity")
            distance, hard_collision = self._collision_metrics(scratch)
            candidate.minimum_collision_distance = distance
            candidate.collision_cost = collision_penalty(
                distance, self.config.collision_warning_distance)
            if hard_collision:
                violations.append("collision_penetration")
        except Exception as exc:
            violations.append("evaluation_failure")
            candidate.rejection_reason = f"evaluation failed: {type(exc).__name__}: {exc}"
        candidate.hard_constraint_violations = tuple(dict.fromkeys(violations))
        candidate.feasible = not candidate.hard_constraint_violations
        candidate.accepted = self._acceptance(candidate)
        candidate.score = normalized_candidate_score(candidate, self.config)
        if not candidate.feasible and candidate.rejection_reason is None:
            candidate.rejection_reason = ", ".join(candidate.hard_constraint_violations)
        elif not candidate.accepted:
            candidate.rejection_reason = "stage acceptance tolerances not met"
        return candidate

    def _seed_set(self, current_full) -> list[tuple[str, np.ndarray]]:
        seeds = [("actual_current", current_full.copy())]
        if self._accepted_history:
            seeds.append(("previous_accepted", self._full_q_with_arm_q(
                self._accepted_history[-1], current_full)))
        if self._last_commanded is not None:
            seeds.append(("previous_commanded", self._full_q_with_arm_q(
                self._last_commanded, current_full)))
        seeds.append(("home", self._full_q_with_arm_q(HOME_QPOS, current_full)))
        center = 0.5 * (self.joint_lower + self.joint_upper)
        seeds.append(("joint_center", self._full_q_with_arm_q(center, current_full)))
        unique = []
        for name, seed in seeds:
            arm = self._arm_q_from_full_q(seed)
            if not any(np.linalg.norm(arm - self._arm_q_from_full_q(old)) <=
                       self.config.candidate_dedup_tolerance for _, old in unique):
                unique.append((name, seed))
        return unique

    def _attempt(self, stage, solver, seed_name, seed, target_pos, target_so3,
                 orientation_cost, orientation_tolerance, current_arm,
                 orientation_relaxation=None) -> IKCandidate:
        q_full, converged, iterations, error = self._solve_attempt(
            target_pos, target_so3, seed, solver=solver,
            orientation_cost=orientation_cost,
            orientation_tolerance=orientation_tolerance)
        candidate = IKCandidate(stage, solver, seed_name,
                                self._arm_q_from_full_q(q_full), q_full=q_full,
                                solver_converged=converged, iterations=iterations,
                                orientation_relaxation=orientation_relaxation)
        candidate = self._evaluate_candidate(candidate, target_pos, target_so3, current_arm)
        if error and candidate.rejection_reason is None:
            candidate.rejection_reason = error
        return candidate

    def solve_nearest_ik(self, pos: np.ndarray, so3: mink.SO3,
                         n_random_seeds: int | None = None,
                         seed_std: float | None = None,
                         max_reach: float = MAX_REACH,
                         verbose: bool = False) -> dict:
        """Run bounded stages and select the best candidate within each stage.

        Legacy random-seed arguments are accepted but intentionally ignored;
        fallback seeds are deterministic and bounded.
        """
        del n_random_seeds, seed_std, max_reach
        target_pos = np.asarray(pos, dtype=float)
        if target_pos.shape != (3,) or not np.isfinite(target_pos).all():
            raise ValueError("IK target position must be a finite shape-(3,) vector")
        started = time.perf_counter()
        current_full = self._current_full_q()  # authoritative seed every request
        current_arm = self._arm_q_from_full_q(current_full)
        seeds = self._seed_set(current_full)
        candidates: list[IKCandidate] = []
        escalation = None

        def add(candidate):
            nonlocal escalation
            candidates.append(candidate)
            if candidate.accepted:
                return candidate
            escalation = candidate.rejection_reason or "candidate not accepted"
            return None

        # Stage 1: let a small deterministic strict pool compete on the full
        # normalized score. This avoids committing to the first locally valid
        # posture while keeping the normal path bounded and predictable.
        strict_pool = []
        for seed_name, seed in seeds[:self.config.max_seeds_per_stage]:
            if len(candidates) >= self.config.max_total_attempts:
                break
            candidate = self._attempt(
                IKStage.FULL_POSE_PRIMARY, self.config.primary_solver,
                seed_name, seed, target_pos, so3, 0.30,
                self.config.strict_orientation_tolerance, current_arm)
            add(candidate)
            strict_pool.append(candidate)
        strict_best = select_best_candidate(strict_pool)
        selected = strict_best if strict_best is not None and strict_best.accepted else None

        # Stage 2: full pose, deterministic secondary seeds and installed solvers.
        if selected is None and self.config.enable_alternates:
            solvers = [name for name in self.config.alternate_solvers
                       if name in self.available_solvers and
                       name != self.config.primary_solver]
            for solver in solvers:
                for seed_name, seed in seeds[:self.config.max_seeds_per_stage]:
                    if len(candidates) >= self.config.max_total_attempts:
                        break
                    selected = add(self._attempt(
                        IKStage.FULL_POSE_ALTERNATES, solver, seed_name, seed,
                        target_pos, so3, 0.30,
                        self.config.strict_orientation_tolerance, current_arm))
                    if selected is not None:
                        break
                if selected is not None:
                    break

        # Stage 3: progressively relax orientation, strongest two seeds only.
        if selected is None and self.config.enable_relaxed_orientation:
            for cost, tolerance in zip(self.config.orientation_relaxation_costs,
                                       self.config.orientation_relaxation_tolerances):
                for seed_name, seed in seeds[:2]:
                    if len(candidates) >= self.config.max_total_attempts:
                        break
                    selected = add(self._attempt(
                        IKStage.RELAXED_ORIENTATION, self.config.primary_solver,
                        seed_name, seed, target_pos, so3, cost, tolerance,
                        current_arm, orientation_relaxation=tolerance))
                    if selected is not None:
                        break
                if selected is not None:
                    break

        # Stage 4: position dominant but retain a weak orientation preference.
        if selected is None and self.config.enable_position_dominant and len(candidates) < self.config.max_total_attempts:
            selected = add(self._attempt(
                IKStage.POSITION_DOMINANT, self.config.primary_solver,
                "actual_current", current_full, target_pos, so3, 0.01, math.pi,
                current_arm, orientation_relaxation=math.pi))

        candidates = deduplicate_candidates(candidates,
                                            self.config.candidate_dedup_tolerance)

        # Stage 5: line-search from current toward best prior iterate; every
        # interpolation is fully reevaluated and never implemented as clipping.
        if selected is None and self.config.enable_projection:
            source = select_best_candidate(candidates)
            if source is None:
                finite = [c for c in candidates if np.isfinite(c.q).all() and c.q.shape == current_arm.shape]
                source = min(finite, key=lambda c: c.position_error) if finite else None
            hold_pe, hold_re = self._evaluate_pose_error(current_full, target_pos, so3)
            if source is not None:
                delta = source.q - current_arm
                max_delta = np.max(np.abs(delta))
                alpha_cap = min(1.0, self.config.maximum_joint_step / max(max_delta, 1e-12))
                for alpha in np.linspace(alpha_cap, alpha_cap / self.config.projection_steps,
                                         self.config.projection_steps):
                    projected_q = current_arm + alpha * delta
                    projected = IKCandidate(
                        IKStage.NEAREST_FEASIBLE, "deterministic_line_search",
                        source.seed_name, projected_q,
                        q_full=self._full_q_with_arm_q(projected_q, current_full),
                        solver_converged=False)
                    projected = self._evaluate_candidate(projected, target_pos, so3, current_arm)
                    projected.accepted = bool(
                        projected.feasible and
                        projected.position_error + self.config.minimum_projection_improvement < hold_pe)
                    if projected.accepted:
                        projected.rejection_reason = None
                        candidates.append(projected)
                        selected = projected
                        break

        if selected is None:
            hold = IKCandidate(IKStage.HOLD_CURRENT, "none", "actual_current",
                               current_arm.copy(), q_full=current_full.copy(),
                               solver_converged=False, feasible=True, accepted=True)
            hold = self._evaluate_candidate(hold, target_pos, so3, current_arm)
            hold.accepted, hold.rejection_reason = True, None
            hold.score = normalized_candidate_score(hold, self.config)
            candidates.append(hold)
            selected = hold

        status = {
            IKStage.FULL_POSE_PRIMARY: "strict_full_pose",
            IKStage.FULL_POSE_ALTERNATES: "full_pose_recovered",
            IKStage.RELAXED_ORIENTATION: "relaxed_orientation",
            IKStage.POSITION_DOMINANT: "position_dominant",
            IKStage.NEAREST_FEASIBLE: "nearest_feasible_projected",
            IKStage.HOLD_CURRENT: "hold_current_no_safe_candidate",
        }[selected.stage]
        ok = selected.stage in (IKStage.FULL_POSE_PRIMARY, IKStage.FULL_POSE_ALTERNATES)
        out = {
            "q_sol": selected.q_full,
            "q_goal": selected.q.copy(),
            "status": status,
            "ok": ok,
            "pos_err": float(selected.position_error),
            "rot_err": float(selected.orientation_error),
            "mode": status,
            "target_variant": "original",
            "ik_stage": selected.stage.name,
            "solver": selected.solver,
            "seed": selected.seed_name,
            "candidate_count": len(candidates),
            "accepted": selected.accepted,
            "feasible": selected.feasible,
            "score": selected.score,
            "joint_displacement": selected.joint_displacement,
            "joint_acceleration_cost": selected.joint_acceleration_cost,
            "joint_limit_cost": selected.joint_limit_cost,
            "singularity_cost": selected.singularity_cost,
            "collision_cost": selected.collision_cost,
            "elbow_out_cost": selected.elbow_out_cost,
            "escalation_reason": escalation if selected.stage > IKStage.FULL_POSE_PRIMARY else None,
            "fallback_reason": None if ok else status,
            "attempt_count": len(candidates),
            "solve_time_ms": 1000.0 * (time.perf_counter() - started),
            "available_solvers": self.available_solvers,
        }
        if verbose or self.config.verbose_trace:
            out["candidate_trace"] = [candidate.compact() for candidate in candidates]
            LOG.info("IK %s stage=%s solver=%s seed=%s pe=%.4g re=%.4g score=%.4g",
                     status, selected.stage.name, selected.solver, selected.seed_name,
                     selected.position_error, selected.orientation_error, selected.score)
        return out

    def _apply(self, q_sol: np.ndarray, teleport: bool = False):
        q_goal = self._arm_q_from_full_q(q_sol)
        if teleport:
            self.data.qpos[self._qadr] = q_goal
            mujoco.mj_forward(self.model, self.data)
            self.reset()  # teleport invalidates continuity history
        self.bridge.publish_arm_qpos(q_goal)
        self.bridge.control_callback()
        self._last_commanded = q_goal.copy()
        return q_goal

    def move_eef_to(self, pos: np.ndarray, so3: mink.SO3,
                    verbose: bool = False, teleport: bool = False,
                    robust: bool = True, **kwargs) -> dict:
        # The hierarchical implementation is authoritative for both paths.
        out = self.solve_nearest_ik(pos, so3, verbose=verbose, **kwargs)
        self._apply(out["q_sol"], teleport=teleport)
        if out["ik_stage"] != IKStage.HOLD_CURRENT.name:
            self._accepted_history.append(out["q_goal"].copy())
            self._accepted_history = self._accepted_history[-2:]
        return out

    def get_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self._site_id].copy()
        rotation = self.data.site_xmat[self._site_id].reshape(3, 3).copy()
        return pos, mat_to_quat(rotation)

    def current_eef_so3(self):
        mujoco.mj_kinematics(self.model, self.data)
        rotation = self.data.site_xmat[self._site_id].reshape(3, 3).copy()
        if hasattr(mink.SO3, "from_matrix"):
            return mink.SO3.from_matrix(rotation)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.flatten())
        try:
            return mink.SO3(wxyz=quat)
        except Exception:
            return mink.SO3.from_wxyz(quat)

    def collision_status(self) -> tuple[float | None, bool]:
        """Classified collision state of the *live* simulation data"""
        return self._collision_metrics(self.data)

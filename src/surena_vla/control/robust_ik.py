"""Dependency-light policy types and scoring for hierarchical SURENA IK."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Sequence

import numpy as np


class IKStage(IntEnum):
    FULL_POSE_PRIMARY = 1
    FULL_POSE_ALTERNATES = 2
    RELAXED_ORIENTATION = 3
    POSITION_DOMINANT = 4
    NEAREST_FEASIBLE = 5
    HOLD_CURRENT = 6


@dataclass(frozen=True)
class IKScoreWeights:
    position: float = 4.0
    orientation: float = 1.0
    joint_displacement: float = 0.35
    joint_acceleration: float = 0.15
    joint_limit: float = 0.25
    singularity: float = 0.20
    collision: float = 2.0
    elbow_out: float = 0.25
    stage: float = 0.05

    def __post_init__(self):
        if any(value < 0 or not math.isfinite(value) for value in vars(self).values()):
            raise ValueError("IK score weights must be finite and non-negative")


@dataclass(frozen=True)
class RobustIKConfig:
    primary_solver: str = "daqp"
    alternate_solvers: tuple[str, ...] = ("osqp", "clarabel", "quadprog", "proxqp")
    strict_position_tolerance: float = 0.003
    strict_orientation_tolerance: float = 0.03
    relaxed_position_tolerance: float = 0.008
    orientation_relaxation_costs: tuple[float, ...] = (0.10, 0.03)
    orientation_relaxation_tolerances: tuple[float, ...] = (0.15, 0.40)
    position_dominant_tolerance: float = 0.015
    position_scale: float = 0.02
    orientation_scale: float = 0.20
    joint_motion_scale: float = 0.35
    acceleration_scale: float = 0.20
    elbow_out_preferred_roll: float = -0.50
    elbow_out_soft_boundary: float = -0.20
    elbow_out_scale: float = 0.30
    joint_limit_soft_margin: float = 0.15
    joint_limit_hard_tolerance: float = 1e-6
    singularity_warning_sigma: float = 0.04
    singularity_critical_sigma: float = 1e-5
    collision_warning_distance: float = 0.02
    collision_critical_distance: float = -0.01
    maximum_joint_step: float = 0.40
    max_iterations: int = 220
    max_total_attempts: int = 12
    max_seeds_per_stage: int = 4
    candidate_dedup_tolerance: float = 1e-6
    projection_steps: int = 8
    minimum_projection_improvement: float = 1e-4
    enable_alternates: bool = True
    enable_relaxed_orientation: bool = True
    enable_position_dominant: bool = True
    enable_projection: bool = True
    weights: IKScoreWeights = field(default_factory=IKScoreWeights)
    verbose_trace: bool = False

    def __post_init__(self):
        positive = (self.strict_position_tolerance, self.strict_orientation_tolerance,
                    self.relaxed_position_tolerance, self.position_dominant_tolerance,
                    self.position_scale, self.orientation_scale,
                    self.joint_motion_scale, self.acceleration_scale,
                    self.elbow_out_scale,
                    self.joint_limit_soft_margin, self.singularity_warning_sigma,
                    self.collision_warning_distance, self.maximum_joint_step)
        if any(value <= 0 or not math.isfinite(value) for value in positive):
            raise ValueError("IK tolerances, scales, and margins must be finite and positive")
        if self.strict_position_tolerance > self.relaxed_position_tolerance:
            raise ValueError("strict position tolerance must not exceed relaxed tolerance")
        if self.relaxed_position_tolerance > self.position_dominant_tolerance:
            raise ValueError("relaxed position tolerance must not exceed position-dominant tolerance")
        if self.collision_critical_distance > self.collision_warning_distance:
            raise ValueError("critical collision distance must not exceed warning distance")
        if not math.isfinite(self.elbow_out_preferred_roll):
            raise ValueError("preferred elbow-out shoulder roll must be finite")
        if not math.isfinite(self.elbow_out_soft_boundary):
            raise ValueError("elbow-out soft boundary must be finite")
        if self.singularity_critical_sigma > self.singularity_warning_sigma:
            raise ValueError("critical singularity sigma must not exceed warning sigma")
        if len(self.orientation_relaxation_costs) != len(self.orientation_relaxation_tolerances):
            raise ValueError("orientation relaxation costs and tolerances must have equal length")
        if any(a >= b for a, b in zip(self.orientation_relaxation_tolerances,
                                      self.orientation_relaxation_tolerances[1:])):
            raise ValueError("orientation relaxation tolerances must increase")
        if self.max_iterations < 1 or self.max_total_attempts < 1 or self.max_seeds_per_stage < 1:
            raise ValueError("IK iteration and attempt limits must be positive")
        if self.projection_steps < 1:
            raise ValueError("projection_steps must be positive")
        if not self.primary_solver:
            raise ValueError("primary_solver must be nonempty")


@dataclass
class IKCandidate:
    stage: IKStage
    solver: str
    seed_name: str
    q: np.ndarray
    q_full: np.ndarray | None = None
    position_error: float = math.inf
    orientation_error: float = math.inf
    joint_displacement: float = math.inf
    joint_acceleration_cost: float = 0.0
    joint_limit_cost: float = math.inf
    singularity_cost: float = 0.0
    collision_cost: float = 0.0
    elbow_out_cost: float = 0.0
    minimum_singular_value: float | None = None
    minimum_collision_distance: float | None = None
    hard_constraint_violations: tuple[str, ...] = ()
    score: float = math.inf
    solver_converged: bool = False
    feasible: bool = False
    accepted: bool = False
    rejection_reason: str | None = None
    iterations: int = 0
    orientation_relaxation: float | None = None

    def compact(self) -> dict:
        return {
            "stage": self.stage.name,
            "solver": self.solver,
            "seed": self.seed_name,
            "position_error": self.position_error,
            "orientation_error": self.orientation_error,
            "joint_displacement": self.joint_displacement,
            "joint_acceleration_cost": self.joint_acceleration_cost,
            "joint_limit_cost": self.joint_limit_cost,
            "singularity_cost": self.singularity_cost,
            "collision_cost": self.collision_cost,
            "elbow_out_cost": self.elbow_out_cost,
            "score": self.score,
            "solver_converged": self.solver_converged,
            "feasible": self.feasible,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "iterations": self.iterations,
        }


def local_acceleration_cost(q: np.ndarray, previous: np.ndarray | None,
                            previous_previous: np.ndarray | None) -> float:
    if previous is None or previous_previous is None:
        return 0.0
    delta2 = np.asarray(q) - 2.0 * np.asarray(previous) + np.asarray(previous_previous)
    return float(delta2 @ delta2)


def trajectory_smoothness(sequence: Sequence[np.ndarray]) -> float:
    q = np.asarray(sequence, dtype=float)
    if q.ndim != 2 or len(q) < 3:
        return 0.0
    second = q[2:] - 2.0 * q[1:-1] + q[:-2]
    return float(np.mean(np.sum(second * second, axis=1)))


def joint_limit_proximity_cost(q: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                               soft_margin_fraction: float = 0.15) -> tuple[float, np.ndarray]:
    """Smooth squared hinge cost inside a fraction of each joint range."""
    q, lower, upper = map(lambda x: np.asarray(x, dtype=float), (q, lower, upper))
    span = np.maximum(upper - lower, 1e-9)
    margin = np.maximum(soft_margin_fraction * span, 1e-9)
    clearance = np.minimum(q - lower, upper - q)
    per_joint = np.square(np.maximum(0.0, (margin - clearance) / margin))
    return float(np.sum(per_joint)), per_joint


def singularity_penalty(minimum_singular_value: float, warning_sigma: float) -> float:
    sigma = max(float(minimum_singular_value), 0.0)
    warning = max(float(warning_sigma), 1e-12)
    return float(np.square(max(0.0, (warning - sigma) / warning)))


def collision_penalty(minimum_distance: float | None, warning_distance: float) -> float:
    if minimum_distance is None:
        return 0.0
    warning = max(float(warning_distance), 1e-12)
    return float(np.square(max(0.0, (warning - minimum_distance) / warning)))


def elbow_out_penalty(shoulder_roll: float, soft_boundary: float,
                      scale: float) -> float:
    """Softly penalize inward (more positive) right shoulder-roll solutions."""
    return float(np.square(max(
        0.0, (float(shoulder_roll) - float(soft_boundary)) / float(scale))))


def normalized_candidate_score(candidate: IKCandidate, config: RobustIKConfig) -> float:
    """Weighted dimensionless score; feasibility must be checked separately."""
    w = config.weights
    terms = (
        w.position * (candidate.position_error / config.position_scale) ** 2,
        w.orientation * (candidate.orientation_error / config.orientation_scale) ** 2,
        w.joint_displacement * (candidate.joint_displacement / config.joint_motion_scale) ** 2,
        w.joint_acceleration * (candidate.joint_acceleration_cost / config.acceleration_scale**2),
        w.joint_limit * candidate.joint_limit_cost,
        w.singularity * candidate.singularity_cost,
        w.collision * candidate.collision_cost,
        w.elbow_out * candidate.elbow_out_cost,
        w.stage * max(0, int(candidate.stage) - 1),
    )
    return float(sum(terms)) if all(math.isfinite(x) for x in terms) else math.inf


def deduplicate_candidates(candidates: Sequence[IKCandidate], tolerance: float) -> list[IKCandidate]:
    unique: list[IKCandidate] = []
    for candidate in candidates:
        if not any(candidate.q.shape == old.q.shape and
                   np.linalg.norm(candidate.q - old.q) <= tolerance for old in unique):
            unique.append(candidate)
    return unique


def select_best_candidate(candidates: Sequence[IKCandidate]) -> IKCandidate | None:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    return min(feasible, key=lambda item: (not item.accepted, item.score,
                                           int(item.stage), item.solver, item.seed_name)) \
        if feasible else None

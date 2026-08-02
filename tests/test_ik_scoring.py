"""Pure tests for normalized IK feasibility and quality terms."""

import numpy as np
import pytest
import importlib.util
from pathlib import Path
import sys

path = Path(__file__).parents[1] / "src/surena_vla/control/robust_ik.py"
spec = importlib.util.spec_from_file_location("_robust_ik_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
IKCandidate = module.IKCandidate
IKStage = module.IKStage
RobustIKConfig = module.RobustIKConfig
collision_penalty = module.collision_penalty
deduplicate_candidates = module.deduplicate_candidates
joint_limit_proximity_cost = module.joint_limit_proximity_cost
local_acceleration_cost = module.local_acceleration_cost
normalized_candidate_score = module.normalized_candidate_score
select_best_candidate = module.select_best_candidate
singularity_penalty = module.singularity_penalty
trajectory_smoothness = module.trajectory_smoothness


def candidate(**kwargs):
    base = dict(stage=IKStage.FULL_POSE_PRIMARY, solver="daqp",
                seed_name="actual_current", q=np.zeros(7),
                position_error=0.002, orientation_error=0.02,
                joint_displacement=0.1, joint_acceleration_cost=0.0,
                joint_limit_cost=0.0, singularity_cost=0.0,
                collision_cost=0.0, feasible=True, accepted=True)
    base.update(kwargs)
    return IKCandidate(**base)


def test_configuration_invariants():
    with pytest.raises(ValueError):
        RobustIKConfig(strict_position_tolerance=0.02,
                       relaxed_position_tolerance=0.01)
    with pytest.raises(ValueError):
        RobustIKConfig(collision_critical_distance=0.03,
                       collision_warning_distance=0.02)


def test_continuity_can_beat_slight_cartesian_advantage():
    cfg = RobustIKConfig()
    jump = candidate(position_error=0.001, joint_displacement=1.2)
    smooth = candidate(position_error=0.002, joint_displacement=0.05)
    jump.score = normalized_candidate_score(jump, cfg)
    smooth.score = normalized_candidate_score(smooth, cfg)
    assert smooth.score < jump.score


def test_infeasible_low_error_never_beats_feasible():
    unsafe = candidate(position_error=0.0, feasible=False, accepted=False)
    safe = candidate(position_error=0.01)
    unsafe.score, safe.score = 0.0, 10.0
    assert select_best_candidate([unsafe, safe]) is safe


def test_score_uses_explicit_normalization_scales():
    cfg = RobustIKConfig(position_scale=0.02, orientation_scale=0.2)
    a = candidate(position_error=0.02, orientation_error=0.0)
    b = candidate(position_error=0.0, orientation_error=0.2)
    a.score, b.score = normalized_candidate_score(a, cfg), normalized_candidate_score(b, cfg)
    assert a.score == pytest.approx(cfg.weights.position +
                                    cfg.weights.joint_displacement * (0.1 / cfg.joint_motion_scale) ** 2)
    assert b.score == pytest.approx(cfg.weights.orientation +
                                    cfg.weights.joint_displacement * (0.1 / cfg.joint_motion_scale) ** 2)


def test_joint_limit_cost_is_symmetric_and_smooth():
    lower, upper = -np.ones(2), np.ones(2)
    center, _ = joint_limit_proximity_cost(np.array([0.0, 0.0]), lower, upper)
    near_low, low_parts = joint_limit_proximity_cost(np.array([-0.95, 0.0]), lower, upper)
    near_high, high_parts = joint_limit_proximity_cost(np.array([0.95, 0.0]), lower, upper)
    assert center == 0.0
    assert near_low == pytest.approx(near_high)
    assert low_parts[0] == pytest.approx(high_parts[0])
    assert low_parts[0] > 0
    farther, _ = joint_limit_proximity_cost(np.array([0.99, 0.0]), lower, upper)
    assert farther > near_low


def test_singularity_and_collision_costs_have_warning_and_hard_inputs():
    assert singularity_penalty(0.04, 0.04) == 0.0
    assert singularity_penalty(0.001, 0.04) > singularity_penalty(0.03, 0.04)
    assert collision_penalty(None, 0.02) == 0.0
    assert collision_penalty(0.03, 0.02) == 0.0
    assert collision_penalty(-0.01, 0.02) > collision_penalty(0.01, 0.02)


def test_candidate_deduplication_is_deterministic():
    first = candidate(seed_name="actual_current")
    duplicate = candidate(seed_name="home", q=np.full(7, 1e-8))
    different = candidate(seed_name="joint_center", q=np.full(7, 0.1))
    assert deduplicate_candidates([first, duplicate, different], 1e-6) == [first, different]


def test_local_second_difference_exactly():
    q0, q1, q2 = np.zeros(2), np.array([1.0, 2.0]), np.array([3.0, 5.0])
    expected = np.array([1.0, 1.0])
    assert local_acceleration_cost(q2, q1, q0) == pytest.approx(expected @ expected)
    assert local_acceleration_cost(q2, q1, None) == 0.0


def test_trajectory_smoothness_distinguishes_linear_and_oscillatory():
    smooth = [np.array([i, 2 * i], float) for i in range(5)]
    oscillatory = [np.array([(-1) ** i, 0.0]) for i in range(5)]
    assert trajectory_smoothness(smooth) == 0.0
    assert trajectory_smoothness(oscillatory) > 0.0


def test_stage_priority_is_deterministic_tie_breaker():
    strict = candidate(stage=IKStage.FULL_POSE_PRIMARY)
    relaxed = candidate(stage=IKStage.RELAXED_ORIENTATION)
    strict.score = normalized_candidate_score(strict, RobustIKConfig())
    relaxed.score = normalized_candidate_score(relaxed, RobustIKConfig())
    assert strict.score < relaxed.score

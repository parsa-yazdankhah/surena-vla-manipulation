"""Pure geometry and anti-windup tests for articulation guidance."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


path = Path(__file__).parents[1] / "src/surena_vla/control/contact_guidance.py"
spec = importlib.util.spec_from_file_location("_contact_guidance_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_positive_hinge_motion_follows_axis_cross_radius():
    delta = module.hinge_tangent_delta(
        point=np.array([1.0, 0.0, 0.0]),
        hinge=np.zeros(3),
        axis=np.array([0.0, 0.0, 1.0]),
        joint_direction=1.0,
        step=0.006,
    )
    assert delta == pytest.approx([0.0, 0.006, 0.0])


def test_negative_hinge_motion_reverses_tangent():
    positive = module.hinge_tangent_delta(
        np.array([1.0, 0.0, 0.0]), np.zeros(3),
        np.array([0.0, 0.0, 1.0]), 1.0, 0.006)
    negative = module.hinge_tangent_delta(
        np.array([1.0, 0.0, 0.0]), np.zeros(3),
        np.array([0.0, 0.0, 1.0]), -1.0, 0.006)
    assert negative == pytest.approx(-positive)


def test_finite_arc_step_preserves_radius_and_has_expected_direction():
    point = np.array([1.0, 0.0, 0.0])
    delta = module.hinge_arc_delta(
        point, np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0, 0.1)
    result = point + delta
    assert np.linalg.norm(result[:2]) == pytest.approx(1.0)
    assert result[1] > 0.0


def test_hinge_tangent_removes_axis_component_and_has_requested_length():
    delta = module.hinge_tangent_delta(
        np.array([1.0, 0.0, 4.0]), np.zeros(3),
        np.array([0.0, 0.0, 2.0]), 1.0, 0.01)
    assert np.linalg.norm(delta) == pytest.approx(0.01)
    assert delta[2] == pytest.approx(0.0)


def test_target_lag_is_unchanged_inside_limit_and_clamped_outside():
    actual = np.array([1.0, 2.0, 3.0])
    near = actual + np.array([0.005, 0.0, 0.0])
    far = actual + np.array([0.10, 0.0, 0.0])
    assert module.clamp_target_lag(near, actual, 0.015) == pytest.approx(near)
    clamped = module.clamp_target_lag(far, actual, 0.015)
    assert np.linalg.norm(clamped - actual) == pytest.approx(0.015)


def test_guidance_config_accepts_preset_mapping_defaults():
    cfg = module.ContactGuidanceConfig.from_mapping({
        "joint_names": ["door_joint"],
        "fixture_body_names": ["microwave_main"],
        "target_q": 0.0,
    })
    assert cfg.joint_names == ("door_joint",)
    assert cfg.fixture_body_names == ("microwave_main",)
    assert cfg.tangent_step == pytest.approx(0.012)
    assert cfg.stall_steps == 5

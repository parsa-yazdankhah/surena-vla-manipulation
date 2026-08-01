"""CPU-only tests for continuous sticky-hand command interpretation."""

import numpy as np
import pytest
import importlib.util
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

from surena_vla.gripper_command import (
    GripperCommandConfig,
    GripperCommandProcessor,
    HandIntent,
)


def load_sticky_module():
    """Load the MuJoCo-lazy module without importing control/__init__ (Mink)."""
    name = "_sticky_gripper_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parents[1] / "src/surena_vla/control/sticky_gripper.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fake_sticky(candidate=None, **config):
    module = load_sticky_module()
    sticky = module.StickyGripper.__new__(module.StickyGripper)
    sticky.settings = module.StickyGripperConfig(**config)
    sticky.processor = GripperCommandProcessor(sticky.settings.command_config())
    sticky.reset()
    sticky.nearest_attachable_body = lambda: candidate[0] if candidate else None

    def attach(self, body_info=None):
        row = body_info or self.nearest_attachable_body()
        if row is None or self.attached:
            return self.attached
        self.attached_body_name = row["body_name"]
        self.attached_body_id = row["body_id"]
        self._set_state(module.StickyGripperState.ATTACHED,
                        "candidate proximity qualified")
        return True

    sticky.attach = MethodType(attach, sticky)
    return sticky, module


def processor(**kwargs):
    defaults = dict(close_threshold=0.65, release_threshold=0.35,
                    close_dwell_ticks=2, release_dwell_ticks=2,
                    normalization_scale=1.0, normalization_offset=0.0)
    defaults.update(kwargs)
    return GripperCommandProcessor(GripperCommandConfig(**defaults))


def test_continuous_values_preserved_and_caller_array_not_mutated():
    action = np.array([0, 0, 0, 0, 0, 0, 0.12], dtype=float)
    original = action.copy()
    p = processor()
    samples = []
    for value in [0.12, 0.38, 0.57, 0.81]:
        action[6] = value
        samples.append(p.update(action[6]))
        assert action[6] == value
    assert [s.raw_command for s in samples] == [0.12, 0.38, 0.57, 0.81]
    assert samples[-1].normalized_command == 0.81
    assert set(action[:6]) == set(original[:6])


def test_close_requires_dwell_and_transitions_once():
    p = processor(close_dwell_ticks=2)
    assert p.update(0.8).qualified_intent is HandIntent.OPEN
    second = p.update(0.8)
    assert second.qualified_intent is HandIntent.CLOSE
    assert second.intent_changed
    assert not p.update(0.9).intent_changed


def test_deadband_retains_qualified_intent():
    p = processor()
    p.update(0.8)
    p.update(0.8)
    for value in [0.64, 0.36, 0.5, 0.4]:
        assert p.update(value).qualified_intent is HandIntent.CLOSE


def test_single_sample_close_noise_does_not_chatter():
    p = processor(close_dwell_ticks=2)
    for value in [0.7, 0.5, 0.8, 0.4, 0.66, 0.3]:
        assert p.update(value).qualified_intent is HandIntent.OPEN


def test_release_dwell_and_reversal():
    p = processor(release_dwell_ticks=3)
    p.update(0.9)
    p.update(0.9)
    assert p.update(0.2).qualified_intent is HandIntent.CLOSE
    assert p.update(0.8).qualified_intent is HandIntent.CLOSE
    assert p.release_counter == 0
    assert p.update(0.2).qualified_intent is HandIntent.CLOSE
    assert p.update(0.2).qualified_intent is HandIntent.CLOSE
    last = p.update(0.2)
    assert last.qualified_intent is HandIntent.OPEN
    assert last.intent_changed
    assert not p.update(0.1).intent_changed


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [0.8], np.array([0.8])])
def test_invalid_input_is_fail_safe(value):
    p = processor()
    sample = p.update(value)
    assert not sample.valid
    assert sample.qualified_intent is HandIntent.OPEN
    assert p.close_counter == 0


def test_reset_clears_filter_intent_and_counters():
    p = processor(filter_alpha=0.5)
    p.update(0.8)
    p.update(0.8)
    p.reset()
    assert p.raw_command is None
    assert p.filtered_command is None
    assert p.qualified_intent is HandIntent.OPEN
    assert p.close_counter == p.release_counter == 0


def test_threshold_order_is_validated():
    with pytest.raises(ValueError, match="strictly less"):
        GripperCommandConfig(close_threshold=0.5, release_threshold=0.5)


def test_ema_equation_and_no_clipping():
    p = processor(filter_alpha=0.25)
    assert p.update(2.0).filtered_command == 2.0
    assert p.update(0.0).filtered_command == pytest.approx(1.5)


def test_bridge_default_maps_zero_to_close_strength_without_mutating_raw():
    p = GripperCommandProcessor()
    sample = p.update(0.12)
    assert sample.raw_command == 0.12
    assert sample.normalized_command == pytest.approx(0.88)


def test_seeking_without_candidate_then_candidate_arrives_and_attaches():
    holder = []
    sticky, _ = fake_sticky(holder, close_dwell_ticks=2,
                            candidate_dwell_ticks=2)
    sticky.update_command(0.1)
    status = sticky.update_command(0.1)
    assert status["state"] == "SEEKING"
    assert not status["attached"]
    holder.append({"body_name": "object_a", "body_id": 7,
                   "dist_to_eef": 0.02})
    assert sticky.update_command(0.1)["state"] == "SEEKING"
    status = sticky.update_command(0.1)
    assert status["state"] == "ATTACHED"
    assert status["attached"] and status["body_name"] == "object_a"


def test_attached_deadband_release_pending_reversal_and_single_release():
    holder = [{"body_name": "object_a", "body_id": 7, "dist_to_eef": 0.02}]
    sticky, _ = fake_sticky(holder, close_dwell_ticks=1,
                            candidate_dwell_ticks=1, release_dwell_ticks=2)
    sticky.update_command(0.1)
    assert sticky.attached
    assert sticky.update_command(0.5)["state"] == "ATTACHED"
    assert sticky.update_command(0.9)["state"] == "RELEASE_PENDING"
    assert sticky.update_command(0.5)["state"] == "ATTACHED"
    assert sticky.update_command(0.9)["state"] == "RELEASE_PENDING"
    released = sticky.update_command(0.9)
    assert released["transition"] == "RELEASE_PENDING -> OPEN"
    assert not released["attached"]
    assert sticky.update_command(0.9)["transition"] is None


def test_backward_compatible_status_and_reset():
    sticky, _ = fake_sticky([])
    status = sticky.status()
    assert status["attached"] is False and status["body_name"] is None
    sticky.update_command(0.1)
    sticky.reset()
    status = sticky.status()
    assert status["state"] == "OPEN"
    assert status["candidate_name"] is None
    assert status["filtered_command"] is None


def test_candidate_tie_break_is_deterministic_by_name():
    sticky, _ = fake_sticky([])
    sticky._candidate_metadata = [
        {"body_id": 2, "body_name": "zeta", "joint_id": 2, "qadr": 7, "dadr": 6},
        {"body_id": 1, "body_name": "alpha", "joint_id": 1, "qadr": 0, "dadr": 0},
    ]
    sticky.data = SimpleNamespace(xpos=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], float))
    sticky.get_eef_pose_mat = lambda: (np.zeros(3), np.eye(3), np.array([1, 0, 0, 0]))
    assert [row["body_name"] for row in sticky.list_candidates()] == ["alpha", "zeta"]


def test_minimal_mujoco_attachment_capture_enforce_and_release():
    mujoco = pytest.importorskip("mujoco")
    module = load_sticky_module()
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <body name="robot_eef" pos="0 0 0"><freejoint/><site name="eef"/><geom type="sphere" size="0.01"/></body>
          <body name="object_a" pos="0.02 0 0"><freejoint/><geom type="sphere" size="0.01"/></body>
        </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    eef_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_eef")

    class Controller:
        def __init__(self):
            self.model, self.data = model, data

        def get_eef_pose(self):
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, data.xmat[eef_bid])
            return data.xpos[eef_bid].copy(), quat

    sticky = module.StickyGripper(None, Controller(), attach_distance=0.05,
                                  close_dwell_ticks=1,
                                  candidate_dwell_ticks=1,
                                  release_dwell_ticks=1)
    status = sticky.update_command(0.0)
    assert status["attached"] and status["body_name"] == "object_a"
    captured = sticky.p_eef_obj.copy()
    eef_qadr = model.jnt_qposadr[model.body_jntadr[eef_bid]]
    data.qpos[eef_qadr] = 0.1
    mujoco.mj_forward(model, data)
    sticky.enforce_attachment()
    object_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a")
    assert data.xpos[object_bid, 0] == pytest.approx(0.1 + captured[0])
    assert sticky.update_command(1.0)["transition"] == "ATTACHED -> OPEN"
    before = data.xpos[object_bid].copy()
    data.qpos[eef_qadr] = 0.2
    mujoco.mj_forward(model, data)
    sticky.enforce_attachment()
    assert np.allclose(data.xpos[object_bid], before)

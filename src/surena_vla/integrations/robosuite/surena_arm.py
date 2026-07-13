"""robosuite ManipulatorModel for the validated SURENA configuration."""

from __future__ import annotations

import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

from surena_vla.paths import SURENA_ARM_XML, validate_asset_layout

class SurenaArm(ManipulatorModel):
    """
    Surena V humanoid, arms-only configuration.
    Both legs and the torso are fixed; only the 7-DOF arms are actuated.
    """

    def __init__(self, idn=0):
        validate_asset_layout()
        super().__init__(str(SURENA_ARM_XML), idn=idn)

    # ------------------------------------------------------------------
    # Required robosuite interface properties
    # ------------------------------------------------------------------

    @property
    def default_mount(self):
        """No mount: Surena's base_link is fixed in the MJCF (robosuite uses None → NullMount)."""
        return None

    @property
    def default_gripper(self):
        """
        Surena has fixed hand_link at the wrist — no actuated gripper.
        Use NullGripper for now; replace with a custom gripper model later.
        """
        return None

    @property
    def default_controller_config(self):
        """
        OSC_POSE gives 6-DOF Cartesian control at each EEF — the natural
        interface for language-conditioned VLA policies.
        Switch to JOINT_VELOCITY if you prefer low-level joint control.
        """
        return "joint_position"

    @property
    def init_qpos(self):
        """
        Home pose (radians) for 7 arm joints (7 right).
        Matches validate_surena_mujoco.py HOME_QPOS / actuator order.
        """
        return np.array([-0.0349, -0.1046, 0.0, -0.2616, 0.0, 0.0, 0.0])

    # @property
    # def base_xpos_offset(self):
    #     """
    #     Offset from arena origin. "table" must be a callable(table_length) per robosuite API.
    #     Surena is placed slightly behind the table so arms reach the workspace.
    #     """
    #     return {
    #         "bins": (-0.5, -0.1, 0),
    #         "empty": (-0.29, 0, 0),
    #         "table": lambda table_length: (-0.50 - table_length / 2, 0, 0),
    #     }
    
    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.29, 0, 0),
            "table": lambda table_length: (-0.25 - table_length / 2, 0.15, 0),
            "kitchen_table": lambda table_length: (-0.25 - table_length / 2, 0.15, 0),
            "living_room_table": lambda table_length: (-0.5 - table_length / 2, 0.15, 0),
            "study_table": lambda table_length: (-0.34 - table_length / 2, 0.15, 0),
        }

    @property
    def top_offset(self):
        """Approx height from base_link origin to shoulder level (m)."""
        return np.array([0.0, 0.0, 1.26])

    @property
    def _horizontal_radius(self):
        """Collision-exclusion radius around the robot base (m)."""
        return 0.5

    @property
    def arm_type(self):
        return "single"

    @property
    def _eef_name(self):
        """Body names where NullGripper attaches (site is right_eef_site)."""
        return "r_hand_link"

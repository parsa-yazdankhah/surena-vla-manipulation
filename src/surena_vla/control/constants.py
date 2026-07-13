"""Canonical constants from the validated SURENA controller."""

from __future__ import annotations

import numpy as np

ARRAY_LEN = 29   # matches /joint_angles_gazebo layout

# Gazebo-bridge index → (bare_actuator_name, bare_joint_name)
GAZEBO_INDEX_MAP_BARE = {
    12: ("act_r_arm_pitch",    "r_arm_pitch_joint"),
    13: ("act_r_arm_roll",     "r_arm_roll_joint"),
    14: ("act_r_elbow_pitch",  "r_elbow_pitch_joint"),
    15: ("act_r_forearm_roll", "r_forearm_roll_joint"),
    23: ("act_r_forearm_link", "r_forearm_link_joint"),
    24: ("act_r_hand_roll",    "r_hand_roll_joint"),
    25: ("act_r_hand_pitch",   "r_hand_pitch_joint"),
}

ARM_INDICES = [12, 13, 14, 15, 23, 24, 25]
N_JOINTS = len(ARM_INDICES)

# Derived from the canonical index map so these lists cannot diverge.
ARM_ACTUATOR_NAMES_BARE = tuple(
    GAZEBO_INDEX_MAP_BARE[index][0]
    for index in ARM_INDICES
)

ARM_JOINT_NAMES_BARE = tuple(
    GAZEBO_INDEX_MAP_BARE[index][1]
    for index in ARM_INDICES
)

EEF_SITE_BARE = "right_eef_site"   # site name in surena_arm.xml

JOINT_LIMITS = np.array([[-1.57, 1.57]] * N_JOINTS)

DEG = np.pi / 180.0
# HOME_QPOS = np.array([
#     -2.0 * DEG,   # r_arm_pitch
#     -6.0 * DEG,   # r_arm_roll
#      0.0,          # r_elbow_pitch
#    -15.0 * DEG,   # r_forearm_roll
#      0.0,          # r_forearm_link
#      0.0,          # r_hand_roll
#      0.0,          # r_hand_pitch
# ])
HOME_QPOS = np.array([
    -0.20,     # r_arm_pitch
    -0.50,     # r_arm_roll
     0.00,     # r_elbow_pitch
    -1.55,     # r_forearm_roll
     0.00,     # r_forearm_link
     0.15,     # r_hand_roll
     0.00,     # r_hand_pitch
])

MAX_REACH = 0.70

# robosuite's model initialization follows MJCF joint-tree order and is
# intentionally independent from HOME_QPOS, which follows the historical ROS array.
ROBOSUITE_INIT_QPOS = np.array(
    [-0.0349, -0.1046, 0.0, -0.2616, 0.0, 0.0, 0.0],
    dtype=float,
)

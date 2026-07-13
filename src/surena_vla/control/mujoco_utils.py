"""Small MuJoCo helpers shared by the SURENA control modules."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .constants import ARM_INDICES, ARRAY_LEN, JOINT_LIMITS

def clamp_joints(q: np.ndarray) -> np.ndarray:
    return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def make_gazebo_array(arm_q: np.ndarray) -> np.ndarray:
    arr = np.zeros(ARRAY_LEN)
    for slot, idx in enumerate(ARM_INDICES):
        arr[idx] = arm_q[slot]
    return arr


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def _mj_id(model, obj_type, name: str) -> int:
    """Resolve a MuJoCo name → id, raising clearly if not found."""
    eid = mujoco.mj_name2id(model, obj_type, name)
    if eid < 0:
        raise RuntimeError(
            f"MuJoCo name not found: type={obj_type.name}  name='{name}'\n"
            "Check the prefix argument and the MJCF file."
        )
    return eid

# Public alias retained for package users; the validated source uses _mj_id.
mj_id = _mj_id


def raw_model_data_from_env(env: Any):
    """Extract raw MuJoCo model/data from robosuite or LIBERO wrappers."""
    current = env
    sim = None
    for _ in range(8):
        sim = getattr(current, "sim", None)
        if sim is not None:
            break
        current = getattr(current, "env", None)
        if current is None:
            break
    if sim is None:
        raise TypeError("Could not locate a .sim object on the supplied environment")
    wrapped_model = getattr(sim, "model", None)
    wrapped_data = getattr(sim, "data", None)
    if wrapped_model is None or wrapped_data is None:
        raise TypeError("Environment sim does not expose model and data")
    return getattr(wrapped_model, "_model", wrapped_model), getattr(wrapped_data, "_data", wrapped_data)

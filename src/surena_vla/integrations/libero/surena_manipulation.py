"""
SurenaManipulation — compact LIBERO-90 environment library for Surena V.

===============================================================================

This module wraps a compact curated subset of LIBERO-90 kitchen and study tasks
around the custom Surena V right-arm robot. The catalog is limited to kitchen and study tasks only.

Design goals
------------
1. Keep the fixed Surena camera exactly as used in the notebook.
2. Keep object/furniture placement inside the Surena right-arm reachable region.
3. Keep a small, debuggable preset set instead of the full LIBERO-90 catalog.
4. Provide robust local heuristic _check_success() implementations for selected
   tasks while still allowing LIBERO's native BDDL checker when it works.
5. Keep backwards-compatible class names and notebook-facing helper APIs:
   register_surena_tasks(), list_presets(), get_preset(), make_env().

Package usage
-------------
    from surena_vla.integrations import register_all
    register_all()

    import surena_vla.integrations.libero.surena_manipulation as sm
"""

from __future__ import annotations

import copy
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union

import mujoco
import numpy as np

from surena_vla.integrations.robosuite import register_surena_robot
from .paths import get_bddl_root

register_surena_robot()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BDDL_ROOT = get_bddl_root()
LIBERO_PATH = str(_BDDL_ROOT.parent)
BDDL_DIR = str(_BDDL_ROOT / "libero_90")
ROBOT_PREFIX = "robot0_"
SURENA_RIGHT_ARM_CONTACT_BIT = 2


def _bddl(name_or_path: str) -> str:
    """Return an absolute BDDL path."""
    if os.path.isabs(name_or_path):
        return name_or_path
    return os.path.join(BDDL_DIR, name_or_path)


# ---------------------------------------------------------------------------
# LIBERO problem-base imports
# ---------------------------------------------------------------------------


def _import_problem_class(module_name: str, class_name: str):
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name), f"{class_name}"
    except Exception as exc:  # keep import robust across LIBERO revisions
        return None, f"{class_name} unavailable: {exc}"


_TabletopBase, _TABLETOP_IMPORT_STATUS = _import_problem_class(
    "libero.libero.envs.problems.libero_tabletop_manipulation",
    "Libero_Tabletop_Manipulation",
)
if _TabletopBase is None:
    raise ImportError(_TABLETOP_IMPORT_STATUS)

_KitchenBase, _KITCHEN_IMPORT_STATUS = _import_problem_class(
    "libero.libero.envs.problems.libero_kitchen_tabletop_manipulation",
    "Libero_Kitchen_Tabletop_Manipulation",
)
_StudyBase, _STUDY_IMPORT_STATUS = _import_problem_class(
    "libero.libero.envs.problems.libero_study_tabletop_manipulation",
    "Libero_Study_Tabletop_Manipulation",
)

_KitchenBase = _KitchenBase or _TabletopBase
_StudyBase = _StudyBase or _TabletopBase



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIGHT_ARM_JOINTS: Tuple[str, ...] = (
    "r_arm_pitch_joint",
    "r_arm_roll_joint",
    "r_elbow_pitch_joint",
    "r_forearm_roll_joint",
    "r_forearm_link_joint",
    "r_hand_pitch_joint",
    "r_hand_roll_joint",
)

RIGHT_ARM_ACTUATORS: Dict[str, str] = {
    "r_arm_pitch_joint": "act_r_arm_pitch",
    "r_arm_roll_joint": "act_r_arm_roll",
    "r_elbow_pitch_joint": "act_r_elbow_pitch",
    "r_forearm_roll_joint": "act_r_forearm_roll",
    "r_forearm_link_joint": "act_r_forearm_link",
    "r_hand_pitch_joint": "act_r_hand_pitch",
    "r_hand_roll_joint": "act_r_hand_roll",
}

# These values are deliberately conservative. They keep the wrist lifted and the
# shoulder rolled toward the right-arm workspace. Individual profiles may override
# a subset of joints.
DEFAULT_HOME_Q: Dict[str, float] = {
    "r_arm_pitch_joint": -0.20,
    "r_arm_roll_joint": -0.50,
    "r_elbow_pitch_joint": 0.00,
    "r_forearm_roll_joint": -1.55,
    "r_forearm_link_joint": 0.00,
    "r_hand_pitch_joint": 0.12,
    "r_hand_roll_joint": 0.00,
}

DRAWER_CLOSED = 0.00
DRAWER_OPEN = -0.16
MICROWAVE_CLOSED = 0.00
MICROWAVE_OPEN = -1.25
STOVE_OFF = 0.00
STOVE_ON = 1.00

# Placement z-levels.
Z_KITCHEN_OBJ = 0.970
Z_LOW_OBJ = 0.930
Z_FURNITURE = 0.905
Z_LIVING_OBJ = 0.970
Z_STUDY_OBJ = 0.937
Z_STUDY_FURNITURE = 0.887

# Workspace diagnostics; these are geometric warnings, not hard safety limits.
REACH_WARN_M = 0.52
REACH_HARD_M = 0.58

AGENTVIEW_CAMERA = {
    "name": "agentview",
    "pos": "0.32 0.05 1.49",
    "xyaxes": "0 -1 0 0.38 0 -1",
    "fovy": "60",
}

# MuJoCo quaternion order is (w, x, y, z).
# 180 deg rotation about vertical z-axis.
Z_ROT_180_QUAT = (0.0, 0.0, 0.0, 1.0)
# 90 deg rotation about vertical z-axis.
Z_ROT_90_QUAT = (0.70710678, 0.0, 0.0, 0.70710678)
# 270 deg rotation about vertical z-axis.
Z_ROT_270_QUAT = (-0.70710678, 0.0, 0.0, 0.70710678)


# ---------------------------------------------------------------------------
# Small helper constructors
# ---------------------------------------------------------------------------


def _deep_merge(base: Mapping[str, Any], override: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    out = copy.deepcopy(dict(base))
    if not override:
        return out
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _with_home(**overrides: float) -> Dict[str, float]:
    out = dict(DEFAULT_HOME_Q)
    out.update({k: float(v) for k, v in overrides.items()})
    return out


def _free(names: Union[str, Sequence[str]], xyz: Sequence[float], quat: Optional[Sequence[float]] = None,
          required: bool = False) -> Dict[str, Any]:
    if isinstance(names, str):
        names = [names]
    spec: Dict[str, Any] = {"names": list(names), "xyz": tuple(float(v) for v in xyz), "required": required}
    if quat is not None:
        spec["quat"] = tuple(float(q) for q in quat)
    return spec


def _body(
    names: Union[str, Sequence[str]],
    xyz: Sequence[float],
    required: bool = False,
    quat: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    if isinstance(names, str):
        names = [names]
    spec: Dict[str, Any] = {
        "names": list(names),
        "pos": tuple(float(v) for v in xyz),
        "required": required,
    }
    if quat is not None:
        spec["quat"] = tuple(float(q) for q in quat)
    return spec


def _joint(names: Union[str, Sequence[str]], q: float, required: bool = False) -> Dict[str, Any]:
    if isinstance(names, str):
        names = [names]
    return {"names": list(names), "q": float(q), "required": required}


def _point(xyz: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(float(v) for v in xyz)


# ---------------------------------------------------------------------------
# Name candidates
# ---------------------------------------------------------------------------

OBJ = {
    "black_bowl_1": ["akita_black_bowl_1_joint0"],
    "black_bowl_2": ["akita_black_bowl_2_joint0"],
    "black_bowl_3": ["akita_black_bowl_3_joint0"],
    "white_bowl_1": ["white_bowl_1_joint0"],
    "plate_1": ["plate_1_joint0"],
    "plate_2": ["plate_2_joint0"],
    "ketchup": ["ketchup_1_joint0"],
    "butter_1": ["butter_1_joint0"],
    "butter_2": ["butter_2_joint0"],
    "chocolate_pudding": ["chocolate_pudding_1_joint0"],
    "frying_pan": ["chefmate_8_frypan_1_joint0", "frying_pan_1_joint0"],
    "moka_pot_1": ["moka_pot_1_joint0"],
    "moka_pot_2": ["moka_pot_2_joint0"],
    "red_mug": ["red_coffee_mug_1_joint0", "red_mug_1_joint0"],
    "white_mug": ["white_mug_1_joint0", "white_yellow_mug_1_joint0", "porcelain_mug_1_joint0"],
    "yellow_white_mug": ["yellow_and_white_mug_1_joint0", "white_yellow_mug_1_joint0"],
    "book_1": ["black_book_1_joint0", "book_1_joint0", "yellow_book_1_joint0"],
    "book_2": ["black_book_2_joint0", "book_2_joint0", "yellow_book_2_joint0"],
    "book_3": ["black_book_3_joint0", "book_3_joint0", "yellow_book_3_joint0"],
}

BODY = {
    "white_cabinet": ["white_cabinet_1_main"],
    "wooden_cabinet": ["wooden_cabinet_1_main"],
    "stove": ["flat_stove_1_main", "stove_1_main"],
    "microwave": ["microwave_1_main"],
    "microwave_door": ["microwave_1_microdoorroot", "microwave_1_door"],
    "caddy": ["desk_caddy_1_main", "caddy_1_main"],
    "shelf": ["wooden_two_layer_shelf_1_main", "wooden_shelf_1_main", "cabinet_shelf_1_main", "shelf_1_main"],
    "study_table": ["study_table"],
}

JNT = {
    "top_drawer": ["white_cabinet_1_top_level", "wooden_cabinet_1_top_level"],
    "middle_drawer": ["white_cabinet_1_middle_level", "wooden_cabinet_1_middle_level"],
    "bottom_drawer": ["white_cabinet_1_bottom_level", "wooden_cabinet_1_bottom_level"],
    "stove_knob": ["flat_stove_1_button", "flat_stove_1_knob_joint", "stove_1_knob_joint", "stove_1_knob_1_joint", "flat_stove_1_knob_1_joint"],
    "microwave_door": ["microwave_1_microjoint", "microwave_1_door_joint", "microwave_1_door"],
}


# ---------------------------------------------------------------------------
# Placement profiles
# ---------------------------------------------------------------------------

BASE_PROFILES: Dict[str, Dict[str, Any]] = {
    "cabinet_base": {
        "home_q": _with_home(r_forearm_roll_joint=-1.55),
        "objects": {
            "black_bowl_1": _free(OBJ["black_bowl_1"], (-0.43, -0.20, Z_KITCHEN_OBJ)),
            "ketchup": _free(OBJ["ketchup"], (-0.27, -0.22, Z_KITCHEN_OBJ)),
            "butter_1": _free(OBJ["butter_1"], (-0.30, -0.12, Z_LOW_OBJ)),
            "butter_2": _free(OBJ["butter_2"], (-0.35, -0.16, Z_LOW_OBJ)),
            "chocolate_pudding": _free(OBJ["chocolate_pudding"], (-0.34, -0.26, Z_KITCHEN_OBJ)),
            "plate_1": _free(OBJ["plate_1"], (-0.32, -0.08, Z_KITCHEN_OBJ)),
        },
        "bodies": {
            "white_cabinet": _body(BODY["white_cabinet"], (-0.33, 0.23, Z_FURNITURE)),
            # "wooden_cabinet": _body(BODY["wooden_cabinet"], (-0.37, 0.18, Z_FURNITURE), quat=Z_ROT_180_QUAT),
        },
        "joints": {
            "top_drawer": _joint(JNT["top_drawer"], DRAWER_CLOSED),
            "middle_drawer": _joint(JNT["middle_drawer"], DRAWER_CLOSED),
            "bottom_drawer": _joint(JNT["bottom_drawer"], DRAWER_CLOSED),
        },
        "targets": {
            "top_drawer_inside": _point((-0.33, 0.13, Z_KITCHEN_OBJ)),
            "bottom_drawer_inside": _point((-0.33, 0.13, Z_LOW_OBJ)),
        },
    },

    "bowl_plate_base": {
        "home_q": _with_home(r_forearm_roll_joint=-1.45),
        "objects": {
            "black_bowl_1": _free(OBJ["black_bowl_1"], (-0.34, -0.13, Z_KITCHEN_OBJ)),
            "plate_1": _free(OBJ["plate_1"], (-0.31, 0.00, Z_KITCHEN_OBJ)),
        },
        "bodies": {},
        "joints": {},
    },

    "stove_base": {
        "home_q": _with_home(r_arm_roll_joint=-0.45, r_forearm_roll_joint=-1.35),
        "objects": {
            "frying_pan": _free(OBJ["frying_pan"], (-0.42, -0.12, Z_KITCHEN_OBJ)),
            "moka_pot_1": _free(OBJ["moka_pot_1"], (-0.30, -0.10, Z_KITCHEN_OBJ)),
        },
        "bodies": {
            "stove": _body(BODY["stove"], (-0.43, 0.10, Z_FURNITURE)),
        },
        "joints": {
            "stove_knob": _joint(JNT["stove_knob"], STOVE_OFF),
        },
        "targets": {
            "burner": _point((-0.30, 0.10, Z_KITCHEN_OBJ)),
        },
    },

    "microwave_base": {
        "home_q": _with_home(r_forearm_roll_joint=-1.42),
        # LIBERO's microwave accepts only contact bit 1, while the Surena
        # right arm emits bit 2. Extend affinity after model compilation.
        "contact_compat_bodies": {"microwave": SURENA_RIGHT_ARM_CONTACT_BIT},
        "objects": {},
        "bodies": {
            "microwave": _body(BODY["microwave"], (-0.35, 0.18, Z_FURNITURE)),
        },
        "joints": {
            "microwave_door": _joint(JNT["microwave_door"], MICROWAVE_CLOSED),
        },
        "targets": {
            "front_of_white_mug": _point((-0.36, -0.31, Z_KITCHEN_OBJ)),
        },
    },

    "study_caddy_base": {
        "home_q": _with_home(r_arm_pitch_joint=0.00, r_forearm_roll_joint=-1.63, r_hand_pitch_joint=0.20),
        "objects": {
            "book_1": _free(OBJ["book_1"], (-0.38, -0.13, Z_STUDY_OBJ)),
            "book_2": _free(OBJ["book_2"], (-0.35, -0.13, Z_STUDY_OBJ)),
            "yellow_white_mug": _free(OBJ["yellow_white_mug"], (-0.25, -0.15, Z_STUDY_OBJ)),
            "white_mug": _free(OBJ["white_mug"], (-0.25, -0.17, Z_STUDY_OBJ)),
            "red_mug": _free(OBJ["red_mug"], (-0.30, -0.14, Z_STUDY_OBJ)),
            "caddy": _free(["caddy_1_joint0"], (-0.46, 0.16, Z_STUDY_FURNITURE)),
        },
        "bodies": {
            "caddy": _body(BODY["caddy"], (-0.46, 0.16, Z_STUDY_FURNITURE)),
        },
        "joints": {},
        "targets": {
            "caddy_left": _point((-0.50, 0.16, Z_STUDY_OBJ)),
            "caddy_right": _point((-0.42, 0.16, Z_STUDY_OBJ)),
            "caddy_front": _point((-0.46, 0.09, Z_STUDY_OBJ)),
            "caddy_back": _point((-0.46, 0.23, Z_STUDY_OBJ)),
        },
    },

    "study_shelf_base": {
        "home_q": _with_home(r_arm_pitch_joint=-0.10, r_forearm_roll_joint=-1.50, r_hand_pitch_joint=0.22),
        "objects": {
            "book_1": _free(OBJ["book_1"], (-0.47, -0.12, Z_STUDY_OBJ)),
            "book_2": _free(OBJ["book_2"], (-0.40, -0.10, Z_STUDY_OBJ)),
            "book_3": _free(OBJ["book_3"], (-0.33, -0.16, Z_STUDY_OBJ)),
        },
        "bodies": {
            "shelf": _body(BODY["shelf"], (-0.44, 0.23, Z_STUDY_FURNITURE)),
        },
        "joints": {},
        "targets": {
            "shelf_under": _point((-0.44, 0.18, Z_STUDY_FURNITURE + 0.02)),
            "shelf_on": _point((-0.44, 0.23, Z_STUDY_OBJ + 0.06)),
            "shelf_top": _point((-0.44, 0.26, Z_STUDY_OBJ + 0.16)),
        },
    },
}


PLACEMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    # Cabinet / drawer
    "cabinet_open_top": _deep_merge(BASE_PROFILES["cabinet_base"], {
        "joints": {"top_drawer": _joint(JNT["top_drawer"], DRAWER_CLOSED)},
    }),
    "cabinet_close_bottom": _deep_merge(BASE_PROFILES["cabinet_base"], {
        "bodies": {"white_cabinet": _body(BODY["white_cabinet"], (-0.36, 0.25, Z_FURNITURE)),},
        "joints": {"bottom_drawer": _joint(JNT["bottom_drawer"], DRAWER_OPEN)},
        "objects": {"plate_1": _free(OBJ["plate_1"], (-0.12, -0.11, Z_KITCHEN_OBJ)),}
    }),
    "cabinet_close_top": _deep_merge(BASE_PROFILES["cabinet_base"], {
        "joints": {"top_drawer": _joint(JNT["top_drawer"], DRAWER_OPEN)},
    }),
    "cabinet_put_ketchup_top": _deep_merge(BASE_PROFILES["cabinet_base"], {
        "joints": {"top_drawer": _joint(JNT["top_drawer"], DRAWER_OPEN)},
        "objects": {
            "ketchup": _free(OBJ["ketchup"], (-0.29, -0.15, Z_KITCHEN_OBJ)),
            "plate_1": _free(OBJ["plate_1"], (-0.32, 0.0, Z_KITCHEN_OBJ)),
            "black_bowl_1": _free(OBJ["black_bowl_1"], (-0.40, -0.33, Z_KITCHEN_OBJ)),
        },
    }),
    "cabinet_put_bowl_top": _deep_merge(BASE_PROFILES["cabinet_base"], {
        "joints": {"top_drawer": _joint(JNT["top_drawer"], DRAWER_OPEN)},
        "objects": {
            "black_bowl_1": _free(OBJ["black_bowl_1"], (-0.33, -0.12, Z_KITCHEN_OBJ)),
            "ketchup": _free(OBJ["ketchup"], (-0.20, -0.26, Z_KITCHEN_OBJ)),
            "plate_1": _free(OBJ["plate_1"], (-0.26, 0.0, Z_KITCHEN_OBJ)),

        },
    }),
    "cabinet_butter_close": _deep_merge(BASE_PROFILES["cabinet_base"], {
        "joints": {"top_drawer": _joint(JNT["top_drawer"], DRAWER_OPEN)},
        "objects": {
            "butter_1": _free(OBJ["butter_1"], (-0.54, -0.22, Z_LOW_OBJ)),
            "butter_2": _free(OBJ["butter_2"], (-0.42, -0.22, Z_LOW_OBJ)),
        },
    }),
    "cabinet_chocolate_close": _deep_merge(BASE_PROFILES["cabinet_base"], {
        "joints": {"top_drawer": _joint(JNT["top_drawer"], DRAWER_OPEN)},
        "objects": {"chocolate_pudding": _free(OBJ["chocolate_pudding"], (-0.54, -0.22, Z_KITCHEN_OBJ))},
    }),

    # Bowl / plate
    "bowl_on_plate": BASE_PROFILES["bowl_plate_base"],
    "white_bowl_on_plate": _deep_merge(BASE_PROFILES["bowl_plate_base"], {
        "objects": {
            "white_bowl_1": _free(OBJ["white_bowl_1"], (-0.54, -0.23, Z_KITCHEN_OBJ)),
            "plate_1": _free(OBJ["plate_1"], (-0.31, -0.07, Z_KITCHEN_OBJ)),
        },
    }),
    "middle_bowl_on_plate": _deep_merge(BASE_PROFILES["bowl_plate_base"], {
        "objects": {
            "black_bowl_1": _free(OBJ["black_bowl_1"], (-0.54+0.10, -0.18, Z_KITCHEN_OBJ)),
            "black_bowl_2": _free(OBJ["black_bowl_2"], (-0.43+0.10, -0.18, Z_KITCHEN_OBJ)),
            "black_bowl_3": _free(OBJ["black_bowl_3"], (-0.32+0.10, -0.18, Z_KITCHEN_OBJ)),
            "plate_1": _free(OBJ["plate_1"], (-0.30, -0.04, Z_KITCHEN_OBJ)),
        },
    }),
    "stack_bowls": _deep_merge(BASE_PROFILES["bowl_plate_base"], {
        "objects": {
            "black_bowl_1": _free(OBJ["black_bowl_1"], (-0.54+0.10, -0.18, Z_KITCHEN_OBJ)),
            "black_bowl_2": _free(OBJ["black_bowl_2"], (-0.42+0.10, -0.18, Z_KITCHEN_OBJ)),
            "black_bowl_3": _free(OBJ["black_bowl_3"], (-0.30+0.10, -0.18, Z_KITCHEN_OBJ)),
            "plate_1": _free(OBJ["plate_1"], (-0.30, 0.06, Z_KITCHEN_OBJ)),
        },
    }),

    # Stove
    "stove_put_pan": _deep_merge(BASE_PROFILES["stove_base"], {
        "objects": {
            "moka_pot_1": _free(OBJ["moka_pot_1"], (-0.34, 0.32, Z_KITCHEN_OBJ)),
            "frying_pan": _free(OBJ["frying_pan"], (-0.24, -0.23, Z_KITCHEN_OBJ), quat=Z_ROT_90_QUAT),
        },
    }),
    "stove_put_moka": _deep_merge(BASE_PROFILES["stove_base"], {
        "objects": {
            "moka_pot_1": _free(OBJ["moka_pot_1"], (-0.34, -0.10, Z_KITCHEN_OBJ)),
            "frying_pan": _free(OBJ["frying_pan"], (-0.34, 0.30, Z_KITCHEN_OBJ)),
        },
    }),
    "stove_turn_on": _deep_merge(BASE_PROFILES["stove_base"], {
        "joints": {"stove_knob": _joint(JNT["stove_knob"], STOVE_OFF)},
        "objects": {
            "frying_pan": _free(OBJ["frying_pan"], (-0.32, 0.30, Z_KITCHEN_OBJ)),
            "moka_pot_1": _free(OBJ["moka_pot_1"], (-0.25, -0.10, Z_KITCHEN_OBJ)),
        },
    }),
    "stove_turn_off": _deep_merge(BASE_PROFILES["stove_base"], {
        "joints": {"stove_knob": _joint(JNT["stove_knob"], STOVE_ON)},
    }),

    # Microwave
    "microwave_close": _deep_merge(BASE_PROFILES["microwave_base"], {
        "joints": {"microwave_door": _joint(JNT["microwave_door"], MICROWAVE_OPEN)},
    }),
    "microwave_open": _deep_merge(BASE_PROFILES["microwave_base"], {
        # Rotate the microwave 180 degrees about the vertical axis for this task.
        "bodies": {"microwave": _body(BODY["microwave"], (-0.35, 0.18, Z_FURNITURE), quat=Z_ROT_180_QUAT)},
        "objects": {
            "white_bowl_1": _free(OBJ["white_bowl_1"], (-0.44, -0.18, Z_KITCHEN_OBJ)),
            "plate_1": _free(OBJ["plate_1"], (-0.30, -0.18, Z_KITCHEN_OBJ)),
        },
        "joints": {"microwave_door": _joint(JNT["microwave_door"], MICROWAVE_CLOSED)},
    }),
    "microwave_mug_front": _deep_merge(BASE_PROFILES["microwave_base"], {
        "objects": {
            "white_mug": _free(OBJ["white_mug"], (-0.36, -0.20, Z_KITCHEN_OBJ)),
            "yellow_white_mug": _free(OBJ["yellow_white_mug"], (-0.54, -0.20, Z_KITCHEN_OBJ)),
        },
    }),

    # Study
    "study_book_left_caddy": BASE_PROFILES["study_caddy_base"],
    "study_book_right_caddy": BASE_PROFILES["study_caddy_base"],
    "study_book_front_caddy": BASE_PROFILES["study_caddy_base"],
    "study_book_back_caddy": BASE_PROFILES["study_caddy_base"],
    "study_mug_right_caddy": BASE_PROFILES["study_caddy_base"],
    "study_book_on_shelf": BASE_PROFILES["study_shelf_base"],
    "study_book_top_shelf": BASE_PROFILES["study_shelf_base"],
}

# Backward-compatible aliases used by older notebook diagnostics.
SURENA_SCENE_PROFILES = PLACEMENT_PROFILES
SCENE_LAYOUTS = PLACEMENT_PROFILES


# ---------------------------------------------------------------------------
# Success spec helpers
# ---------------------------------------------------------------------------


def _success_drawer(drawer_key: str, target: str) -> Dict[str, Any]:
    return {"type": "drawer", "joint_names": JNT[drawer_key], "target": target}


def _success_articulation(joint_key: str, target: str,
                          threshold: Optional[float] = None) -> Dict[str, Any]:
    spec = {"type": "articulation", "joint_names": JNT[joint_key], "target": target}
    if threshold is not None:
        spec["threshold"] = float(threshold)
    return spec


def _success_near(obj_key: str, target: Union[str, Sequence[float]], threshold: float = 0.12,
                  z_min: Optional[float] = None) -> Dict[str, Any]:
    return {"type": "near", "object_names": OBJ[obj_key], "target": target, "threshold": threshold, "z_min": z_min}


def _success_in_site(obj_key: str, site_names: Sequence[str], margin: float = 0.005) -> Dict[str, Any]:
    return {"type": "in_site", "object_names": OBJ[obj_key],
            "site_names": list(site_names), "margin": float(margin)}


def _success_on(obj_key: str, target_obj_key: str, threshold: float = 0.10, min_z_offset: float = -0.01, max_z_offset: float = 0.06) -> Dict[str, Any]:
    return {"type": "on", "object_names": OBJ[obj_key], "target_object_names": OBJ[target_obj_key],
            "threshold": threshold, "min_z_offset": min_z_offset, "max_z_offset": max_z_offset}


def _success_stack(top_obj_key: str, bottom_obj_key: str, threshold: float = 0.09) -> Dict[str, Any]:
    return {"type": "stack", "object_names": OBJ[top_obj_key], "target_object_names": OBJ[bottom_obj_key], "threshold": threshold}


def _success_all(*items: Mapping[str, Any]) -> Dict[str, Any]:
    return {"type": "all", "items": [dict(x) for x in items]}


def _success_near_target(obj_label: str, target_name: str, threshold: float = 0.15) -> Dict[str, Any]:
    return {"type": "near_target", "object": obj_label, "target": target_name, "threshold": float(threshold),}


# ---------------------------------------------------------------------------
# Task catalog: about 30 meaningful tasks
# ---------------------------------------------------------------------------

TASK_SPECS: Dict[str, Dict[str, Any]] = {
    # Compact kitchen / study task set only.
    "SurenaOpenTopDrawer": {
        "short_key": "open_top_drawer", "domain": "kitchen", "profile": "cabinet_open_top",
        "bddl": "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.bddl",
        "instruction": "open the top drawer of the cabinet", "mode": "drawer",
        "interaction_bodies": ["white_cabinet"],
        "success": _success_drawer("top_drawer", "open"),
    },
    "SurenaCloseBottomDrawer": {
        "short_key": "close_bottom_drawer", "domain": "kitchen", "profile": "cabinet_close_bottom",
        "bddl": "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.bddl",
        "instruction": "close the bottom drawer of the cabinet", "mode": "drawer",
        "interaction_bodies": ["white_cabinet"],
        "success": _success_drawer("bottom_drawer", "closed"),
    },
    "SurenaCloseTopDrawer": {
        "short_key": "close_top_drawer", "domain": "kitchen", "profile": "cabinet_close_top",
        "bddl": "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.bddl",
        "instruction": "close the top drawer of the cabinet", "mode": "drawer",
        "interaction_bodies": ["white_cabinet"],
        "success": _success_drawer("top_drawer", "closed"),
    },
    "SurenaPutKetchupInTopDrawer": {
        "short_key": "put_ketchup_in_top_drawer", "domain": "kitchen", "profile": "cabinet_put_ketchup_top",
        "bddl": "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet.bddl",
        "instruction": "pick up the ketchup bottle and place it in the top drawer of the cabinet", "mode": "pick_place",
        "success": _success_all(
            _success_in_site("ketchup", ["white_cabinet_1_top_region"]),
            _success_drawer("top_drawer", "open"),
        ),
        "runner_overrides": {
            "object_name_filter": "ketchup_1_main",
            "grasp_assist_body": "ketchup_1_main",
            "grasp_approach_offset": (-0.02, -0.055, 0.02),
            "grasp_approach_step": 0.010,
            "grasp_close_distance": 0.063,
            "sticky_attach_distance": 0.065,
            "grasp_approach_tolerance": 0.002,
            "grip_close_dwell_ticks": 1,
            "grip_release_dwell_ticks": 1,
            "grip_candidate_dwell_ticks": 1,
            "grasp_place_site": "white_cabinet_1_top_region",
            "grasp_transport_clearance": 0.180,
            "grasp_lift_step": 0.030,
            "grasp_transport_step": 0.015,
            "grasp_place_tolerance": 0.015,
            "grasp_lower_stall_epsilon": 0.0005,
            "grasp_lower_stall_ticks": 3,
            "use_libero_success_first": False,
        },
    },
    "SurenaPutBlackBowlInTopDrawer": {
        "short_key": "put_black_bowl_in_top_drawer", "domain": "kitchen", "profile": "cabinet_put_bowl_top",
        "bddl": "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl",
        "instruction": "pick up the black bowl and place it in the top drawer of the cabinet", "mode": "pick_place",
        "success": _success_near("black_bowl_1", (-0.33, 0.13, Z_KITCHEN_OBJ), threshold=0.18),
    },
    "SurenaPutBlackBowlOnPlate": {
        "short_key": "put_black_bowl_on_plate", "domain": "kitchen", "profile": "bowl_on_plate",
        "bddl": "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl",
        "instruction": "pick up the black bowl and put it on the plate", "mode": "pick_place",
        "success": _success_on("black_bowl_1", "plate_1", threshold=0.11),
        "runner_overrides": {
            "object_name_filter": "akita_black_bowl_1_main",
            "grasp_assist_body": "akita_black_bowl_1_main",
            "grasp_close_distance": 0.090,
            "sticky_attach_distance": 0.105,
            "grasp_approach_offset": (-0.02,-0.055,0.07),
            "grasp_approach_tolerance": 0.01,
            "grasp_close_pose_tolerance": 0.02,
            "grasp_approach_stall_epsilon": 0.0005,
            "grasp_approach_stall_ticks": 2,
            "grasp_approach_step": 0.012,
            "grasp_close_step": 0.006,
            "grasp_hover_clearance": 0.10,
            "grasp_hover_xy_tolerance": 0.02,
            "grasp_hover_step": 0.015,
            "grasp_place_body": "plate_1_main",
            "grasp_transport_clearance": 0.10,
            "grasp_place_tolerance": 0.015,
            "success_hold_steps": 2,
            "use_libero_success_first": False,
        },
    },
    "SurenaPutMiddleBlackBowlOnPlate": {
        "short_key": "put_middle_black_bowl_on_plate", "domain": "kitchen", "profile": "middle_bowl_on_plate",
        "bddl": "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate.bddl",
        "instruction": "pick up the middle black bowl and put it on the plate", "mode": "pick_place",
        "success": _success_on("black_bowl_2", "plate_1", threshold=0.11),
    },
    "SurenaStackMiddleBowlOnBackBowl": {
        "short_key": "stack_middle_bowl_on_back_bowl", "domain": "kitchen", "profile": "stack_bowls",
        "bddl": "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl.bddl",
        "instruction": "stack the middle black bowl on the back black bowl", "mode": "pick_place",
        "success": _success_stack("black_bowl_2", "black_bowl_3"),
    },
    "SurenaPutFryingPanOnStove": {
        "short_key": "put_frying_pan_on_stove", "domain": "kitchen", "profile": "stove_put_pan",
        "bddl": "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove.bddl",
        "instruction": "pick up the frying pan and put it on the stove", "mode": "pick_place",
        "success": _success_near("frying_pan", (-0.30, 0.10, Z_KITCHEN_OBJ), threshold=0.16),
    },
    "SurenaPutMokaPotOnStove": {
        "short_key": "put_moka_pot_on_stove", "domain": "kitchen", "profile": "stove_put_moka",
        "bddl": "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove.bddl",
        "instruction": "pick up the moka pot and put it on the stove", "mode": "pick_place",
        "success": _success_near("moka_pot_1", (-0.30, 0.10, Z_KITCHEN_OBJ), threshold=0.16),
    },
    "SurenaTurnOnStove": {
        "short_key": "turn_on_stove", "domain": "kitchen", "profile": "stove_turn_on",
        "bddl": "KITCHEN_SCENE3_turn_on_the_stove.bddl",
        "instruction": "turn on the stove", "mode": "articulation",
        "interaction_bodies": ["stove"],
        "success": _success_articulation("stove_knob", "on"),
    },
    "SurenaCloseMicrowave": {
        "short_key": "close_microwave", "domain": "kitchen", "profile": "microwave_close",
        "bddl": "KITCHEN_SCENE6_close_the_microwave.bddl",
        "instruction": "close the microwave", "mode": "articulation",
        "interaction_bodies": ["microwave_door"],
        "runner_overrides": {
            "pos_scale": 0.040,
            "success_q_closed": -0.040,
            "success_hold_steps": 2,
        },
        "contact_guidance": {
            "type": "hinge_contact",
            "joint_names": JNT["microwave_door"],
            "fixture_body_names": BODY["microwave"],
            "target_q": MICROWAVE_CLOSED,
            "tangent_step": 0.012,
            "contact_preload": 0.002,
            "max_target_lag": 0.015,
            "stall_steps": 5,
            "recovery_retract_steps": 2,
        },
        "success": _success_articulation("microwave_door", "closed", threshold=-0.040),
    },
    "SurenaOpenMicrowave": {
        "short_key": "open_microwave", "domain": "kitchen", "profile": "microwave_open",
        "bddl": "KITCHEN_SCENE7_open_the_microwave.bddl",
        "instruction": "open the microwave", "mode": "articulation",
        "interaction_bodies": ["microwave_door"],
        "success": _success_articulation("microwave_door", "open"),
    },
    "SurenaPutBookInLeftCaddy": {
        "short_key": "put_book_in_left_caddy", "domain": "study", "profile": "study_book_left_caddy",
        "bddl": "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.bddl",
        "instruction": "pick up the book and place it in the left compartment of the caddy", "mode": "pick_place",
        "success": _success_near("book_1", (-0.50, 0.16, Z_STUDY_OBJ), threshold=0.13),
    },
    "SurenaPutBookOnShelf": {
        "short_key": "put_book_on_shelf", "domain": "study", "profile": "study_book_on_shelf",
        "bddl": "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf.bddl",
        "instruction": "pick up the book in the middle and place it on the cabinet shelf", "mode": "pick_place",
        "success": _success_near("book_2", (-0.44, 0.23, Z_STUDY_FURNITURE + 0.06), threshold=0.15),
    },
}



# ---------------------------------------------------------------------------
# Core mixin
# ---------------------------------------------------------------------------

class SurenaSceneTaskMixin:
    DEFAULT_BDDL_NAME: Optional[str] = None
    DEFAULT_PROFILE_NAME: Optional[str] = None
    TASK_INSTRUCTION: str = ""
    TASK_MODE: str = "pick_place"
    SUCCESS_SPEC: Optional[Mapping[str, Any]] = None

    def __init__(
        self,
        bddl_file_name: Optional[str] = None,
        scene_profile_name: Optional[str] = None,
        scene_profile: Optional[Mapping[str, Any]] = None,
        debug_reachability: bool = False,
        warn_missing_scene_elements: bool = False,
        use_libero_success_first: bool = True,
        **kwargs,
    ):
        if bddl_file_name is None:
            if self.DEFAULT_BDDL_NAME is None:
                raise ValueError(f"{self.__class__.__name__} needs DEFAULT_BDDL_NAME or bddl_file_name")
            bddl_file_name = self.DEFAULT_BDDL_NAME
        bddl_file_name = _bddl(bddl_file_name)
        if not os.path.exists(bddl_file_name):
            raise FileNotFoundError(f"BDDL file not found: {bddl_file_name}")

        profile_name = scene_profile_name or self.DEFAULT_PROFILE_NAME
        if scene_profile is None:
            if profile_name is None:
                raise ValueError(f"{self.__class__.__name__} needs DEFAULT_PROFILE_NAME or scene_profile")
            if profile_name not in PLACEMENT_PROFILES:
                raise KeyError(f"Unknown scene_profile_name={profile_name!r}. Known: {sorted(PLACEMENT_PROFILES)}")
            scene_profile = PLACEMENT_PROFILES[profile_name]

        self.bddl_file_name = bddl_file_name
        self.scene_profile_name = profile_name
        self.scene_layout_name = profile_name  # compatibility
        self.scene_profile = copy.deepcopy(dict(scene_profile))
        self.scene_layout = self.scene_profile  # compatibility
        self.debug_reachability = bool(debug_reachability)
        self.warn_missing_scene_elements = bool(warn_missing_scene_elements)
        self.use_libero_success_first = bool(use_libero_success_first)
        self._missing_scene_elements: List[str] = []
        self._placed_free_joint_roots: set[str] = set()
        self._success_count = 0

        # Force robot and cameras.
        kwargs["robots"] = ["SurenaArm"]
        kwargs.setdefault("camera_names", ["agentview", "birdview", "robot0_eye_in_hand"])
        kwargs.setdefault("camera_heights", 224)
        kwargs.setdefault("camera_widths", 224)


        super().__init__(bddl_file_name=bddl_file_name, **kwargs)

    # ------------------------------------------------------------------
    # LIBERO / robosuite hooks
    # ------------------------------------------------------------------

    def _load_model(self):
        super()._load_model()
        worldbody = self.model.worldbody
        for cam in list(worldbody.findall("camera")):
            if cam.get("name") == "agentview":
                worldbody.remove(cam)
        worldbody.insert(0, ET.Element("camera", attrib=dict(AGENTVIEW_CAMERA)))

    def _reset_internal(self):
        super()._reset_internal()
        self._success_count = 0
        self._missing_scene_elements = []
        self._placed_free_joint_roots = set()
        self.apply_surena_profile(self.scene_profile)

    def _assert_problem_name(self):
        # LIBERO checks the original problem name. These are Surena wrappers.
        pass

    def reward(self, action=None):
        return 1.0 if self._check_success() else 0.0

    def _check_success(self):
        # Native LIBERO BDDL success is often more exact than geometry heuristics.
        # But it may raise or be too strict after our reachability placement, so
        # this is used as a first positive signal, not as the only checker.
        if self.use_libero_success_first:
            try:
                if bool(super()._check_success()):
                    return True
            except Exception:
                pass
        try:
            return bool(self._evaluate_success_spec(getattr(self, "SUCCESS_SPEC", None)))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Raw MuJoCo helpers
    # ------------------------------------------------------------------

    def _model_data(self):
        return self.sim.model._model, self.sim.data._data

    def _prefixed(self, bare: str) -> str:
        return bare if bare.startswith(ROBOT_PREFIX) else ROBOT_PREFIX + bare

    def _mark_missing(self, kind: str, name: Union[str, Sequence[str]]):
        self._missing_scene_elements.append(f"{kind}:{name}")

    def _resolve_name(self, candidates: Union[str, Sequence[str]], kind: str) -> Optional[str]:
        model, _ = self._model_data()
        if isinstance(candidates, str):
            candidates = [candidates]
        obj_type = {
            "joint": mujoco.mjtObj.mjOBJ_JOINT,
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "site": mujoco.mjtObj.mjOBJ_SITE,
            "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        }[kind]
        for name in candidates:
            if mujoco.mj_name2id(model, obj_type, name) >= 0:
                return name
        return None

    def _free_joint_pos(self, names: Union[str, Sequence[str]]) -> Optional[np.ndarray]:
        model, data = self._model_data()
        name = self._resolve_name(names, "joint")
        if name is None:
            return None
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = int(model.jnt_qposadr[jid])
        return data.qpos[qadr:qadr + 3].copy()

    def _body_pos(self, names: Union[str, Sequence[str]]) -> Optional[np.ndarray]:
        model, data = self._model_data()
        name = self._resolve_name(names, "body")
        if name is None:
            return None
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[bid].copy()

    def _scalar_joint_qpos(self, names: Union[str, Sequence[str]]) -> Optional[float]:
        model, data = self._model_data()
        name = self._resolve_name(names, "joint")
        if name is None:
            return None
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = int(model.jnt_qposadr[jid])
        return float(data.qpos[qadr])

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def apply_surena_profile(self, profile: Mapping[str, Any]) -> None:
        self._apply_surena_home_pose(profile.get("home_q", DEFAULT_HOME_Q))
        self._place_free_joints(profile.get("objects", {}))
        self._place_bodies(profile.get("bodies", {}))
        self._set_scalar_joints(profile.get("joints", {}))
        self._apply_contact_compatibility(profile.get("contact_compat_bodies", {}))

        model, data = self._model_data()
        mujoco.mj_forward(model, data)
        self.sim.forward()

        if self.debug_reachability:
            self._debug_reachability(profile)
        if self.warn_missing_scene_elements and self._missing_scene_elements:
            print(f"[Surena:{self.__class__.__name__}] missing: {sorted(set(self._missing_scene_elements))}")

    # Backward-compatible older notebook name.
    def apply_surena_layout(self, layout: Mapping[str, Any]) -> None:
        self.apply_surena_profile(layout)

    def _apply_surena_home_pose(self, home_q: Mapping[str, float]) -> None:
        model, data = self._model_data()
        for bare_joint_name in RIGHT_ARM_JOINTS:
            q = float(home_q.get(bare_joint_name, DEFAULT_HOME_Q[bare_joint_name]))
            joint_name = self._prefixed(bare_joint_name)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                self._mark_missing("joint", joint_name)
                continue
            qadr = int(model.jnt_qposadr[jid])
            dadr = int(model.jnt_dofadr[jid])
            data.qpos[qadr] = q
            data.qvel[dadr] = 0.0

            actuator_name = self._prefixed(RIGHT_ARM_ACTUATORS[bare_joint_name])
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if aid >= 0:
                data.ctrl[aid] = q
            else:
                self._mark_missing("actuator", actuator_name)

    def _place_free_joints(self, specs: Mapping[str, Mapping[str, Any]]) -> None:
        model, data = self._model_data()
        for label, spec in specs.items():
            names = spec.get("names", [label])
            name = self._resolve_name(names, "joint")
            if name is None:
                if spec.get("required", False) or "quat" in spec:
                    raise RuntimeError(f"Free joint for '{label}' not found. Tried: {names}")
                self._mark_missing("free_joint", names)
                continue
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                self._mark_missing("not_free_joint", name)
                continue
            qadr = int(model.jnt_qposadr[jid])
            dadr = int(model.jnt_dofadr[jid])

            if "xyz" in spec:
                xyz = np.asarray(spec["xyz"], dtype=float)
                if xyz.shape != (3,):
                    raise ValueError(f"xyz for {label} must have shape (3,), got {xyz!r}")
                data.qpos[qadr:qadr + 3] = xyz
            elif "xy" in spec:
                data.qpos[qadr + 0] = float(spec["xy"][0])
                data.qpos[qadr + 1] = float(spec["xy"][1])
            if "z" in spec:
                data.qpos[qadr + 2] = float(spec["z"])
            if "quat" in spec:
                quat = np.asarray(spec["quat"], dtype=float)
                if quat.shape != (4,):
                    raise ValueError(f"quat for {label} must have shape (4,), got {quat!r}")
                n = float(np.linalg.norm(quat))
                if n > 1e-8:
                    quat = quat / n
                data.qpos[qadr + 3:qadr + 7] = quat
            data.qvel[dadr:dadr + 6] = 0.0

            if name.endswith("_joint0"):
                self._placed_free_joint_roots.add(name[:-7])

    def _place_bodies(self, specs: Mapping[str, Mapping[str, Any]]) -> None:
        model, _ = self._model_data()
        for label, spec in specs.items():
            names = spec.get("names", [label])
            name = self._resolve_name(names, "body")
            if name is None:
                if spec.get("required", False) or "quat" in spec:
                    raise RuntimeError(f"Body for '{label}' not found. Tried: {names}")
                self._mark_missing("body", names)
                continue
            # Avoid double-shifting objects that were already placed by free joint.
            root = name[:-5] if name.endswith("_main") else name
            if root in self._placed_free_joint_roots:
                continue
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            model.body_pos[bid] = np.asarray(spec["pos"], dtype=float)
            if "quat" in spec:
                quat = np.asarray(spec["quat"], dtype=float)
                if quat.shape != (4,):
                    raise ValueError(f"quat for body {label} must have shape (4,), got {quat!r}")
                n = float(np.linalg.norm(quat))
                if n > 1e-8:
                    quat = quat / n
                model.body_quat[bid] = quat

    def _set_scalar_joints(self, specs: Mapping[str, Mapping[str, Any]]) -> None:
        model, data = self._model_data()
        for label, spec in specs.items():
            names = spec.get("names", [label])
            name = self._resolve_name(names, "joint")
            if name is None:
                if spec.get("required", False):
                    raise RuntimeError(f"Required joint for '{label}' not found. Tried: {names}")
                if spec.get("warn_missing", False):
                    self._mark_missing("joint", names)
                continue
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            jtype = int(model.jnt_type[jid])
            if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                self._mark_missing("not_scalar_joint", name)
                continue
            qadr = int(model.jnt_qposadr[jid])
            dadr = int(model.jnt_dofadr[jid])
            data.qpos[qadr] = float(spec["q"])
            data.qvel[dadr] = 0.0

    def _apply_contact_compatibility(self, specs: Mapping[str, int]) -> None:
        """Make selected fixture collision geoms accept Surena contact bits.

        Only geoms whose collision affinity is already nonzero are modified,
        so visual-only geoms remain non-colliding. Descendant bodies are
        included because articulated parts such as the microwave door live
        below the fixture's named root body.
        """
        model, _ = self._model_data()
        for label, contact_bit in specs.items():
            candidates = BODY.get(label, [label])
            root_name = self._resolve_name(candidates, "body")
            if root_name is None:
                self._mark_missing("contact_compat_body", candidates)
                continue
            root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
            bit = int(contact_bit)
            compatible_geoms = 0
            for gid in range(model.ngeom):
                if int(model.geom_conaffinity[gid]) == 0:
                    continue
                bid = int(model.geom_bodyid[gid])
                while bid > 0:
                    if bid == root_id:
                        model.geom_conaffinity[gid] = int(model.geom_conaffinity[gid]) | bit
                        compatible_geoms += 1
                        break
                    bid = int(model.body_parentid[bid])
            if compatible_geoms == 0:
                raise RuntimeError(
                    f"No physical collision geoms found below body {root_name!r}; "
                    f"cannot enable Surena contact bit {bit}"
                )

    # ------------------------------------------------------------------
    # Success evaluation
    # ------------------------------------------------------------------

    def _target_pos(self, target: Union[str, Sequence[float]]) -> Optional[np.ndarray]:
        if isinstance(target, str):
            # Named target in current profile.
            if target in self.scene_profile.get("targets", {}):
                return np.asarray(self.scene_profile["targets"][target], dtype=float)
            # Object/free joint target.
            if target in OBJ:
                p = self._free_joint_pos(OBJ[target])
                if p is not None:
                    return p
            # Body target.
            if target in BODY:
                p = self._body_pos(BODY[target])
                if p is not None:
                    return p
            return None
        arr = np.asarray(target, dtype=float)
        return arr if arr.shape == (3,) else None

    def _evaluate_success_spec(self, spec: Optional[Mapping[str, Any]]) -> bool:
        if not spec:
            return False
        typ = spec.get("type")

        if typ == "all":
            return all(self._evaluate_success_spec(item) for item in spec.get("items", []))
        if typ == "any":
            return any(self._evaluate_success_spec(item) for item in spec.get("items", []))

        if typ == "drawer":
            q = self._scalar_joint_qpos(spec["joint_names"])
            if q is None:
                return False
            target = spec.get("target")
            if target == "open":
                return q <= -0.08
            if target == "closed":
                return q >= -0.04
            return False

        if typ == "articulation":
            q = self._scalar_joint_qpos(spec["joint_names"])
            if q is None:
                return False
            target = spec.get("target")
            if target == "on":
                return q >= 0.45
            if target == "off":
                return q <= 0.25
            if target == "open":
                return q <= float(spec.get("threshold", -0.50))
            if target == "closed":
                return q >= float(spec.get("threshold", -0.25))
            return False

        if typ == "near":
            obj = self._free_joint_pos(spec["object_names"])
            tgt = self._target_pos(spec["target"])
            if obj is None or tgt is None:
                return False
            dxy = float(np.linalg.norm(obj[:2] - tgt[:2]))
            z_min = spec.get("z_min")
            return dxy <= float(spec.get("threshold", 0.12)) and (z_min is None or obj[2] >= float(z_min))

        if typ == "in_site":
            obj = self._free_joint_pos(spec["object_names"])
            site_name = self._resolve_name(spec["site_names"], "site")
            if obj is None or site_name is None:
                return False
            model, data = self._model_data()
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            local = data.site_xmat[sid].reshape(3, 3).T @ (obj - data.site_xpos[sid])
            bounds = np.maximum(model.site_size[sid] - float(spec.get("margin", 0.0)), 0.0)
            return bool(np.all(np.abs(local) <= bounds))

        if typ == "on":
            obj = self._free_joint_pos(spec["object_names"])
            tgt = self._free_joint_pos(spec["target_object_names"])
            if obj is None or tgt is None:
                return False
            dxy = float(np.linalg.norm(obj[:2] - tgt[:2]))
            dz = float(obj[2] - tgt[2])
            min_z_offset = float(spec.get("min_z_offset", -0.01))
            max_z_offset = float(spec.get("max_z_offset", 0.06))
            return (dxy <= float(spec.get("threshold", 0.10)) and min_z_offset <= dz <= max_z_offset)

        if typ == "stack":
            obj = self._free_joint_pos(spec["object_names"])
            tgt = self._free_joint_pos(spec["target_object_names"])
            if obj is None or tgt is None:
                return False
            dxy = float(np.linalg.norm(obj[:2] - tgt[:2]))
            return dxy <= float(spec.get("threshold", 0.09)) and obj[2] >= tgt[2] + 0.025

        if typ == "front_of":
            obj = self._free_joint_pos(spec["object_names"])
            tgt = self._free_joint_pos(spec["target_object_names"])
            if obj is None or tgt is None:
                return False
            # In these table scenes, visually/front in the camera-aligned layout
            # is represented as more negative y.
            return obj[1] <= tgt[1] - float(spec.get("margin", 0.05))

        return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _right_shoulder_pos(self) -> Tuple[Optional[str], Optional[np.ndarray]]:
        model, data = self._model_data()
        for name in ["robot0_r_arm_pitch", "robot0_r_arm_roll", "robot0_base_link"]:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return name, data.xpos[bid].copy()
        return None, None

    def _debug_reachability(self, profile: Optional[Mapping[str, Any]] = None):
        shoulder_name, shoulder_pos = self._right_shoulder_pos()
        if shoulder_pos is None:
            print("[Reachability] Could not find right shoulder body.")
            return
        print(f"[Reachability] shoulder={shoulder_name!r} @ {np.round(shoulder_pos, 4)}")
        profile = profile or self.scene_profile
        for label, spec in profile.get("objects", {}).items():
            p = self._free_joint_pos(spec.get("names", [label]))
            if p is None:
                continue
            d = float(np.linalg.norm(p - shoulder_pos))
            flag = ""
            if d > REACH_HARD_M:
                flag = "  !! OUT OF REACH"
            elif d > REACH_WARN_M:
                flag = "  ~ near limit"
            print(f"  object:{label:24s} pos={np.round(p, 4)} dist={d:.3f} m{flag}")
        for label, spec in profile.get("bodies", {}).items():
            p = self._body_pos(spec.get("names", [label]))
            if p is None:
                continue
            d = float(np.linalg.norm(p - shoulder_pos))
            flag = ""
            if d > REACH_HARD_M:
                flag = "  !! OUT OF REACH"
            elif d > REACH_WARN_M:
                flag = "  ~ near limit"
            print(f"  body:{label:26s} pos={np.round(p, 4)} dist={d:.3f} m{flag}")


# ---------------------------------------------------------------------------
# Domain bases
# ---------------------------------------------------------------------------

class SurenaKitchenSceneTask(SurenaSceneTaskMixin, _KitchenBase):
    pass



class SurenaStudySceneTask(SurenaSceneTaskMixin, _StudyBase):
    pass


# Backward-compatible name for old code; use kitchen by default.
class SurenaManipulationEnv(SurenaKitchenSceneTask):
    pass


DOMAIN_BASE_CLASS: Dict[str, Type[Any]] = {
    "kitchen": SurenaKitchenSceneTask,
    "study": SurenaStudySceneTask,
}


# ---------------------------------------------------------------------------
# Dynamic class creation
# ---------------------------------------------------------------------------


def _make_task_class(class_name: str, spec: Mapping[str, Any]) -> Type[Any]:
    base_cls = DOMAIN_BASE_CLASS[spec["domain"]]
    attrs = {
        "__module__": __name__,
        "__doc__": spec.get("instruction", class_name),
        "DEFAULT_BDDL_NAME": spec["bddl"],
        "DEFAULT_PROFILE_NAME": spec["profile"],
        "TASK_INSTRUCTION": spec.get("instruction", class_name),
        "TASK_MODE": spec.get("mode", "pick_place"),
        "SUCCESS_SPEC": copy.deepcopy(spec.get("success")),
    }
    return type(class_name, (base_cls,), attrs)


for _class_name, _spec in TASK_SPECS.items():
    globals()[_class_name] = _make_task_class(_class_name, _spec)


# Compatibility names expected by older notebooks.
class SurenaKitchenCabinetTopDrawer(globals()["SurenaOpenTopDrawer"]):
    pass


class SurenaKitchenCabinetBottomDrawer(globals()["SurenaCloseBottomDrawer"]):
    pass


class SurenaKitchenTopDrawerClose(globals()["SurenaCloseTopDrawer"]):
    pass


class SurenaKitchenKetchupTopDrawer(globals()["SurenaPutKetchupInTopDrawer"]):
    pass


class SurenaKitchenBowlPlate(globals()["SurenaPutBlackBowlOnPlate"]):
    pass


class SurenaKitchenStovePan(globals()["SurenaPutFryingPanOnStove"]):
    pass


class SurenaKitchenMicrowaveMug(globals()["SurenaCloseMicrowave"]):
    pass





class SurenaLibero90Task(globals()["SurenaOpenTopDrawer"]):
    """Backward-compatible generic wrapper; default is a kitchen drawer task."""
    pass


class SurenaLift(globals()["SurenaCloseTopDrawer"]):
    """Legacy alias for older notebooks."""
    pass


class SurenaPickPlace(globals()["SurenaPutKetchupInTopDrawer"]):
    """Legacy alias for older notebooks."""
    pass


_COMPAT_CLASSES: Tuple[Type[Any], ...] = (
    SurenaKitchenCabinetTopDrawer,
    SurenaKitchenCabinetBottomDrawer,
    SurenaKitchenTopDrawerClose,
    SurenaKitchenKetchupTopDrawer,
    SurenaKitchenBowlPlate,
    SurenaKitchenStovePan,
    SurenaKitchenMicrowaveMug,
    SurenaLibero90Task,
    SurenaLift,
    SurenaPickPlace,
)

SURENA_TASK_CLASSES: Tuple[Type[Any], ...] = tuple([globals()[name] for name in TASK_SPECS.keys()] + list(_COMPAT_CLASSES))
SURENA_CLASSES: Dict[str, Type[Any]] = {cls.__name__: cls for cls in SURENA_TASK_CLASSES}


# ---------------------------------------------------------------------------
# Notebook-facing presets
# ---------------------------------------------------------------------------

MODE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "drawer": {
        "max_vla_steps": 75,
        "pos_scale": 0.035,
        "rot_scale": 0.004,
        "max_rot_delta": 0.010,
        "success_delta": 0.15,
        "sticky": False,
    },
    "articulation": {
        "max_vla_steps": 90,
        "pos_scale": 0.030,
        "rot_scale": 0.004,
        "max_rot_delta": 0.015,
        "success_delta": 0.15,
        "sticky": False,
    },
    "pick_place": {
        "max_vla_steps": 125,
        "pos_scale": 0.035,
        "rot_scale": 0.005,
        "max_rot_delta": 0.020,
        "sticky": True,
        "sticky_attach_distance": 0.05,
    },
    "combo": {
        "max_vla_steps": 170,
        "pos_scale": 0.035,
        "rot_scale": 0.005,
        "max_rot_delta": 0.020,
        "success_delta": 0.15,
        "sticky": True,
        "sticky_attach_distance": 0.05,
    },
}


def _preset_from_spec(class_name: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    mode = spec.get("mode", "pick_place")
    cfg = dict(MODE_DEFAULTS.get(mode, MODE_DEFAULTS["pick_place"]))
    interaction_body_candidates = []
    for label in spec.get("interaction_bodies", ()):
        interaction_body_candidates.extend(BODY.get(label, ()))
    cfg.update({
        "env_class": class_name,
        "env_class_name": class_name,
        "bddl_file_name": _bddl(spec["bddl"]),
        "scene_profile_name": spec["profile"],
        "instruction": spec["instruction"],
        "mode": mode,
        "bddl_exists": os.path.exists(_bddl(spec["bddl"])),
        "interaction_body_candidates": interaction_body_candidates,
    })
    if "contact_guidance" in spec:
        cfg["contact_guidance"] = copy.deepcopy(spec["contact_guidance"])
    cfg.update(copy.deepcopy(spec.get("runner_overrides", {})))
    if "drawer" in spec["instruction"]:
        if "top" in spec["instruction"]:
            cfg["joint_hint"] = "top"
        elif "bottom" in spec["instruction"]:
            cfg["joint_hint"] = "bottom"
    return cfg


SURENA_TASK_PRESETS: Dict[str, Dict[str, Any]] = {}
SURENA_PRESET_ALIASES: Dict[str, str] = {}

for _class_name, _spec in TASK_SPECS.items():
    _short = _spec["short_key"]
    _cfg = _preset_from_spec(_class_name, _spec)
    SURENA_TASK_PRESETS[_short] = _cfg
    # Long BDDL-style alias for compatibility with the previous generated file.
    _long = os.path.splitext(os.path.basename(_spec["bddl"]))[0].lower()
    SURENA_PRESET_ALIASES[_long] = _short
    SURENA_PRESET_ALIASES[_class_name.lower()] = _short

# Legacy short aliases.
SURENA_PRESET_ALIASES.update({
    "surena_lift": "close_top_drawer",
    "surena_pickplace": "put_ketchup_in_top_drawer",
    "put_ketchup_top_drawer": "put_ketchup_in_top_drawer",
})

EPISODE_PRESETS = SURENA_TASK_PRESETS


def resolve_preset_name(name: str) -> str:
    key = str(name).strip()
    if key in SURENA_TASK_PRESETS:
        return key
    low = key.lower()
    if low in SURENA_TASK_PRESETS:
        return low
    if low in SURENA_PRESET_ALIASES:
        return SURENA_PRESET_ALIASES[low]
    raise KeyError(
        f"Unknown Surena preset {name!r}. Available presets: {sorted(SURENA_TASK_PRESETS)}. "
        f"Known aliases include: {sorted(SURENA_PRESET_ALIASES)[:20]} ..."
    )


def get_preset(name: str) -> Dict[str, Any]:
    key = resolve_preset_name(name)
    p = dict(SURENA_TASK_PRESETS[key])
    p["preset_name"] = key
    p["env_class"] = SURENA_CLASSES[p["env_class_name"]]
    return p


def list_presets(verbose: bool = True, include_aliases: bool = False) -> List[str]:
    names = sorted(SURENA_TASK_PRESETS.keys())
    if verbose:
        print(f"{len(names)} Surena task presets:")
        for i, name in enumerate(names, 1):
            p = SURENA_TASK_PRESETS[name]
            flag = "" if p["bddl_exists"] else "  [MISSING BDDL]"
            print(f"  {i:2d}. {name:42s} mode={p['mode']:12s} class={p['env_class_name']}{flag}")
            print(f"      instruction: {p['instruction']!r}")
        if include_aliases:
            print(f"\n{len(SURENA_PRESET_ALIASES)} aliases:")
            for a, target in sorted(SURENA_PRESET_ALIASES.items()):
                print(f"  {a:75s} -> {target}")
    return names


def make_env(preset_name: str, **env_kwargs):
    """
    Instantiate the environment for a preset and return (env, instruction).

    Example:
        env, instruction = sm.make_env(
            "put_ketchup_in_top_drawer",
            controller_configs=joint_pos_config,
            has_offscreen_renderer=True,
            use_camera_obs=True,
        )
    """
    preset = get_preset(preset_name)
    EnvClass = preset["env_class"]

    kwargs = dict(env_kwargs)
    kwargs.setdefault("bddl_file_name", preset["bddl_file_name"])
    kwargs.setdefault("scene_profile_name", preset["scene_profile_name"])

    env = EnvClass(**kwargs)
    return env, preset["instruction"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_surena_tasks(task_mapping: Optional[MutableMapping[str, Type[Any]]] = None):
    """Register canonical SURENA task classes and all supported aliases.

    LIBERO's ``TASK_MAPPING`` is a string-to-class registry. Historically, the
    project used several spellings for the same environments, including class
    names (``SurenaLift``), lowercase class names (``surenalift``), and
    underscore aliases (``surena_lift``). The notebook preset resolver also
    exposes BDDL-style and short aliases. Registering all of them here keeps
    direct LIBERO construction and notebook preset lookup consistent.

    The function is idempotent: assigning the same keys again simply refreshes
    them to the canonical classes from this package.
    """
    if task_mapping is None:
        from libero.libero.envs.bddl_base_domain import TASK_MAPPING as task_mapping

    # Canonical classes and lowercase class-name spellings.
    for cls in SURENA_TASK_CLASSES:
        task_mapping[cls.__name__] = cls
        task_mapping[cls.__name__.lower()] = cls

    # Every notebook/preset alias should also be a valid LIBERO task key.
    for alias, preset_name in SURENA_PRESET_ALIASES.items():
        preset = SURENA_TASK_PRESETS[preset_name]
        env_class = SURENA_CLASSES[preset["env_class_name"]]
        task_mapping[alias] = env_class

    # Preserve the compatibility wrapper classes for their historical keys.
    # These explicit assignments intentionally override the generic preset
    # aliases above where both forms exist.
    task_mapping["surena_lift"] = SurenaLift
    task_mapping["surena_pickplace"] = SurenaPickPlace

    return task_mapping


try:
    register_surena_tasks()
except Exception as exc:
    print(f"[SurenaManipulation] Auto-registration skipped: {exc}")


__all__ = [
    "LIBERO_PATH",
    "BDDL_DIR",
    "BASE_PROFILES",
    "PLACEMENT_PROFILES",
    "SURENA_SCENE_PROFILES",
    "SCENE_LAYOUTS",
    "TASK_SPECS",
    "EPISODE_PRESETS",
    "SURENA_TASK_PRESETS",
    "SURENA_PRESET_ALIASES",
    "MODE_DEFAULTS",
    "SurenaSceneTaskMixin",
    "SurenaKitchenSceneTask",
    "SurenaStudySceneTask",
    "SurenaManipulationEnv",
    "SurenaLibero90Task",
    "SurenaLift",
    "SurenaPickPlace",
    "SURENA_TASK_CLASSES",
    "SURENA_CLASSES",
    "resolve_preset_name",
    "get_preset",
    "list_presets",
    "make_env",
    "register_surena_tasks",
] + list(TASK_SPECS.keys()) + [cls.__name__ for cls in _COMPAT_CLASSES]

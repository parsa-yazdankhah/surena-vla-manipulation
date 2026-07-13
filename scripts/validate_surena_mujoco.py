#!/usr/bin/env python3
"""Headless structural validation of the package-owned SURENA MJCF."""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from surena_vla.control import (
    ARM_ACTUATOR_NAMES_BARE,
    ARM_JOINT_NAMES_BARE,
    EEF_SITE_BARE,
    HOME_QPOS,
    GazeboStyleController,
)
from surena_vla.paths import SURENA_ARM_XML, validate_asset_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    validate_asset_layout()
    model = mujoco.MjModel.from_xml_path(str(SURENA_ARM_XML))
    data = mujoco.MjData(model)

    missing = []
    for name in ARM_JOINT_NAMES_BARE:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            missing.append(name)
    for name in ARM_ACTUATOR_NAMES_BARE:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0:
            missing.append(name)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE_BARE) < 0:
        missing.append(EEF_SITE_BARE)
    if missing:
        raise RuntimeError(f"MJCF is missing required names: {missing}")

    controller = GazeboStyleController(model, data, prefix="", apply_home=True)
    for _ in range(args.steps):
        controller.control_callback()
        mujoco.mj_step(model, data)

    error = np.max(np.abs(controller.get_arm_qpos() - HOME_QPOS))
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("Simulation produced non-finite state")
    if error > 0.20:
        raise RuntimeError(
            f"Arm failed to hold HOME pose: maximum error is {error:.6f} rad"
        )

    print(f"MJCF: {SURENA_ARM_XML}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"max home-pose error after {args.steps} steps: {error:.6f} rad")
    print("PASS: standalone MuJoCo model and controller are structurally valid")


if __name__ == "__main__":
    main()

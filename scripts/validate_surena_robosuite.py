#!/usr/bin/env python3
"""Validate registration and MJCF loading through robosuite."""

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
import robosuite.robots as robot_runtime

from surena_vla.integrations.robosuite import register_surena_robot


def main() -> None:
    SurenaArm = register_surena_robot()

    assert REGISTERED_ROBOTS.get("SurenaArm") is SurenaArm
    assert "SurenaArm" in robot_runtime.ROBOT_CLASS_MAPPING

    robot_model = SurenaArm(idn=0)
    compiled = robot_model.get_model()

    print(f"robot model: {robot_model.__class__.__name__}")
    print(f"runtime class: {robot_runtime.ROBOT_CLASS_MAPPING['SurenaArm'].__name__}")
    print(f"compiled nq={compiled.nq}, nv={compiled.nv}, nu={compiled.nu}")
    print("PASS: robosuite registration works without copying package files")


if __name__ == "__main__":
    main()

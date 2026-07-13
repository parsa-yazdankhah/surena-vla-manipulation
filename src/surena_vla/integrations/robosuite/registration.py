"""Runtime registration without copying files into the robosuite repository."""

from __future__ import annotations


def register_surena_robot():
    """Register ``SurenaArm`` for LIBERO's robosuite 1.4.x runtime.

    Importing the model class registers it in robosuite's model registry through
    ``RobotModelMeta``. robosuite 1.4.x separately maintains a runtime mapping,
    so that mapping is updated explicitly here.
    """
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    import robosuite.robots as robot_runtime

    from .surena_arm import SurenaArm

    runtime_class = getattr(robot_runtime, "SingleArm", None)
    if runtime_class is None:
        raise RuntimeError(
            "The installed robosuite does not expose SingleArm. This integration "
            "targets the LIBERO-compatible robosuite 1.4.x API; use the version "
            "required by your LIBERO fork."
        )

    robot_runtime.ROBOT_CLASS_MAPPING["SurenaArm"] = runtime_class
    REGISTERED_ROBOTS["SurenaArm"] = SurenaArm
    return SurenaArm

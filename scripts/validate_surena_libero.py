#!/usr/bin/env python3
"""Create, reset, and step a headless SURENA LIBERO environment."""

from __future__ import annotations

import argparse

import numpy as np

from surena_vla.integrations import register_all
register_all()

from surena_vla.integrations.libero import SurenaLift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    from libero.libero.envs.bddl_base_domain import TASK_MAPPING

    expected_task_entries = {
        "SurenaLift": SurenaLift,
        "surenalift": SurenaLift,
        "surena_lift": SurenaLift,
    }
    missing_aliases = sorted(
        key for key in expected_task_entries if key not in TASK_MAPPING
    )
    if missing_aliases:
        raise RuntimeError(f"Missing LIBERO task aliases: {missing_aliases}")

    wrong_aliases = {
        key: TASK_MAPPING[key]
        for key, expected_class in expected_task_entries.items()
        if TASK_MAPPING[key] is not expected_class
    }
    if wrong_aliases:
        details = ", ".join(
            f"{key} -> {value.__module__}.{value.__name__}"
            for key, value in wrong_aliases.items()
        )
        raise RuntimeError(f"Incorrect LIBERO task alias targets: {details}")

    print(
        "task aliases: "
        + ", ".join(sorted(expected_task_entries))
        + " [OK]"
    )

    env = SurenaLift(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=224,
        camera_widths=224,
        control_freq=20,
        horizon=max(args.steps + 1, 20),
        reward_shaping=False,
    )

    try:
        observation = env.reset()
        required_observations = {
            "agentview_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
        }
        missing = sorted(required_observations.difference(observation))
        if missing:
            raise RuntimeError(f"Missing observations: {missing}")

        action_low, _ = env.action_spec
        action = np.zeros_like(action_low, dtype=float)
        for _ in range(args.steps):
            observation, reward, done, info = env.step(action)

        print(f"action_dim={action.size}")
        print(f"agentview_image={observation['agentview_image'].shape}")
        print(f"eef_pos={np.asarray(observation['robot0_eef_pos']).round(4)}")
        print("PASS: LIBERO reset and rollout completed")
    finally:
        env.close()


if __name__ == "__main__":
    main()

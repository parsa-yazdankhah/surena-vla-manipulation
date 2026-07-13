"""Small standalone MuJoCo runner using the shared controller package."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mink
import mujoco
import mujoco.viewer
import numpy as np

from surena_vla.control import DEG, GazeboStyleController, SurenaIK
from surena_vla.paths import SURENA_ARM_XML

SIM_FREQUENCY = 200
CONTROL_FREQUENCY = 20
STEPS_PER_CONTROL = SIM_FREQUENCY // CONTROL_FREQUENCY


class SurenaSimulation:
    def __init__(self, xml_path: str | Path = SURENA_ARM_XML):
        self.xml_path = Path(xml_path).expanduser().resolve()
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"MJCF not found: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 1.0 / SIM_FREQUENCY
        self.bridge = GazeboStyleController(
            self.model,
            self.data,
            prefix="",
            apply_home=True,
        )
        self.ik = SurenaIK(self.model, self.data, self.bridge, prefix="")

    def hold(self) -> None:
        self._viewer_loop(self.bridge.control_callback)

    def move_to(self, target: np.ndarray) -> None:
        if target.shape != (6,):
            raise ValueError("Target must contain x,y,z,roll,pitch,yaw")
        orientation = mink.SO3.from_rpy_radians(*target[3:])
        self.ik.move_eef_to(target[:3], orientation, verbose=True)
        self._viewer_loop(self.bridge.control_callback)

    def demo_ik(self) -> None:
        target = np.array(
            [0.221, -0.37, 0.80, 56 * DEG, -23 * DEG, -114 * DEG],
            dtype=float,
        )
        self.move_to(target)

    def _viewer_loop(self, callback) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step = 0
            while viewer.is_running():
                start = time.perf_counter()
                mujoco.mj_step(self.model, self.data)
                step += 1
                if step % STEPS_PER_CONTROL == 0:
                    callback()
                viewer.sync()
                remaining = self.model.opt.timestep - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)


def parse_target(value: str) -> np.ndarray:
    values = np.fromstring(value.strip("[]"), sep=",")
    if values.shape != (6,):
        raise argparse.ArgumentTypeError(
            "Target must be six comma-separated values: x,y,z,roll,pitch,yaw"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone SURENA MuJoCo model")
    parser.add_argument("--xml", type=Path, default=SURENA_ARM_XML)
    parser.add_argument("--mode", choices=("hold", "ik"), default="hold")
    parser.add_argument("--target", type=parse_target)
    args = parser.parse_args()

    simulation = SurenaSimulation(args.xml)
    if args.target is not None:
        simulation.move_to(args.target)
    elif args.mode == "ik":
        simulation.demo_ik()
    else:
        simulation.hold()

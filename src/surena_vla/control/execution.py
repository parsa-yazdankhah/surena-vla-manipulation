"""Raw stepping, rendering, diagnostics, animation, and video helpers."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

from surena_vla.paths import VIDEOS_ROOT

def get_loop_params(env, verbose: bool = True):
    """
    Reads MuJoCo dt and LIBERO control_freq, then computes how many raw physics
    steps correspond to one controller tick.
    """
    mj_model = env.sim.model._model
    dt = float(mj_model.opt.timestep)
    sim_freq = 1.0 / dt
    ctrl_freq = float(env.control_freq)
    steps_per_ctrl = max(1, int(round(sim_freq / ctrl_freq)))
    if verbose:
        print(f"MuJoCo dt:           {dt:.4f} s")
        print(f"Derived SIM_FREQ:    {sim_freq:.1f} Hz")
        print(f"LIBERO CTRL_FREQ:    {ctrl_freq:.1f} Hz")
        print(f"STEPS_PER_CTRL:      {steps_per_ctrl}")
    return dt, sim_freq, ctrl_freq, steps_per_ctrl


def render_frame(env, camera_name: str = "agentview", width: int = 224,
                 height: int = 224, flip_vertical: bool = True,
                 flip_horizontal: bool = False):
    """
    Renders a fresh RGB frame from env.sim.render(). This avoids stale observation
    images when stepping MuJoCo directly.
    """
    env.sim.forward()
    frame = env.sim.render(camera_name=camera_name, width=width, height=height, depth=False)
    if isinstance(frame, tuple):
        frame = frame[0]
    frame = np.asarray(frame).copy()
    if flip_vertical:
        frame = np.flipud(frame)
    if flip_horizontal:
        frame = np.fliplr(frame)
    return frame[::-1, ::-1]


def settle_with_env_step(env, n_steps: int = 50):
    """
    Performs short normal LIBERO settling using env.step(zero). Use only before
    custom IK/VLA control. Do not use it after commanding Surena through ctrl.bridge.
    """
    obs = None
    zero = np.zeros(env.action_dim)
    for _ in range(n_steps):
        obs, _, done, _ = env.step(zero)
        if done:
            break
    return obs


def reset_settle_rebind(env, ctrl=None, settle_steps: int = 50,
                        prefix: str = "robot0_"):
    """
    Resets the env, optionally settles it, then creates/rebinds SurenaArmController
    to the current MuJoCo model/data after reset.
    """
    obs = env.reset()
    mj_model = env.sim.model._model
    mj_data = env.sim.data._data

    if ctrl is None:
        from .arm_controller import SurenaArmController
        ctrl = SurenaArmController(mj_model, mj_data, prefix=prefix, apply_home=False)
    else:
        ctrl.rebind(env)

    for _ in range(settle_steps):
        ctrl.bridge.control_callback()
        if ctrl.sticky is not None:
            ctrl.sticky_enforce()
        mujoco.mj_step(mj_model, mj_data)

    ctrl.reset_vla()
    env.sim.forward()
    return obs, ctrl


def smoothstep(a):
    """
    Smooth interpolation scalar in [0, 1]. Used to avoid abrupt joint jumps.
    """
    a = float(np.clip(a, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


def execute_joint_target(env, ctrl, q_goal, ctrl_ticks: int = 80,
                         steps_per_ctrl: int | None = None,
                         record: bool = True,
                         render_stride: int = 5,
                         camera_name: str = "agentview",
                         gripper_action: float | None = None,
                         hold_ticks: int = 10):
    """
    Smoothly interpolates current arm joints to q_goal, sends targets through
    ctrl.bridge, raw-steps MuJoCo, and optionally records frames/EEF positions.
    """
    return ctrl.execute_joint_target(
        q_goal=q_goal,
        env=env,
        ctrl_ticks=ctrl_ticks,
        steps_per_ctrl=steps_per_ctrl,
        record=record,
        render_stride=render_stride,
        camera_name=camera_name,
        gripper_action=gripper_action,
        hold_ticks=hold_ticks,
    )


def raw_step_with_bridge(env, ctrl, n_steps: int, record: bool = False,
                         render_stride: int = 5,
                         camera_name: str = "agentview",
                         gripper_action: float | None = None):
    """
    Advances MuJoCo directly while repeatedly applying the latest bridge command.
    Use this after IK/VLA has written custom actuator targets.
    """
    mj_model = env.sim.model._model
    mj_data = env.sim.data._data
    _, _, _, steps_per_ctrl = get_loop_params(env, verbose=False)

    frames = []
    eef_log = []
    obj_log = []

    for k in range(n_steps):
        if k % steps_per_ctrl == 0:
            ctrl.bridge.control_callback()
        ctrl._step_once(env=env, gripper_action=gripper_action)
        if record and (k % render_stride == 0):
            frames.append(render_frame(env, camera_name=camera_name))
            eef_log.append(ctrl.get_eef_pose()[0].copy())
            obj_log.append(ctrl.get_sticky_object_pos(default_nan=True))

    env.sim.forward()
    if gripper_action is None:
        return frames, np.asarray(eef_log)
    return frames, np.asarray(eef_log), np.asarray(obj_log)


def make_rgb_animation(frames, fps: int = 20, title: str | None = None,
                       max_frames: int = 500):
    """
    Converts recorded RGB frames into an inline notebook animation.
    """
    if len(frames) == 0:
        raise ValueError("No frames to animate.")
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idx]

    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    im = ax.imshow(frames[0])
    txt = ax.text(5, 18, "", color="white", fontsize=9,
                  bbox=dict(facecolor="black", alpha=0.45))

    def update(i):
        im.set_data(frames[i])
        txt.set_text(f"{title + ' | ' if title else ''}frame {i+1}/{len(frames)}")
        return [im, txt]

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=int(1000 / fps), blit=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_eef_log(eef_log, target_pos=None):
    """
    Plots EEF x/y/z trajectory and optional target lines for debugging tracking.
    """
    import matplotlib.pyplot as plt

    if eef_log is None or len(eef_log) == 0:
        print("No EEF log.")
        return
    eef_log = np.asarray(eef_log)
    t = np.arange(len(eef_log))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    for i, name in enumerate(["x", "y", "z"]):
        axes[i].plot(t, eef_log[:, i], label=f"EEF {name}")
        if target_pos is not None:
            axes[i].axhline(target_pos[i], linestyle="--", linewidth=1, label="target")
        axes[i].grid(True)
        axes[i].legend()
    plt.tight_layout()
    plt.show()


def save_episode_video(frames, out_path=None, fps=60):
    """Save MP4 video file of the frames"""
    from pathlib import Path
    import numpy as np
    import imageio.v2 as imageio
    import time

    if out_path is None:
        out_dir = VIDEOS_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"surena_vla_episode_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for f in frames:
            f = np.asarray(f)
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            writer.append_data(f)

    print(f"Saved video: {out_path}")
    print(f"Frames: {len(frames)} | fps: {fps} | duration: {len(frames) / fps:.1f} s")
    return out_path

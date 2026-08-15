"""One compact HDF5 file per episode rollout.

Design summary (see the accompanying design discussion for the full
rationale):

- One ``.h5`` file per episode, fully self-contained: session/hardware
  metadata, episode config snapshots, per-VLA-step telemetry, per-physics-
  tick telemetry, and two JPEG-encoded frame streams all live in the same
  file. No cross-file joins needed to reconstruct one rollout.
- Two frame streams, both driven by frames the pipeline renders anyway
  (never re-rendered by the logger):
    * ``keyframes``      — exactly the frame handed to ``predict_vla_action``,
                            one per VLA decision.
    * ``video_frames``   — the existing render_stride-cadence frames that
                            already flow out of ``execute_joint_target`` /
                            ``raw_step_with_bridge``, so a full video can be
                            rebuilt (with any overlay/editing choice) later
                            without re-running the episode.
- A lightweight ``manifest.csv`` is appended to on every ``save()`` as a
  fast cross-episode index. It is a convenience index only — each ``.h5``
  file remains the single source of truth for its own episode.
- Nothing here talks to MuJoCo/robosuite/torch directly except through the
  passed-in ``ctrl``/``env`` objects, and only inside ``on_tick`` /
  ``log_vla_step`` — so this module stays importable and unit-testable on
  its own.
"""

from __future__ import annotations

import csv
import json
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .codec import encode_jpeg, quat_wxyz_to_euler

_STR_DTYPE = None  # resolved lazily so importing this module never requires h5py


def _str_dtype():
    global _STR_DTYPE
    if _STR_DTYPE is None:
        import h5py
        _STR_DTYPE = h5py.string_dtype(encoding="utf-8")
    return _STR_DTYPE


def _json_safe(obj: Any) -> str:
    """json.dumps that never raises on stray non-serializable objects
    (e.g. the ``env_class`` entry inside the notebook's preset ``cfg``)."""
    return json.dumps(obj, default=str)


def _object_array(items) -> np.ndarray:
    """np.array(list_of_variable_stuff, dtype=object) silently collapses into
    a regular N-D array when every element happens to have equal length
    (e.g. identical-size JPEG frames, or same-length strings) — the classic
    numpy ragged-array gotcha. Building the object array by explicit
    element assignment avoids that entirely, which matters here because
    every keyframe/string dataset would otherwise risk being written as
    raw scalars instead of the vlen bytes/strings h5py expects."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _h5_attr_safe(value: Any):
    """h5py attrs can't hold None or arbitrary nested structures, and
    shouldn't be handed bare numpy scalars (see ``_set_attr`` for why)."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return _json_safe(value)
    if isinstance(value, bool):
        return int(value)
    return value


def _set_attr(attrs, key: str, value):
    """Write one HDF5 attribute without ever letting h5py *auto-infer* a
    numpy dtype for it.

    ``attrs[key] = "some string"`` asks h5py to infer a dtype via
    ``numpy.asarray(value)``, which produces a fixed-width numpy unicode
    dtype (``<U11``, etc.) rather than h5py's own vlen-string marker on
    some h5py/numpy/HDF5-C-library version combinations — that's exactly
    what raises ``TypeError: No conversion path for dtype: dtype('<U..')``.
    Requesting ``h5py.string_dtype()`` explicitly sidesteps the inference
    step entirely, so this is version-safe."""
    import h5py

    safe = _h5_attr_safe(value)
    if isinstance(safe, str):
        attrs.create(key, data=safe, dtype=h5py.string_dtype(encoding="utf-8"))
    else:
        attrs[key] = safe


class EpisodeLogger:
    """Accumulates one episode's telemetry in memory; writes it in one shot.

    Buffering in memory (rather than incremental HDF5 writes) keeps the
    write path simple and fast — a full episode's scalar traces are a few
    hundred KB and the frame streams are the only sizeable part, both well
    within what fits comfortably in RAM for a single rollout.

    Usage (see the notebook integration for the exact call sites)::

        logger = EpisodeLogger(output_root, preset_name="close_top_drawer",
                                session_meta=SESSION_META)
        logger.set_loop_params(dt, sim_freq, ctrl_freq)
        logger.set_ik_config(ctrl.ik.config)
        logger.set_sticky_config(ctrl.sticky.config() if ctrl.sticky else None)
        logger.set_episode_cfg(cfg)
        ...
        logger.begin_vla_step(step)
        ik_info = ctrl.apply_vla_action(exec_action, ...)
        frames, eef_log = execute_joint_target(env, ctrl, ik_info["q_goal"],
                                                on_tick=logger.on_tick, ...)
        logger.log_video_frames(frames, vla_step_idx=step)
        logger.log_vla_step(keyframe=frame, raw_action=raw_action,
                             exec_action=exec_action, ik_info=ik_info, ...)
        ...
        logger.finalize(success=..., stop_reason=...)
        path = logger.save()
    """

    def __init__(self, output_root, preset_name: str, session_meta: dict,
                 jpeg_quality: int = 90, hdf5_compression: str = "gzip",
                 hdf5_compression_level: int = 4):
        self.output_root = Path(output_root)
        self.preset_name = preset_name
        self.session_meta = dict(session_meta)
        self.jpeg_quality = int(jpeg_quality)
        self.hdf5_compression = hdf5_compression
        self.hdf5_compression_level = int(hdf5_compression_level)

        self.episode_id = uuid.uuid4().hex[:12]
        self.t_start = time.time()
        self.t_start_iso = datetime.now(timezone.utc).isoformat()
        self.t_end: float | None = None

        # Config snapshots (set once, near the start of the episode).
        self.physics_dt: float | None = None
        self.sim_freq_hz: float | None = None
        self.ctrl_freq_hz: float | None = None
        self.ik_config_json: str | None = None
        self.sticky_config_json: str | None = None
        self.episode_cfg_json: str | None = None
        self.extra_meta: dict = {}

        # Per-VLA-step buffers (dict-of-lists; converted to arrays on save).
        self._step_rows: list[dict] = []
        self._keyframes: list[np.ndarray] = []

        # Per-physics-tick buffers.
        self._tick_rows: list[dict] = []
        self._current_vla_step_idx = -1

        # Video-frame stream (render_stride cadence, reused from execution.py).
        self._video_frames: list[np.ndarray] = []
        self._video_frame_vla_step_idx: list[int] = []
        self._video_frame_order: list[int] = []

        # Running aggregates, updated incrementally so finalize() is O(1).
        self._ik_stage_counts: Counter = Counter()
        self._hard_collision_ticks = 0
        self._warn_collision_ticks = 0
        self._ik_pos_err: list[float] = []
        self._ik_rot_err: list[float] = []
        self._tracking_pos_err: list[float] = []
        self._tracking_rot_err: list[float] = []
        self._prev_sticky_state: str | None = None
        self.grasp_attempts = 0
        self.grasp_attaches = 0
        self.grasp_releases = 0

        self.success: bool | None = None
        self.stop_reason: str | None = None
        self.completed = False

    # ------------------------------------------------------------------
    # One-time config snapshots
    # ------------------------------------------------------------------

    def set_loop_params(self, dt: float, sim_freq: float, ctrl_freq: float):
        self.physics_dt = float(dt)
        self.sim_freq_hz = float(sim_freq)
        self.ctrl_freq_hz = float(ctrl_freq)

    def set_ik_config(self, config):
        import dataclasses
        self.ik_config_json = _json_safe(dataclasses.asdict(config))

    def set_sticky_config(self, config_dict: dict | None):
        self.sticky_config_json = _json_safe(config_dict) if config_dict else None

    def set_episode_cfg(self, cfg: dict):
        self.episode_cfg_json = _json_safe(dict(cfg))

    def set_extra_meta(self, **kwargs):
        """Anything else worth attaching to episode attrs (instruction,
        bddl_file_name, preset metadata, git info override, etc.)."""
        self.extra_meta.update(kwargs)

    # ------------------------------------------------------------------
    # Per-physics-tick capture — pass ``logger.on_tick`` as the ``on_tick``
    # callback threaded through execute_joint_target / raw_step_with_bridge.
    # Never renders a frame; scalar reads only.
    # ------------------------------------------------------------------

    def begin_vla_step(self, step_idx: int):
        self._current_vla_step_idx = int(step_idx)

    def on_tick(self, ctrl, env):
        row = _capture_tick_row(ctrl, self._current_vla_step_idx,
                                 tick_idx=len(self._tick_rows))
        self._tick_rows.append(row)
        if row["collision_hard"]:
            self._hard_collision_ticks += 1
        if row["collision_min_dist"] is not None and \
                row["collision_min_dist"] < self._warn_collision_threshold(ctrl):
            self._warn_collision_ticks += 1

    @staticmethod
    def _warn_collision_threshold(ctrl) -> float:
        try:
            return float(ctrl.ik.config.collision_warning_distance)
        except Exception:
            return 0.02

    # ------------------------------------------------------------------
    # Per-VLA-step capture
    # ------------------------------------------------------------------

    def log_video_frames(self, frames, vla_step_idx: int):
        """Absorb already-rendered render_stride-cadence frames (no re-render)."""
        for frame in frames:
            self._video_frames.append(encode_jpeg(frame, quality=self.jpeg_quality))
            self._video_frame_vla_step_idx.append(int(vla_step_idx))
            self._video_frame_order.append(len(self._video_frame_order))

    def log_vla_step(self, *, keyframe: np.ndarray, raw_action, exec_action,
                      ik_info: dict, eef_pos_before, eef_quat_before,
                      eef_pos_after, eef_quat_after, vla_inference_ms: float,
                      execution_wall_ms: float, sticky_status: dict | None,
                      success: bool, native_success: bool, scalar_success: bool,
                      commanded_target_pos=None, commanded_target_quat=None):
        step_idx = self._current_vla_step_idx
        rpy_before = quat_wxyz_to_euler(eef_quat_before)
        rpy_after = quat_wxyz_to_euler(eef_quat_after)

        tracking_pos_err = tracking_rot_err = None
        if commanded_target_pos is not None:
            tracking_pos_err = float(np.linalg.norm(
                np.asarray(eef_pos_after) - np.asarray(commanded_target_pos)))
        if commanded_target_quat is not None:
            rpy_target = quat_wxyz_to_euler(commanded_target_quat)
            tracking_rot_err = float(np.linalg.norm(rpy_after - rpy_target))

        sticky_status = sticky_status or {}
        state = sticky_status.get("state")
        self._update_grasp_counters(state)

        row = dict(
            step_idx=step_idx,
            t_wall=time.time() - self.t_start,
            raw_action=np.asarray(raw_action, dtype=float),
            exec_action=np.asarray(exec_action, dtype=float),
            vla_inference_ms=float(vla_inference_ms),
            execution_wall_ms=float(execution_wall_ms),
            ik_solve_ms=float(ik_info.get("solve_time_ms", np.nan)),
            ik_stage=str(ik_info.get("ik_stage", "")),
            ik_solver=str(ik_info.get("solver", "")),
            ik_seed=str(ik_info.get("seed", "")),
            ik_status=str(ik_info.get("status", "")),
            ik_ok=bool(ik_info.get("ok", False)),
            ik_accepted=bool(ik_info.get("accepted", False)),
            ik_feasible=bool(ik_info.get("feasible", False)),
            ik_score=float(ik_info.get("score", np.nan)),
            ik_pos_err=float(ik_info.get("pos_err", np.nan)),
            ik_rot_err=float(ik_info.get("rot_err", np.nan)),
            ik_joint_displacement=float(ik_info.get("joint_displacement", np.nan)),
            ik_joint_acceleration_cost=float(ik_info.get("joint_acceleration_cost", np.nan)),
            ik_joint_limit_cost=float(ik_info.get("joint_limit_cost", np.nan)),
            ik_singularity_cost=float(ik_info.get("singularity_cost", np.nan)),
            ik_collision_cost=float(ik_info.get("collision_cost", np.nan)),
            ik_escalation_reason=str(ik_info.get("escalation_reason") or ""),
            ik_fallback_reason=str(ik_info.get("fallback_reason") or ""),
            ik_attempt_count=int(ik_info.get("attempt_count", 0)),
            ik_candidate_trace_json=_json_safe(ik_info.get("candidate_trace", [])),
            eef_pos_before=np.asarray(eef_pos_before, dtype=float),
            eef_rpy_before=rpy_before,
            eef_pos_after=np.asarray(eef_pos_after, dtype=float),
            eef_rpy_after=rpy_after,
            commanded_target_pos=(np.asarray(commanded_target_pos, dtype=float)
                                  if commanded_target_pos is not None else np.full(3, np.nan)),
            commanded_target_quat=(np.asarray(commanded_target_quat, dtype=float)
                                   if commanded_target_quat is not None else np.full(4, np.nan)),
            tracking_pos_err=tracking_pos_err if tracking_pos_err is not None else np.nan,
            tracking_rot_err=tracking_rot_err if tracking_rot_err is not None else np.nan,
            sticky_state=str(state or ""),
            sticky_transition=str(sticky_status.get("transition") or ""),
            sticky_transition_reason=str(sticky_status.get("transition_reason") or ""),
            sticky_raw_command=float(sticky_status.get("raw_command", np.nan) or np.nan),
            sticky_normalized_command=float(sticky_status.get("normalized_command", np.nan) or np.nan),
            sticky_filtered_command=float(sticky_status.get("filtered_command", np.nan) or np.nan),
            sticky_attached=bool(sticky_status.get("attached", False)),
            sticky_body_name=str(sticky_status.get("body_name") or ""),
            sticky_candidate_distance=float(sticky_status.get("candidate_distance", np.nan) or np.nan),
            success=bool(success),
            native_success=bool(native_success),
            scalar_success=bool(scalar_success),
            keyframe_index=len(self._keyframes),
        )
        self._step_rows.append(row)
        self._keyframes.append(encode_jpeg(keyframe, quality=self.jpeg_quality))

        self._ik_stage_counts[row["ik_stage"]] += 1
        if np.isfinite(row["ik_pos_err"]):
            self._ik_pos_err.append(row["ik_pos_err"])
        if np.isfinite(row["ik_rot_err"]):
            self._ik_rot_err.append(row["ik_rot_err"])
        if np.isfinite(row["tracking_pos_err"]):
            self._tracking_pos_err.append(row["tracking_pos_err"])
        if np.isfinite(row["tracking_rot_err"]):
            self._tracking_rot_err.append(row["tracking_rot_err"])

    def _update_grasp_counters(self, new_state: str | None):
        prev = self._prev_sticky_state
        if prev == "OPEN" and new_state == "SEEKING":
            self.grasp_attempts += 1
        elif prev == "SEEKING" and new_state == "ATTACHED":
            self.grasp_attaches += 1
        elif prev in ("ATTACHED", "RELEASE_PENDING") and new_state == "OPEN":
            self.grasp_releases += 1
        if new_state is not None:
            self._prev_sticky_state = new_state

    # ------------------------------------------------------------------
    # Finalize + save
    # ------------------------------------------------------------------

    def finalize(self, success: bool, stop_reason: str, completed: bool = True,
                 mem_start: dict | None = None, mem_end: dict | None = None):
        self.success = bool(success)
        self.stop_reason = str(stop_reason)
        self.completed = bool(completed)
        self.t_end = time.time()
        self.mem_start = mem_start or {}
        self.mem_end = mem_end or {}

    def _summary_attrs(self) -> dict:
        n_ticks = len(self._tick_rows)
        completion_time_wall_s = (self.t_end or time.time()) - self.t_start
        completion_time_sim_s = (n_ticks * self.physics_dt) if self.physics_dt else None
        attempts = max(self.grasp_attempts, 1) if self.grasp_attempts else 0
        attach_rate = (self.grasp_attaches / self.grasp_attempts) if self.grasp_attempts else None

        attrs = dict(
            episode_id=self.episode_id,
            session_id=self.session_meta.get("session_id"),
            preset_name=self.preset_name,
            t_start_iso=self.t_start_iso,
            t_end_iso=datetime.now(timezone.utc).isoformat(),
            success=self.success,
            stop_reason=self.stop_reason,
            completed=self.completed,
            completion_steps=len(self._step_rows),
            completion_ticks=n_ticks,
            completion_time_wall_s=completion_time_wall_s,
            completion_time_sim_s=completion_time_sim_s,
            sim_freq_hz=self.sim_freq_hz,
            ctrl_freq_hz=self.ctrl_freq_hz,
            physics_dt=self.physics_dt,
            hard_collision_rate=(self._hard_collision_ticks / n_ticks) if n_ticks else None,
            warn_collision_rate=(self._warn_collision_ticks / n_ticks) if n_ticks else None,
            grasp_attempts=self.grasp_attempts,
            grasp_attaches=self.grasp_attaches,
            grasp_releases=self.grasp_releases,
            attach_rate=attach_rate,
            ik_stage_counts_json=_json_safe(dict(self._ik_stage_counts)),
            ik_pos_err_mean=float(np.mean(self._ik_pos_err)) if self._ik_pos_err else None,
            ik_pos_err_max=float(np.max(self._ik_pos_err)) if self._ik_pos_err else None,
            ik_rot_err_mean=float(np.mean(self._ik_rot_err)) if self._ik_rot_err else None,
            ik_rot_err_max=float(np.max(self._ik_rot_err)) if self._ik_rot_err else None,
            tracking_pos_err_mean=float(np.mean(self._tracking_pos_err)) if self._tracking_pos_err else None,
            tracking_pos_err_max=float(np.max(self._tracking_pos_err)) if self._tracking_pos_err else None,
            tracking_rot_err_mean=float(np.mean(self._tracking_rot_err)) if self._tracking_rot_err else None,
            tracking_rot_err_max=float(np.max(self._tracking_rot_err)) if self._tracking_rot_err else None,
            ram_used_start_gb=self.mem_start.get("ram_used_gb") if getattr(self, "mem_start", None) else None,
            ram_used_end_gb=self.mem_end.get("ram_used_gb") if getattr(self, "mem_end", None) else None,
            vram_used_start_gb=self.mem_start.get("vram_used_gb") if getattr(self, "mem_start", None) else None,
            vram_used_end_gb=self.mem_end.get("vram_used_gb") if getattr(self, "mem_end", None) else None,
            ik_config_json=self.ik_config_json,
            sticky_config_json=self.sticky_config_json,
            episode_cfg_json=self.episode_cfg_json,
        )
        attrs.update(self.extra_meta)
        return attrs

    def save(self) -> Path:
        import h5py

        self.output_root.mkdir(parents=True, exist_ok=True)
        fname = f"{self.preset_name}__{self.episode_id}.h5"
        path = self.output_root / fname

        with h5py.File(path, "w") as f:
            for k, v in self._summary_attrs().items():
                _set_attr(f.attrs, k, v)

            g_session = f.create_group("session")
            for k, v in self.session_meta.items():
                _set_attr(g_session.attrs, k, v)

            self._write_step_group(f)
            self._write_tick_group(f)
            self._write_frame_datasets(f)

        self._manifest_append(path)
        return path

    # ------------------------------------------------------------------
    # Internal HDF5 writers
    # ------------------------------------------------------------------

    def _kwargs(self):
        if not self.hdf5_compression:
            return {}
        return dict(compression=self.hdf5_compression,
                    compression_opts=self.hdf5_compression_level)

    def _write_step_group(self, f):
        g = f.create_group("vla_steps")
        rows = self._step_rows
        n = len(rows)
        if n == 0:
            return
        str_fields = ("ik_stage", "ik_solver", "ik_seed", "ik_status",
                      "ik_escalation_reason", "ik_fallback_reason",
                      "ik_candidate_trace_json", "sticky_state", "sticky_transition",
                      "sticky_transition_reason", "sticky_body_name")
        array_fields = ("raw_action", "exec_action", "eef_pos_before", "eef_rpy_before",
                        "eef_pos_after", "eef_rpy_after", "commanded_target_pos",
                        "commanded_target_quat")
        skip = set(str_fields) | set(array_fields)

        for key in rows[0].keys():
            if key in str_fields:
                data = _object_array([r[key] for r in rows])
                g.create_dataset(key, data=data, dtype=_str_dtype())
            elif key in array_fields:
                data = np.stack([r[key] for r in rows]).astype(float)
                g.create_dataset(key, data=data, **self._kwargs())
            elif key not in skip:
                data = np.array([r[key] for r in rows])
                g.create_dataset(key, data=data, **(self._kwargs() if data.size > 64 else {}))

    def _write_tick_group(self, f):
        g = f.create_group("ticks")
        rows = self._tick_rows
        n = len(rows)
        if n == 0:
            return
        array_fields = ("q_cmd", "q_actual", "qvel", "qacc", "actuator_force",
                        "qfrc_applied", "eef_pos", "eef_rpy")
        for key in rows[0].keys():
            if key in array_fields:
                data = np.stack([r[key] for r in rows]).astype(float)
                g.create_dataset(key, data=data, **self._kwargs())
            elif key == "collision_min_dist":
                data = np.array([np.nan if r[key] is None else r[key] for r in rows], dtype=float)
                g.create_dataset(key, data=data, **self._kwargs())
            else:
                data = np.array([r[key] for r in rows])
                g.create_dataset(key, data=data, **(self._kwargs() if data.size > 64 else {}))

    def _write_frame_datasets(self, f):
        vlen_dtype = __import__("h5py").vlen_dtype(np.dtype("uint8"))

        if self._keyframes:
            f.create_dataset("keyframes", data=_object_array(self._keyframes),
                              dtype=vlen_dtype)
        if self._video_frames:
            f.create_dataset("video_frames", data=_object_array(self._video_frames),
                              dtype=vlen_dtype)
            f.create_dataset("video_frames_vla_step_idx",
                              data=np.array(self._video_frame_vla_step_idx, dtype=int))
            f.create_dataset("video_frames_order",
                              data=np.array(self._video_frame_order, dtype=int))

    def _manifest_append(self, path: Path):
        manifest_path = self.output_root / "manifest.csv"
        row = {
            "episode_id": self.episode_id,
            "session_id": self.session_meta.get("session_id"),
            "preset_name": self.preset_name,
            "success": self.success,
            "stop_reason": self.stop_reason,
            "completed": self.completed,
            "completion_steps": len(self._step_rows),
            "completion_time_sim_s": (len(self._tick_rows) * self.physics_dt) if self.physics_dt else "",
            "attach_rate": (self.grasp_attaches / self.grasp_attempts) if self.grasp_attempts else "",
            "hard_collision_rate": (self._hard_collision_ticks / len(self._tick_rows)) if self._tick_rows else "",
            "t_start_iso": self.t_start_iso,
            "path": str(path),
        }
        write_header = not manifest_path.exists()
        with open(manifest_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _capture_tick_row(ctrl, vla_step_idx: int, tick_idx: int) -> dict:
    """Pure(ish) scalar capture for one physics tick. Kept as a free function
    (not a method) so it can be unit tested against a duck-typed fake
    controller without constructing an EpisodeLogger."""
    dof_ids = ctrl.get_arm_dof_ids()
    act_ids = ctrl.bridge.get_arm_actuator_ids()

    eef_pos, eef_quat = ctrl.get_eef_pose()
    min_dist, hard = ctrl.ik.collision_status()

    return dict(
        tick_idx=tick_idx,
        vla_step_idx=vla_step_idx,
        t_wall=time.time(),
        q_cmd=ctrl.bridge.get_arm_ctrl().astype(float),
        q_actual=ctrl.get_arm_qpos().astype(float),
        qvel=np.asarray(ctrl.data.qvel[dof_ids], dtype=float),
        qacc=np.asarray(ctrl.data.qacc[dof_ids], dtype=float),
        actuator_force=np.asarray(ctrl.data.actuator_force[act_ids], dtype=float),
        qfrc_applied=np.asarray(ctrl.data.qfrc_applied[dof_ids], dtype=float),
        eef_pos=np.asarray(eef_pos, dtype=float),
        eef_rpy=quat_wxyz_to_euler(eef_quat),
        collision_min_dist=None if min_dist is None else float(min_dist),
        collision_hard=bool(hard),
    )

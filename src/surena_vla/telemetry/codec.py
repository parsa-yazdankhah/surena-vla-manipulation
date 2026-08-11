"""Small, dependency-minimal codecs used by the episode logger.

Kept separate from ``episode_logger.py`` so the encode/decode contract
(exact JPEG quality, RGB<->BGR handling, quaternion convention) has one
authoritative place and can be unit tested without MuJoCo/robosuite.
"""

from __future__ import annotations

import numpy as np


def encode_jpeg(frame_rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    """RGB uint8 HxWx3 -> 1-D uint8 array of JPEG bytes (for HDF5 vlen storage)."""
    import cv2

    frame_rgb = np.asarray(frame_rgb)
    if frame_rgb.dtype != np.uint8:
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.reshape(-1)


def decode_jpeg(buf: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_jpeg`. Returns RGB uint8 HxWx3."""
    import cv2

    bgr = cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("JPEG decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def quat_wxyz_to_euler(quat_wxyz) -> np.ndarray:
    """MuJoCo (w, x, y, z) quaternion -> (roll, pitch, yaw) radians, XYZ intrinsic.

    Pure numpy; matches the convention used throughout ``surena_vla.control``
    (``mujoco.mju_mat2Quat`` / ``mat_to_quat``), so this is safe to use on
    every quaternion already flowing through the controller.
    """
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=float)

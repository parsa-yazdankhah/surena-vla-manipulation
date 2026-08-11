"""Episode-rollout telemetry for thesis evaluation metrics.

One compact HDF5 file per rollout (session/episode metadata, per-VLA-step
and per-physics-tick traces, keyframes, and video-cadence frames). See
``episode_logger.py`` for the schema.

The aggregation/plotting/statistics layer (scanning a directory of ``.h5``
files into thesis tables, violin plots, trajectory plots, and the
with/without-action-history video export) is intentionally not part of this
module yet — it's a separate, deferred piece of work that consumes what
this module logs.
"""

from .episode_logger import EpisodeLogger
from .hardware import collect_session_meta, sample_memory_usage
from .codec import encode_jpeg, decode_jpeg, quat_wxyz_to_euler

__all__ = [
    "EpisodeLogger",
    "collect_session_meta",
    "sample_memory_usage",
    "encode_jpeg",
    "decode_jpeg",
    "quat_wxyz_to_euler",
]

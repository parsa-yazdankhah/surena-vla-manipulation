"""One-shot session/hardware metadata capture (the 'laptop config' record).

Call once per kernel session, not per episode. The resulting dict is small
and is duplicated into every episode file's attrs so each ``.h5`` stays
self-contained (no join required to know what hardware produced it).
"""

from __future__ import annotations

from datetime import datetime, timezone
import platform
import uuid


def collect_session_meta(vla_checkpoint=None, vla_unnorm_key: str | None = None) -> dict:
    meta: dict = {
        "session_id": uuid.uuid4().hex[:12],
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "vla_checkpoint": str(vla_checkpoint) if vla_checkpoint else None,
        "vla_unnorm_key": vla_unnorm_key,
    }

    try:
        import psutil
        meta["cpu_model"] = platform.processor() or platform.uname().processor or "unknown"
        meta["cpu_count_logical"] = psutil.cpu_count(logical=True)
        meta["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 3)
    except Exception as exc:
        meta["hardware_meta_error"] = f"psutil unavailable: {exc!r}"

    try:
        import torch
        meta["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info()
            meta["vram_total_gb"] = round(total_b / 1e9, 3)
    except Exception as exc:
        meta["torch_meta_error"] = f"torch/cuda unavailable: {exc!r}"

    from importlib import metadata as _md
    for pkg in ("mujoco", "mink", "robosuite", "transformers", "h5py",
                "opencv-python", "numpy", "GitPython"):
        try:
            meta[f"{pkg.replace('-', '_').lower()}_version"] = _md.version(pkg)
        except Exception:
            pass

    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        meta["git_commit"] = repo.head.commit.hexsha
        meta["git_dirty"] = bool(repo.is_dirty())
    except Exception:
        meta["git_commit"] = None
        meta["git_dirty"] = None

    return meta


def sample_memory_usage() -> dict:
    """Cheap point-in-time RAM/VRAM sample. Call at episode start/end only —
    not per-step/per-tick, since querying CUDA can add sync overhead that
    would bias the latency numbers you're trying to measure."""
    out: dict = {}
    try:
        import psutil
        out["ram_used_gb"] = round(psutil.virtual_memory().used / 1e9, 3)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            out["vram_used_gb"] = round((total_b - free_b) / 1e9, 3)
            out["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return out

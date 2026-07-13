"""Resolve LIBERO data paths without trusting stale global configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import yaml


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def get_libero_package_root() -> Path:
    """Return the active editable-installed ``libero.libero`` directory."""
    import libero.libero as libero_package

    return Path(libero_package.__file__).resolve().parent


def get_libero_config_file() -> Path:
    """Return LIBERO's global YAML configuration path."""
    config_dir = Path(
        os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")
    ).expanduser()
    return config_dir / "config.yaml"


def _configured_bddl_root() -> Path | None:
    config_file = get_libero_config_file()
    if not config_file.is_file():
        return None

    try:
        config = yaml.safe_load(config_file.read_text()) or {}
    except Exception:
        return None

    value = config.get("bddl_files")
    return Path(value) if value else None


def get_bddl_root() -> Path:
    candidates: list[Path] = []

    override = os.environ.get("LIBERO_BDDL_ROOT")
    if override:
        candidates.append(Path(override))

    configured = _configured_bddl_root()
    if configured is not None:
        candidates.append(configured)

    package_root = get_libero_package_root()
    candidates.append(package_root / "bddl_files")

    tried = _unique_paths(candidates)
    for root in tried:
        if root.is_dir():
            return root

    tried_text = "\n  - ".join(str(path) for path in tried)
    raise FileNotFoundError(
        "Could not locate a valid LIBERO BDDL root. Tried:\n"
        f"  - {tried_text}\n"
        f"LIBERO config file: {get_libero_config_file()}\n"
        "Run scripts/configure_libero_paths.py or set LIBERO_BDDL_ROOT."
    )


def libero90_bddl(filename: str) -> Path:
    path = get_bddl_root() / "libero_90" / filename
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO task file does not exist: {path}")
    return path

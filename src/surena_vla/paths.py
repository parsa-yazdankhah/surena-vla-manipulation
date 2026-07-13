"""Filesystem paths for package assets and development outputs."""

from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"
MESHES_ROOT = ASSETS_ROOT / "meshes"
SURENA_ARM_XML = ASSETS_ROOT / "surena_arm.xml"

def _find_project_root() -> Path:
    override = os.environ.get("SURENA_VLA_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for candidate in PACKAGE_ROOT.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()

PROJECT_ROOT = _find_project_root()
OUTPUTS_ROOT = Path(os.environ.get("SURENA_VLA_OUTPUTS_DIR", PROJECT_ROOT / "outputs")).expanduser().resolve()
VIDEOS_ROOT = OUTPUTS_ROOT / "videos"

def validate_asset_layout() -> None:
    if not SURENA_ARM_XML.is_file():
        raise FileNotFoundError(f"SURENA MJCF not found: {SURENA_ARM_XML}")
    if not MESHES_ROOT.is_dir():
        raise FileNotFoundError(f"SURENA mesh directory not found: {MESHES_ROOT}")
    root = ET.parse(SURENA_ARM_XML).getroot()
    missing = []
    for mesh in root.findall(".//asset/mesh[@file]"):
        mesh_path = (ASSETS_ROOT / mesh.attrib["file"]).resolve()
        if not mesh_path.is_file():
            missing.append(str(mesh_path))
    if missing:
        raise FileNotFoundError("Missing SURENA mesh files:\n  - " + "\n  - ".join(missing))

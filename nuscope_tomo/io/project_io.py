"""Project persistence: a NuScope project is a directory containing one
zarr store per named volume (raw / aligned / segmentation) plus a
project.json manifest. Saving re-uses `VolumeData.to_zarr`, so writing a
project is chunk-wise and never loads a full volume into memory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..data.volume import VolumeData

MANIFEST_NAME = "project.json"


def save_project(
    project_path: str | Path,
    volumes: dict[str, VolumeData],
    active: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict[str, VolumeData]:
    """Write `volumes` into `project_path`, one zarr store per key.

    Returns the (possibly re-bound) volumes dict, since a volume not
    already stored at its target path is copied there.
    """
    project_path = Path(project_path)
    project_path.mkdir(parents=True, exist_ok=True)

    entries = {}
    saved: dict[str, VolumeData] = {}
    for key, vol in volumes.items():
        target = project_path / f"{key}.zarr"
        if vol.store_path is not None and vol.store_path.resolve() == target.resolve():
            saved_vol = vol
        else:
            saved_vol = vol.to_zarr(target)
        entries[key] = target.name
        saved[key] = saved_vol

    manifest = {
        "active": active or next(iter(saved), None),
        "volumes": entries,
        "extra": extra or {},
    }
    (project_path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return saved


def load_project(project_path: str | Path) -> tuple[dict[str, VolumeData], Optional[str], dict]:
    project_path = Path(project_path)
    manifest_path = project_path / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Not a NuScope project (no {MANIFEST_NAME}): {project_path}")
    manifest = json.loads(manifest_path.read_text())

    volumes = {
        key: VolumeData.from_zarr(project_path / filename)
        for key, filename in manifest.get("volumes", {}).items()
    }
    return volumes, manifest.get("active"), manifest.get("extra", {})

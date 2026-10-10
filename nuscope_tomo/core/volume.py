"""Out-of-core 3D volume: a Dask array plus physical voxel size and metadata.

Every volume in NuScope-TOMO is (Z, Y, X) with Z the FIB milling direction.
Data lives in chunked Zarr v3 stores and is touched through Dask, so stacks
larger than RAM can flow through the whole pipeline. ``preview()`` and
``roi()`` are the explicit points where data is materialised in memory.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import dask.array as da
import numpy as np
import zarr

META_FILE = "nuscope_meta.json"


@dataclass
class VoxelSize:
    """Physical voxel size in nanometres, (z, y, x)."""

    z: float = 1.0
    y: float = 1.0
    x: float = 1.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.z, self.y, self.x)

    def scaled(self, fz: float, fy: float, fx: float) -> "VoxelSize":
        return VoxelSize(self.z * fz, self.y * fy, self.x * fx)

    @property
    def voxel_volume_nm3(self) -> float:
        return self.z * self.y * self.x


@dataclass
class Volume:
    """A lazily evaluated 3D volume with physical calibration.

    Parameters
    ----------
    data : dask or numpy array, shape (Z, Y, X)
    voxel_size : VoxelSize in nm
    name : label shown in the UI and in provenance logs
    kind : "intensity", "labels" (integer classes/objects) or "element" (EDS map)
    meta : free-form metadata (provenance, element name, thresholds, ...)
    """

    data: Any
    voxel_size: VoxelSize = field(default_factory=VoxelSize)
    name: str = "volume"
    kind: str = "intensity"
    meta: dict = field(default_factory=dict)
    store_path: Optional[Path] = None

    def __post_init__(self):
        if not isinstance(self.data, da.Array):
            arr = np.asarray(self.data)
            self.data = da.from_array(arr, chunks=_default_chunks(arr.shape))
        if self.data.ndim != 3:
            raise ValueError(f"Volume expects (Z, Y, X), got shape {self.data.shape}")

    # ------------------------------------------------------------------ props
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    @property
    def extent_nm(self) -> tuple[float, float, float]:
        return tuple(n * v for n, v in zip(self.shape, self.voxel_size.as_tuple()))

    def __repr__(self) -> str:
        z, y, x = self.extent_nm
        return (f"Volume({self.name!r}, kind={self.kind}, shape={self.shape}, "
                f"dtype={self.dtype}, voxel={self.voxel_size.as_tuple()} nm, "
                f"extent={z/1e3:.2f}x{y/1e3:.2f}x{x/1e3:.2f} µm)")

    def derive(self, data, name: Optional[str] = None, kind: Optional[str] = None,
               voxel_size: Optional[VoxelSize] = None, **meta) -> "Volume":
        """New volume that inherits calibration/metadata from this one."""
        return Volume(data, voxel_size or self.voxel_size, name or self.name,
                      kind or self.kind, {**self.meta, **meta})

    # ----------------------------------------------------------------- access
    def numpy(self) -> np.ndarray:
        return np.asarray(self.data.compute() if isinstance(self.data, da.Array) else self.data)

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        index = int(np.clip(index, 0, self.shape[axis] - 1))
        sl = [slice(None)] * 3
        sl[axis] = index
        return np.asarray(self.data[tuple(sl)])

    def roi(self, z=slice(None), y=slice(None), x=slice(None)) -> "Volume":
        return self.derive(self.data[z, y, x])

    def preview(self, max_voxels: int = 256**3, isotropic: bool = False) -> "Volume":
        """Downsampled in-memory copy (block mean for intensity, stride for labels).

        With ``isotropic=True`` the per-axis factors also equalise physical
        voxel size, which is what 3D rendering wants for strongly
        anisotropic FIB-SEM data (thin XY pixels, thick slices).
        """
        vs = np.array(self.voxel_size.as_tuple())
        if isotropic:
            target = vs.max()
            f = np.maximum(1, np.round(target / vs)).astype(int)
        else:
            f = np.ones(3, int)
        total = np.prod(np.array(self.shape) / f)
        if total > max_voxels:
            g = (total / max_voxels) ** (1 / 3)
            f = np.maximum(1, np.ceil(f * g)).astype(int)
        f = tuple(int(v) for v in f)
        if f == (1, 1, 1):
            return self.derive(np.asarray(self.data))
        if self.kind == "labels":
            out = np.asarray(self.data[::f[0], ::f[1], ::f[2]])
        else:
            trimmed = self.data[: self.shape[0] // f[0] * f[0],
                                : self.shape[1] // f[1] * f[1],
                                : self.shape[2] // f[2] * f[2]]
            out = da.coarsen(np.mean, trimmed.astype(np.float32),
                             dict(enumerate(f)), trim_excess=True).compute()
            out = out.astype(self.dtype) if np.issubdtype(self.dtype, np.integer) else out
        return self.derive(out, voxel_size=self.voxel_size.scaled(*f), downsample=f)

    def histogram(self, bins: int = 256, step: int = 4,
                  value_range: Optional[Sequence[float]] = None):
        arr = self.data[::step, ::step, ::step]
        if value_range is None:
            if np.issubdtype(self.dtype, np.integer) and self.dtype.itemsize == 1:
                value_range = (0, 255)
            else:
                lo, hi = da.compute(arr.min(), arr.max())
                value_range = (float(lo), float(hi))
        h, edges = da.histogram(arr, bins=bins, range=value_range)
        return np.asarray(h.compute()), np.asarray(edges)

    # -------------------------------------------------------------------- I/O
    def to_zarr(self, path: str | Path, chunks: Optional[Sequence[int]] = None,
                overwrite: bool = True) -> "Volume":
        """Write chunk by chunk and return a volume bound to the new store."""
        path = Path(path)
        if path.exists():
            if not overwrite:
                raise FileExistsError(path)
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        chunks = tuple(chunks) if chunks else _default_chunks(self.shape)
        arr = self.data.rechunk(chunks)
        store = zarr.create_array(str(path), shape=self.shape, chunks=chunks,
                                  dtype=self.dtype, overwrite=True)
        da.store(arr, store, lock=False)
        self._write_meta(path)
        return Volume.from_zarr(path)

    def _write_meta(self, path: Path):
        meta = {"name": self.name, "kind": self.kind,
                "voxel_size": asdict(self.voxel_size), "meta": _jsonable(self.meta)}
        (path / META_FILE).write_text(json.dumps(meta, indent=2))

    @classmethod
    def from_zarr(cls, path: str | Path) -> "Volume":
        path = Path(path)
        arr = zarr.open_array(str(path), mode="r")
        data = da.from_zarr(arr)
        name, kind, vs, meta = path.stem, "intensity", VoxelSize(), {}
        mp = path / META_FILE
        if mp.exists():
            m = json.loads(mp.read_text())
            name = m.get("name", name)
            kind = m.get("kind", "labels" if "mask" in name else kind)
            vs = VoxelSize(**m.get("voxel_size", {}))
            # v0.1 stores kept extra keys at top level
            meta = m.get("meta", {k: v for k, v in m.items()
                                  if k not in ("name", "kind", "voxel_size")})
        return cls(data, vs, name, kind, meta, store_path=path)


def _default_chunks(shape) -> tuple[int, int, int]:
    """Slab chunks for slice-wise FIB-SEM processing, tiled in XY if huge."""
    z, y, x = shape
    return (min(z, 16), min(y, 1024), min(x, 1024))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 10000 else f"<array {obj.shape}>"
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj

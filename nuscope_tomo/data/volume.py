m"""Out-of-core 3D volume backed by a zarr store, indexed lazily via dask.

A FIB-SEM slice stack is stored on disk as a zarr array (shape = (Z, Y, X))
and only ever touched through dask, so a dataset far larger than RAM can be
imported, aligned and segmented without loading it whole. Anything that must
reach the GPU (volume rendering, marching cubes) goes through `preview()` or
`roi()`, which are the two deliberate, documented points where data is
materialized into memory.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import dask.array as da
import numpy as np
import zarr


@dataclass
class VoxelSize:
    """Physical voxel size in nanometers, (z, y, x) order."""

    z: float = 1.0
    y: float = 1.0
    x: float = 1.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.z, self.y, self.x)


class VolumeData:
    """A lazily-loaded 3D volume backed by a zarr store on disk.

    Parameters
    ----------
    array:
        A dask array wrapping a zarr store, shape (Z, Y, X).
    store_path:
        Location of the backing zarr store, if any (None for purely
        in-memory volumes created for tests).
    voxel_size:
        Physical size of one voxel in nanometers.
    name:
        Human-readable label for the volume (shown in the UI).
    """

    METADATA_FILENAME = "nuscope_meta.json"

    def __init__(
        self,
        array: da.Array,
        store_path: Optional[Path] = None,
        voxel_size: Optional[VoxelSize] = None,
        name: str = "volume",
    ):
        if array.ndim != 3:
            raise ValueError(f"VolumeData expects a 3D array, got shape {array.shape}")
        self.array = array
        self.store_path = Path(store_path) if store_path is not None else None
        self.voxel_size = voxel_size or VoxelSize()
        self.name = name

    # -- basic properties -------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def chunks(self):
        return self.array.chunks

    @property
    def nbytes(self) -> int:
        return self.array.nbytes

    def __repr__(self) -> str:
        return (
            f"VolumeData(name={self.name!r}, shape={self.shape}, dtype={self.dtype}, "
            f"voxel_size={self.voxel_size.as_tuple()} nm, "
            f"store={self.store_path})"
        )

    # -- disk I/O -----------------------------------------------------------
    @classmethod
    def from_zarr(cls, path: str | Path) -> "VolumeData":
        path = Path(path)
        z = zarr.open(str(path), mode="r")
        array = da.from_zarr(z)
        voxel_size = VoxelSize()
        name = path.stem
        meta_path = path / cls.METADATA_FILENAME
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            voxel_size = VoxelSize(**meta.get("voxel_size", {}))
            name = meta.get("name", name)
        return cls(array, store_path=path, voxel_size=voxel_size, name=name)

    def to_zarr(self, path: str | Path, chunks: Optional[tuple[int, int, int]] = None,
                overwrite: bool = True) -> "VolumeData":
        """Persist this volume to a new zarr store, chunk-by-chunk (out-of-core).

        Returns a new VolumeData bound to the written store.
        """
        path = Path(path)
        if path.exists():
            if not overwrite:
                raise FileExistsError(path)
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        array = self.array.rechunk(chunks) if chunks else self.array
        da.to_zarr(array, str(path), overwrite=True)

        meta = {"voxel_size": asdict(self.voxel_size), "name": self.name}
        (path / self.METADATA_FILENAME).write_text(json.dumps(meta, indent=2))

        return VolumeData.from_zarr(path)

    # -- data access for rendering / processing ------------------------------
    def get_slice(self, axis: int, index: int) -> np.ndarray:
        """Read a single 2D plane straight from disk (cheap, no full load)."""
        if not (0 <= axis <= 2):
            raise ValueError("axis must be 0 (Z), 1 (Y) or 2 (X)")
        index = int(np.clip(index, 0, self.shape[axis] - 1))
        indexer = [slice(None)] * 3
        indexer[axis] = index
        return np.asarray(self.array[tuple(indexer)])

    def roi(self, z: slice, y: slice, x: slice) -> np.ndarray:
        """Load a full-resolution crop into memory."""
        return np.asarray(self.array[z, y, x])

    def preview(self, max_voxels: int = 128**3) -> np.ndarray:
        """Return a downsampled, in-memory copy of the whole volume.

        Used for whole-volume rendering / meshing where materializing the
        full-resolution array would not fit in RAM or on the GPU. The
        downsample factor is chosen (per axis, isotropically where possible)
        so the result has roughly `max_voxels` voxels.
        """
        total = int(np.prod(self.shape))
        if total <= max_voxels:
            return np.asarray(self.array)

        factor = round((total / max_voxels) ** (1 / 3))
        factor = max(factor, 1)
        factors = {0: factor, 1: factor, 2: factor}
        # coarsen requires each axis to be evenly divisible by its factor;
        # trim the array first.
        trimmed_shape = tuple(
            (self.shape[ax] // factors[ax]) * factors[ax] for ax in range(3)
        )
        trimmed = self.array[: trimmed_shape[0], : trimmed_shape[1], : trimmed_shape[2]]
        coarse = da.coarsen(np.mean, trimmed, factors, trim_excess=True)
        return np.asarray(coarse).astype(self.array.dtype)

    def histogram(self, bins: int = 256, sample_step: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Out-of-core intensity histogram (dask reduction, no full load)."""
        arr = self.array[::sample_step, ::sample_step, ::sample_step]
        info = np.iinfo(self.dtype) if np.issubdtype(self.dtype, np.integer) else None
        lo, hi = (0, info.max) if info is not None else (float(arr.min()), float(arr.max()))
        h, edges = da.histogram(arr, bins=bins, range=(lo, hi))
        return np.asarray(h.compute()), np.asarray(edges)

    def physical_extent_nm(self) -> tuple[float, float, float]:
        """Total physical size of the volume in nm, (z, y, x)."""
        vz, vy, vx = self.voxel_size.as_tuple()
        z, y, x = self.shape
        return (z * vz, y * vy, x * vx)

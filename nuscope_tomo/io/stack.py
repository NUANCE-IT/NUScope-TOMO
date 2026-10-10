"""Slice-and-view image import/export.

Supports folders of 2D images (TIFF/PNG/BMP/JPEG, natural-sorted by the
trailing number in the file name) and multi-page TIFF stacks. Folders are
streamed one slice at a time into a Zarr store so memory stays bounded by a
single frame. Pixel size is read from Thermo Fisher (FEI) TIFF metadata when
present.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import dask.array as da
import numpy as np
import tifffile
import zarr

from ..core.registry import op
from ..core.volume import META_FILE, Volume, VoxelSize, _jsonable

SUFFIXES = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}
_NUM = re.compile(r"(\d+)(?=\D*$)")


def natural_key(p: Path):
    m = _NUM.search(p.stem)
    return (p.stem[: m.start()] if m else p.stem, int(m.group(1)) if m else -1)


def list_slices(folder: str | Path, pattern: Optional[str] = None) -> list[Path]:
    folder = Path(folder)
    files = (list(folder.glob(pattern)) if pattern
             else [p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES
                   and not p.name.startswith(".")])
    if not files:
        raise FileNotFoundError(f"No slice images in {folder}")
    return sorted(files, key=natural_key)


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(str(path))
    else:
        import imageio.v3 as iio
        arr = iio.imread(str(path))
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):  # grey saved as RGB(A)
        arr = arr[..., :3].mean(-1).astype(arr.dtype) if not np.array_equal(
            arr[..., 0], arr[..., 1]) else arr[..., 0]
    return arr


def read_pixel_size_nm(path: Path) -> Optional[float]:
    """Pixel size from Thermo Fisher/FEI SEM TIFF tags, if available."""
    try:
        with tifffile.TiffFile(str(path)) as tf:
            fei = getattr(tf, "fei_metadata", None)
            if fei:
                for sect in ("Scan", "EScan"):
                    if sect in fei and "PixelWidth" in fei[sect]:
                        return float(fei[sect]["PixelWidth"]) * 1e9
            page = tf.pages[0]
            unit = page.tags.get("ResolutionUnit")
            xres = page.tags.get("XResolution")
            if unit and xres and unit.value == 3:  # centimetre
                num, den = xres.value
                if num:
                    return 1e7 * den / num
    except Exception:
        return None
    return None


def read_tiff_folder(folder: str | Path, voxel_size: Optional[VoxelSize] = None,
                     cache: Optional[str | Path] = None, pattern: Optional[str] = None,
                     progress: Optional[Callable[[int, int], None]] = None) -> Volume:
    """Stream an image sequence into a Zarr store (``cache``) and open it lazily."""
    files = list_slices(folder, pattern)
    first = read_image(files[0])
    h, w = first.shape
    if voxel_size is None:
        px = read_pixel_size_nm(files[0]) or 1.0
        voxel_size = VoxelSize(px, px, px)
    cache = Path(cache) if cache else Path(folder).with_suffix(".zarr")
    chunk_z = 8
    store = zarr.create_array(str(cache), shape=(len(files), h, w), chunks=(chunk_z, h, w),
                              dtype=first.dtype, overwrite=True)
    buf = np.empty((chunk_z, h, w), first.dtype)
    for i, f in enumerate(files):
        img = first if i == 0 else read_image(f)
        if img.shape != (h, w):  # tolerate off-by-few frames
            pad = np.zeros((h, w), first.dtype)
            hh, ww = min(h, img.shape[0]), min(w, img.shape[1])
            pad[:hh, :ww] = img[:hh, :ww]
            img = pad
        buf[i % chunk_z] = img
        if i % chunk_z == chunk_z - 1 or i == len(files) - 1:
            start = i - i % chunk_z
            store[start: i + 1] = buf[: i - start + 1]
        if progress:
            progress(i + 1, len(files))
    vol = Volume(da.from_zarr(store), voxel_size, Path(folder).name, "intensity",
                 {"source_folder": str(folder), "n_files": len(files)}, store_path=cache)
    vol._write_meta(cache)
    return vol


def read_tiff_stack(path: str | Path, voxel_size: Optional[VoxelSize] = None) -> Volume:
    path = Path(path)
    arr = tifffile.memmap(str(path)) if path.suffix.lower() in (".tif", ".tiff") else None
    if arr is None or arr.ndim != 3:
        arr = tifffile.imread(str(path))
    if voxel_size is None:
        px = read_pixel_size_nm(path) or 1.0
        voxel_size = VoxelSize(px, px, px)
    return Volume(da.from_array(arr, chunks=(8, -1, -1)), voxel_size, path.stem)


@op("io.export_tiff")
def export_tiff(volume: Volume, path: str = "export.tif", bigtiff: bool = True) -> Volume:
    """Write a volume as a (Big)TIFF stack with ImageJ-compatible calibration."""
    vs = volume.voxel_size
    with tifffile.TiffWriter(path, bigtiff=bigtiff, imagej=not bigtiff) as tw:
        for z in range(volume.shape[0]):
            tw.write(volume.get_slice(0, z), contiguous=True,
                     resolution=(1e3 / vs.x, 1e3 / vs.y),
                     metadata={"spacing": vs.z / 1e3, "unit": "um"} if z == 0 else None)
    return volume


@op("io.save")
def save(volume: Volume, path: str = "volume.zarr") -> Volume:
    """Persist a volume as a chunked Zarr v3 store."""
    return volume.to_zarr(path)


__all__ = ["list_slices", "read_image", "read_pixel_size_nm", "read_tiff_folder",
           "read_tiff_stack", "export_tiff", "save", "META_FILE", "_jsonable"]

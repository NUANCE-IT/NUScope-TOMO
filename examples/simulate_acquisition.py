"""Replay a stored stack as if a FIB-SEM were writing it, slice by slice.

    python examples/simulate_acquisition.py projects/fib_sem_335/raw.zarr /tmp/acq --rate 20

Used to demonstrate and test ``nuscope watch`` without a microscope.
"""
import argparse
import time
from pathlib import Path

import tifffile
import zarr

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("dest")
ap.add_argument("--rate", type=float, default=10, help="slices per second")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=0)
a = ap.parse_args()
src = zarr.open_array(a.source, mode="r")
dest = Path(a.dest)
dest.mkdir(parents=True, exist_ok=True)
end = src.shape[0] if not a.count else min(src.shape[0], a.start + a.count)
for i in range(a.start, end):
    tmp = dest / f".slice_{i:04d}.tif"
    tifffile.imwrite(tmp, src[i])
    tmp.rename(dest / f"Electron Image (SE) {i + 1:04d}.tif")
    time.sleep(1 / a.rate)
print(f"wrote {end - a.start} slices to {dest}")

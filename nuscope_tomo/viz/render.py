"""Publication-quality 3D rendering and animation (GPU, off-screen).

Scenes are in physical units (µm). Typical figures:

* ``render_volume``   – direct volume rendering of grey values
* ``render_phases``   – smooth iso-surfaces of segmented classes over an
  orthoslice cut-away (the classic Amira figure)
* ``render_objects``  – every separated object coloured by a measurement
* ``turntable``       – MP4/GIF orbit around any scene
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pyvista as pv
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from ..core.volume import Volume

pv.OFF_SCREEN = True

# colour-blind-safe, high-contrast categorical palette (Okabe–Ito inspired)
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
           "#CC79A7", "#8DD3C7", "#FB8072", "#80B1D3"]
BG_TOP, BG_BOTTOM = "#1b2232", "#05070c"


def to_grid(vol: Volume, name: str = "values") -> pv.ImageData:
    """Volume → VTK image in µm (VTK x,y,z = data X, Y, Z)."""
    arr = vol.numpy()
    vz, vy, vx = (v / 1e3 for v in vol.voxel_size.as_tuple())
    g = pv.ImageData(dimensions=(arr.shape[2], arr.shape[1], arr.shape[0]),
                     spacing=(vx, vy, vz))
    g.point_data[name] = arr.ravel(order="C")
    return g


def mask_surface(mask: np.ndarray, spacing_nm, offset_vox=(0, 0, 0), smooth: float = 1.0,
                 decimate: float = 0.0) -> Optional[pv.PolyData]:
    """Smooth closed iso-surface of a binary mask, in µm."""
    if not mask.any():
        return None
    f = np.pad(mask.astype(np.float32), 2)
    if smooth:
        f = ndi.gaussian_filter(f, smooth)
    try:
        v, faces, normals, _ = marching_cubes(f, 0.5, spacing=tuple(spacing_nm))
    except (ValueError, RuntimeError):
        return None
    v = (v - 2 * np.asarray(spacing_nm) + np.asarray(offset_vox) * np.asarray(spacing_nm)) / 1e3
    v = v[:, ::-1]  # (z,y,x) → (x,y,z)
    poly = pv.PolyData(v, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    if decimate:
        poly = poly.decimate(decimate)
    return poly


def _plotter(size=(1600, 1200)) -> pv.Plotter:
    p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    p.set_background(BG_BOTTOM, top=BG_TOP)
    p.enable_anti_aliasing("ssaa")
    return p


def _finish(p: pv.Plotter, bounds, out: Optional[str], camera: Optional[dict],
            axes: bool = True, scale_bar: bool = True):
    if axes:
        p.add_mesh(pv.Box(bounds).outline(), color="#8a94a6", line_width=1.2)
    p.camera_position = "iso"
    p.camera.azimuth = (camera or {}).get("azimuth", 35)
    p.camera.elevation = (camera or {}).get("elevation", 20)
    p.reset_camera()
    p.camera.zoom((camera or {}).get("zoom", 1.25))
    if scale_bar:
        _scale_bar(p, bounds)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        p.screenshot(out)
    return p


def _scale_bar(p: pv.Plotter, bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    span = x1 - x0
    nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    L = max(n for n in nice if n <= span * 0.4) if span > 0.25 else span / 2
    start = np.array([x0, y1 + 0.04 * (y1 - y0), z0])
    p.add_mesh(pv.Tube(pointa=start, pointb=start + [L, 0, 0], radius=span * 0.006), color="white")
    p.add_point_labels([start + [L / 2, 0.05 * (y1 - y0), 0]],
                       [f"{L:g} µm"], text_color="white", font_size=26, shape=None,
                       show_points=False, always_visible=True)


def render_volume(vol: Volume, out: Optional[str] = None, cmap: str = "bone",
                  clim: Optional[Sequence[float]] = None, opacity="sigmoid",
                  size=(1600, 1200), camera: Optional[dict] = None, title: str = ""):
    g = to_grid(vol)
    arr = g.point_data["values"]
    clim = clim or np.percentile(arr[arr > 0] if (arr > 0).any() else arr, (2, 99.8))
    p = _plotter(size)
    p.add_volume(g, scalars="values", cmap=cmap, clim=clim, opacity=opacity,
                 show_scalar_bar=False, shade=False, blending="composite")
    if title:
        p.add_text(title, font_size=14, color="white", position="upper_left")
    return _finish(p, g.bounds, out, camera)


def phase_surfaces(labels: Volume, classes: Optional[Sequence[int]] = None,
                   smooth: float = 1.0, decimate: float = 0.5) -> dict[int, pv.PolyData]:
    lab = labels.numpy()
    classes = classes or [int(c) for c in np.unique(lab) if c != 0]
    out = {}
    for c in classes:
        s = mask_surface(lab == c, labels.voxel_size.as_tuple(), smooth=smooth, decimate=decimate)
        if s is not None:
            out[c] = s
    return out


def render_phases(labels: Volume, intensity: Optional[Volume] = None,
                  classes: Optional[Sequence[int]] = None, colors: Optional[dict] = None,
                  opacities: Optional[dict] = None, cutaway: bool = True,
                  out: Optional[str] = None, size=(1600, 1200), camera: Optional[dict] = None,
                  title: str = "", legend: bool = True):
    """Iso-surfaces of classes; optional grey orthoslices of ``intensity``.

    With ``cutaway`` the front octant is removed so the interior and the
    raw data on the cut faces are visible.
    """
    lab = labels.numpy()
    names = labels.meta.get("classes", {})
    classes = classes or [int(c) for c in np.unique(lab) if c != 0]
    vs = labels.voxel_size.as_tuple()
    if cutaway:
        cz, cy, cx = (s // 2 for s in lab.shape)
        lab = lab.copy()
        lab[cz:, cy:, cx:] = 0
    p = _plotter(size)
    for i, c in enumerate(classes):
        s = mask_surface(lab == c, vs, decimate=0.4)
        if s is None:
            continue
        col = (colors or {}).get(c, PALETTE[i % len(PALETTE)])
        p.add_mesh(s, color=col, smooth_shading=True, specular=0.4, specular_power=20,
                   opacity=(opacities or {}).get(c, 1.0),
                   label=names.get(str(c), f"class {c}"))
    g = to_grid(labels)
    if intensity is not None:
        gi = to_grid(intensity)
        clim = np.percentile(gi.point_data["values"], (1, 99.5))
        if cutaway:
            nz, ny, nx = intensity.shape
            iv = intensity.numpy()
            vz, vy, vx = (v / 1e3 for v in intensity.voxel_size.as_tuple())
            # three cut faces of the removed octant
            for plane in _cut_faces(iv, (nz // 2, ny // 2, nx // 2), (vz, vy, vx)):
                p.add_mesh(plane, scalars="values", cmap="gray", clim=clim,
                           show_scalar_bar=False, lighting=False)
        else:
            p.add_mesh(gi.slice_orthogonal(), cmap="gray", clim=clim, show_scalar_bar=False,
                       lighting=False)
    if legend:
        p.add_legend(bcolor=None, face="circle", size=(0.18, 0.12), loc="upper right")
    if title:
        p.add_text(title, font_size=14, color="white", position="upper_left")
    return _finish(p, g.bounds, out, camera)


def _cut_faces(iv, centre, spacing_um):
    """Planes showing data on the three faces exposed by the cut-away octant."""
    cz, cy, cx = centre
    vz, vy, vx = spacing_um
    nz, ny, nx = iv.shape
    planes = []
    o = (cx * vx, cy * vy, cz * vz)
    planes.append(_plane(iv[cz, cy:, cx:], o, du=(vx, 0, 0), dv=(0, vy, 0)))   # z = cz
    planes.append(_plane(iv[cz:, cy, cx:], o, du=(vx, 0, 0), dv=(0, 0, vz)))   # y = cy
    planes.append(_plane(iv[cz:, cy:, cx], o, du=(0, vy, 0), dv=(0, 0, vz)))   # x = cx
    return planes


def _plane(img2d, origin, du, dv):
    """2D array (rows = v, cols = u) as a textured structured grid."""
    nv, nu = img2d.shape
    u, v = np.meshgrid(np.arange(nu), np.arange(nv))
    pts = (np.asarray(origin)[None, None, :] + u[..., None] * np.asarray(du)
           + v[..., None] * np.asarray(dv))
    g = pv.StructuredGrid(pts[..., 0], pts[..., 1], pts[..., 2])
    g.point_data["values"] = img2d.ravel(order="F")
    return g


def render_objects(objects: Volume, values: Optional[dict] = None, cmap: str = "viridis",
                   clim: Optional[Sequence[float]] = None, label: str = "",
                   max_objects: int = 3000, context: Optional[Volume] = None,
                   out: Optional[str] = None, size=(1600, 1200),
                   camera: Optional[dict] = None, title: str = ""):
    """Each object as its own surface, coloured by ``values[label_id]``."""
    lab = objects.numpy()
    vs = objects.voxel_size.as_tuple()
    sl = ndi.find_objects(lab)
    meshes = []
    for i, s in enumerate(sl[:max_objects], 1):
        if s is None:
            continue
        sub = lab[s] == i
        m = mask_surface(sub, vs, offset_vox=[q.start for q in s], smooth=0.8)
        if m is None or m.n_points == 0:
            continue
        m.point_data["value"] = np.full(m.n_points,
                                        (values or {}).get(i, i), np.float32)
        meshes.append(m)
    p = _plotter(size)
    if meshes:
        merged = pv.merge(meshes)
        vals = merged.point_data["value"]
        p.add_mesh(merged, scalars="value", cmap=cmap,
                   clim=clim or np.percentile(vals, (2, 98)), smooth_shading=True,
                   specular=0.3, scalar_bar_args={"title": label, "color": "white",
                                                  "vertical": True, "position_x": 0.88,
                                                  "position_y": 0.2, "height": 0.6,
                                                  "title_font_size": 22,
                                                  "label_font_size": 18})
    g = to_grid(objects)
    if context is not None:
        s = mask_surface(context.numpy() > 0, context.voxel_size.as_tuple(), decimate=0.6)
        if s is not None:
            p.add_mesh(s, color="#c8d0dc", opacity=0.08)
    if title:
        p.add_text(title, font_size=14, color="white", position="upper_left")
    return _finish(p, g.bounds, out, camera)


def turntable(build: Callable[[], pv.Plotter], out: str, n_frames: int = 180,
              fps: int = 30, elevation_wobble: float = 8.0) -> str:
    """Orbit 360° around the scene produced by ``build()`` and write MP4/GIF."""
    import imageio.v2 as iio

    p = build()
    base_el = p.camera.elevation
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(out, fps=fps, quality=8, macro_block_size=8) as w:
        for k in range(n_frames):
            p.camera.azimuth = p.camera.azimuth + 360.0 / n_frames
            p.camera.elevation = base_el + elevation_wobble * np.sin(2 * np.pi * k / n_frames)
            p.render()
            w.append_data(p.screenshot(return_img=True))
    p.close()
    return out


__all__ = ["to_grid", "mask_surface", "render_volume", "phase_surfaces", "render_phases",
           "render_objects", "turntable", "PALETTE"]

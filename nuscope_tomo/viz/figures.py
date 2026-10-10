"""2D figure panels (matplotlib) with physical aspect ratios and scale bars."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from ..core.volume import Volume  # noqa: E402
from .render import PALETTE  # noqa: E402

STYLE = {
    "figure.facecolor": "#0b0f17", "axes.facecolor": "#0b0f17", "savefig.facecolor": "#0b0f17",
    "text.color": "#e6e9ef", "axes.labelcolor": "#c9cfda", "xtick.color": "#9aa3b2",
    "ytick.color": "#9aa3b2", "axes.edgecolor": "#3a4354", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.grid": False,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
}


def label_cmap(n: int, alpha: float = 0.45) -> ListedColormap:
    from matplotlib.colors import to_rgba
    cols = [(0, 0, 0, 0)] + [to_rgba(PALETTE[i % len(PALETTE)], alpha) for i in range(n)]
    return ListedColormap(cols)


def scale_bar(ax, px_nm: float, width_px: int, color="white"):
    span_nm = width_px * px_nm * 0.25
    nice = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    L = nice[nice <= span_nm].max() if (nice <= span_nm).any() else span_nm
    x1 = width_px * 0.96
    y = ax.get_ylim()[0] * 0.95 if ax.get_ylim()[0] > ax.get_ylim()[1] else ax.get_ylim()[1] * 0.95
    ax.plot([x1 - L / px_nm, x1], [y, y], color=color, lw=3, solid_capstyle="butt")
    ax.text(x1 - L / px_nm / 2, y, f"{L / 1000:g} µm" if L >= 1000 else f"{L:g} nm",
            color=color, ha="center", va="bottom", fontsize=10, fontweight="bold")


def ortho_views(vol: Volume, labels: Optional[Volume] = None, pos: Optional[Sequence[int]] = None,
                title: str = "", out: Optional[str] = None, clim=None, alpha: float = 0.45):
    """XY section plus XZ / YZ strips (Z running left→right) in true physical
    proportions, with an optional label overlay."""
    nz, ny, nx = vol.shape
    pos = pos or (nz // 2, ny // 2, nx // 2)
    vz, vy, vx = vol.voxel_size.as_tuple()
    secs = [("XY", vol.get_slice(0, pos[0]), vx, vy),
            ("XZ", vol.get_slice(1, pos[1]).T, vz, vx),
            ("YZ", vol.get_slice(2, pos[2]).T, vz, vy)]
    labs = None
    if labels is not None:
        labs = [labels.get_slice(0, pos[0]), labels.get_slice(1, pos[1]).T,
                labels.get_slice(2, pos[2]).T]
    ncls = int(labels.data.max().compute()) if labels is not None else 0
    clim = clim or np.percentile(secs[0][1], (1, 99.5))
    ext = [(im.shape[1] * sx, im.shape[0] * sy) for _, im, sx, sy in secs]   # (w, h) nm
    with plt.rc_context(STYLE):
        W = ext[0][0] + ext[1][0]
        H = max(ext[0][1], ext[1][1] + ext[2][1])
        fig = plt.figure(figsize=(16, 16 * H / W * 1.08))
        gs = fig.add_gridspec(2, 2, width_ratios=[ext[0][0], ext[1][0]],
                              height_ratios=[ext[1][1], ext[2][1]], wspace=0.04, hspace=0.12)
        axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
        for i, (ax, (name, img, sx, sy)) in enumerate(zip(axes, secs)):
            ax.imshow(img, cmap="gray", vmin=clim[0], vmax=clim[1], aspect=sy / sx,
                      interpolation="lanczos")
            if labs is not None:
                ax.imshow(labs[i], cmap=label_cmap(ncls, alpha), vmin=0, vmax=ncls,
                          aspect=sy / sx, interpolation="nearest")
            hint = "" if i == 0 else "   Z (milling) →"
            ax.set_title(f"{name}  ({['z', 'y', 'x'][i]} = {pos[i]}){hint}", loc="left")
            ax.set_xticks([]), ax.set_yticks([])
            scale_bar(ax, sx, img.shape[1])
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
        if out:
            fig.savefig(out, dpi=150, bbox_inches="tight")
        return fig


def compare(images: dict[str, np.ndarray], out: Optional[str] = None, clim=None,
            zoom: Optional[tuple] = None, title: str = "", cmap="gray"):
    """Side-by-side panels (e.g. raw → corrected), optional zoomed row."""
    n = len(images)
    rows = 2 if zoom else 1
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, n, figsize=(4.6 * n, 3.9 * rows), squeeze=False)
        for j, (name, img) in enumerate(images.items()):
            c = clim or np.percentile(img, (1, 99.5))
            axes[0, j].imshow(img, cmap=cmap, vmin=c[0], vmax=c[1])
            axes[0, j].set_title(name, loc="left")
            if zoom:
                (y0, y1), (x0, x1) = zoom
                axes[0, j].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                                   ec="#E69F00", lw=1.5))
                axes[1, j].imshow(img[y0:y1, x0:x1], cmap=cmap, vmin=c[0], vmax=c[1],
                                  interpolation="nearest")
            for a in axes[:, j]:
                a.set_xticks([]), a.set_yticks([])
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
        fig.tight_layout()
        if out:
            fig.savefig(out, dpi=140, bbox_inches="tight")
        return fig


__all__ = ["ortho_views", "compare", "scale_bar", "label_cmap", "STYLE"]

"""NuScope-TOMO: open FIB-SEM slice-and-view tomography — reconstruction,
segmentation, quantification and correlative EDS chemistry."""
__version__ = "0.2.0"

from .core.volume import Volume, VoxelSize  # noqa: E402

__all__ = ["Volume", "VoxelSize", "__version__"]

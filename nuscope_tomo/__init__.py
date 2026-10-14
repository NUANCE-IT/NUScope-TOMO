"""NUScope-TOMO: open FIB-SEM slice-and-view tomography — reconstruction,
segmentation, quantification and correlative EDS chemistry."""
__version__ = "0.3.2"

from .core.volume import Volume, VoxelSize  # noqa: E402

__all__ = ["Volume", "VoxelSize", "__version__"]

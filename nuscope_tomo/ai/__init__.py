"""Deep learning (optional dependency: ``pip install torch``).

* ``denoise_n2v`` — Noise2Void self-supervised denoising, trained on the
  dataset itself (no clean images needed).
* ``unet_segment`` — 2D U-Net trained from sparse scribbles (masked loss),
  the same annotations the random forest uses.

Runs on Apple Silicon (MPS), CUDA or CPU.
"""
from .unet import UNet2D, device  # noqa: F401
from .n2v import denoise_n2v, train_n2v  # noqa: F401
from .segment import train_unet, unet_segment  # noqa: F401

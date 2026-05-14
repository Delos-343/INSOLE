"""Image preprocessing helpers (independent of pipelines)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_and_orient(path: str | Path) -> Image.Image:
    """Load image and respect EXIF orientation tag."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def square_pad(img: Image.Image, size: int, pad_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Letterbox-pad an image to a square `size x size`."""
    img.thumbnail((size, size), Image.LANCZOS)
    new = Image.new("RGB", (size, size), pad_color)
    new.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return new


def to_numpy(img: Image.Image, normalize: bool = True) -> np.ndarray:
    """PIL.Image -> HxWxC float32 numpy."""
    arr = np.asarray(img, dtype=np.float32)
    if normalize:
        arr = arr / 255.0
    return arr

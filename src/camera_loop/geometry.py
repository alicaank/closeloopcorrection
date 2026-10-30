"""Explicit OpenCV camera-to-world geometry; no trajectory-derived scale fitting."""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates


def poses(value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape[-2:] != (4, 4) or not np.isfinite(a).all():
        raise ValueError("Expected finite (...,4,4) camera poses")
    if not np.allclose(a[..., 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous row")
    r = a[..., :3, :3]
    if (
        np.max(np.abs(np.swapaxes(r, -1, -2) @ r - np.eye(3))) > 1e-3
        or np.max(np.abs(np.linalg.det(r) - 1)) > 1e-3
    ):
        raise ValueError("Camera rotation must be a proper SO(3) matrix")
    return a.copy()


def reanchor(cameras):
    a = poses(cameras)
    if a.ndim != 3 or len(a) == 0:
        raise ValueError("Expected nonempty pose sequence")
    return np.linalg.inv(a[0]) @ a


def basis(axes):
    if axes == "opencv":
        return np.eye(4)
    if axes == "opengl":
        return np.diag([1.0, -1.0, -1.0, 1.0])
    raise ValueError(
        "Camera axes must be opencv or opengl; supply an explicit conversion for other conventions"
    )


@dataclass(frozen=True)
class PixelTransform:
    """Pixel-center resize/crop applied identically to RGB, depth, and intrinsics.

    `crop` is (left, top, width, height) in the source image. Sizes are (width,height).
    Native libraries using a different pixel-center convention must supply their own transform.
    """

    source_size: tuple[int, int]
    output_size: tuple[int, int]
    crop: tuple[int, int, int, int] | None = None

    def __post_init__(self):
        w, h = self.source_size
        ow, oh = self.output_size
        x, y, cw, ch = self.region
        if min(w, h, ow, oh, cw, ch) <= 0 or x < 0 or y < 0 or x + cw > w or y + ch > h:
            raise ValueError("Invalid crop or image size")

    @property
    def region(self):
        return self.crop or (0, 0, *self.source_size)

    @property
    def matrix(self):
        x, y, w, h = self.region
        sx = self.output_size[0] / w
        sy = self.output_size[1] / h
        return np.array(
            [[sx, 0, sx * (0.5 - x) - 0.5], [0, sy, sy * (0.5 - y) - 0.5], [0, 0, 1.0]]
        )

    def intrinsics(self, K):
        k = np.asarray(K, dtype=float)
        if k.shape != (3, 3) or not np.isfinite(k).all():
            raise ValueError("Expected finite 3x3 intrinsics")
        return self.matrix @ k

    def depth(self, depth):
        d = np.asarray(depth, dtype=float)
        w, h = self.source_size
        if d.shape != (h, w):
            raise ValueError("Depth dimensions do not match source RGB")
        ow, oh = self.output_size
        x, y, cw, ch = self.region
        yy, xx = np.mgrid[:oh, :ow]
        coordinates = [
            np.clip(y + (yy + 0.5) * ch / oh - 0.5, y, y + ch - 1),
            np.clip(x + (xx + 0.5) * cw / ow - 0.5, x, x + cw - 1),
        ]
        valid = np.isfinite(d) & (d > 0)
        values = map_coordinates(
            np.where(valid, d, 0), coordinates, order=1, mode="nearest"
        )
        weights = map_coordinates(
            valid.astype(float), coordinates, order=1, mode="nearest"
        )
        return np.where(weights > 0.999, values / np.maximum(weights, 1e-12), np.nan)

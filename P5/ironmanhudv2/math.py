from dataclasses import dataclass

import numpy as np

from ironmanhudv2.units import Size

def euclid_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def exp_moving_avg(prev: float | None, curr: float, alpha: float):
    """
    Computes the exponential moving average of the current value and the previous value.
    If the previous value is None, it returns the current value.
    
    Smooth a time series by giving more weight
    to recent observations and making older ones less relevant.
    """
    
    if prev is None:
        return curr
    
    return alpha * curr + (1.0 - alpha) * prev

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

def compute_bbox(pts_2d: np.ndarray, size: Size) -> BBox:
    x1 = int(np.clip(np.min(pts_2d[:, 0]), 0, size.width - 1))
    y1 = int(np.clip(np.min(pts_2d[:, 1]), 0, size.height - 1))
    x2 = int(np.clip(np.max(pts_2d[:, 0]), 0, size.width - 1))
    y2 = int(np.clip(np.max(pts_2d[:, 1]), 0, size.height - 1))
    
    return BBox(x1, y1, x2, y2)
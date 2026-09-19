"""CPU mask thinning with identical output and less work on white-background images."""
import numpy as np
from skimage.morphology import thin


def thin_foreground(mask):
    foreground = np.asarray(mask) > 0
    result = np.zeros(foreground.shape, dtype=np.uint8)
    ys, xs = np.nonzero(foreground)
    if not len(xs):
        return result
    # A one-pixel zero border preserves the boundary conditions of full-image thin().
    y0, y1 = max(0, ys.min() - 1), min(foreground.shape[0], ys.max() + 2)
    x0, x1 = max(0, xs.min() - 1), min(foreground.shape[1], xs.max() + 2)
    result[y0:y1, x0:x1] = thin(foreground[y0:y1, x0:x1]).astype(np.uint8) * 255
    return result

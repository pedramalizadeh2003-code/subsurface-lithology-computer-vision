"""
Module: curve_digitizer
Isolates continuous sensor signal trajectories from graphical well logs,
applies median centroid tracking, handles occlusions via interpolation, and calibrates engineering units.
"""

from typing import Tuple
import cv2
import numpy as np


def extract_blue_curve_trajectory(curve_img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the median horizontal coordinate x for each vertical scanline y
    representing the blue sensor trace.
    """
    h, w = curve_img_bgr.shape[:2]
    b, g, r = cv2.split(curve_img_bgr)

    # Blue dominant color mask
    mask_blue = (b > 120) & (r < 50) & (g < 50)

    x_by_y = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        xs = np.where(mask_blue[y] > 0)[0]
        if xs.size > 0:
            x_by_y[y] = np.median(xs)

    valid = np.where(~np.isnan(x_by_y))[0]
    if len(valid) == 0:
        return np.arange(h), np.zeros(h, dtype=np.float32)

    ymin, ymax = valid.min(), valid.max()
    y_range = np.arange(ymin, ymax + 1)

    # 1D Linear Interpolation for occluded/missing rows
    x_interp = np.interp(y_range, valid, x_by_y[valid])
    return y_range, x_interp


def calibrate_and_resample_curve(
    x_interp: np.ndarray,
    img_width: int,
    step_samples: float,
    depth_start: int = 50,
    depth_step: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsamples the pixel trajectory into discrete depth intervals and normalizes values to [0, 100].
    """
    n_bins = int(np.ceil(len(x_interp) / step_samples))
    depth_labels = []
    values = []

    for i in range(n_bins):
        d0 = depth_start + depth_step * i
        d1 = d0 + depth_step
        depth_labels.append(f"{d0}-{d1}")

        start = int(round(i * step_samples))
        end = int(start + step_samples)
        start = max(0, min(start, len(x_interp)))
        end = max(0, min(end, len(x_interp)))

        sl = x_interp[start:end]
        val = float(np.mean(sl)) if sl.size else 0.0

        # Mapping: Inverted scale (0 to 100)
        mapped = 100.0 - (100.0 * val / float(img_width))
        values.append(float(np.clip(mapped, 0.0, 100.0)))

    return np.array(depth_labels), np.array(values)

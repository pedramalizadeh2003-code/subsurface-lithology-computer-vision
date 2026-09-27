"""
Module: symbol_classifier
Identifies geological lithology rock symbols using HSV color-space background/foreground segmentation,
binary morphological feature analysis, and directional transition profiling.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

# HSV Color Threshold Constants
V_BLACK_MAX = 60
S_YELLOW_MIN = 70
V_YELLOW_MIN = 120
H_YELLOW_RANGE = (15, 40)

H_BLUE_RANGE = (85, 135)
H_GREEN_RANGE = (35, 85)
H_RED_LOW = (0, 10)
H_RED_HIGH = (170, 179)
V_DARK_MAX = 70


def dominant_hsv(patch_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    """
    Computes robust median HSV values across an image patch, optionally constrained by a mask.
    Handles circular Hue wrapping for red spectral components.
    """
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3) if mask is None else hsv[mask > 0].reshape(-1, 3)

    if pixels.shape[0] == 0:
        return (0, 0, 0)

    # Hue red wraparound adjustment
    red_mask = (pixels[:, 0] >= 170) & (pixels[:, 0] <= 179)
    pixels[red_mask, 0] = 9 - (pixels[red_mask, 0] - 170)

    h = int(np.median(pixels[:, 0]))
    s = int(np.median(pixels[:, 1]))
    v = int(np.median(pixels[:, 2]))
    return (h, s, v)


def classify_background(cell_bgr: np.ndarray) -> str:
    """
    Classifies patch background as 'black', 'yellow', or 'white'.
    """
    h, s, v = dominant_hsv(cell_bgr)
    if v <= V_BLACK_MAX:
        return "black"
    if (H_YELLOW_RANGE[0] <= h <= H_YELLOW_RANGE[1]) and (s >= S_YELLOW_MIN) and (v >= V_YELLOW_MIN):
        return "yellow"
    return "white"


def foreground_mask(cell_bgr: np.ndarray, bg: str) -> np.ndarray:
    """
    Extracts foreground stroke mask based on detected background color.
    """
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)

    if bg == "black":
        return (hsv[:, :, 2] > (V_BLACK_MAX + 20)).astype(np.uint8) * 255

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if bg == "yellow":
        yellow_bg = ((h >= H_YELLOW_RANGE[0]) & (h <= H_YELLOW_RANGE[1]) & (s >= 40) & (v >= 80))
        return (~yellow_bg).astype(np.uint8) * 255

    # White background
    white_bg = (s < 150) & (v > 200)
    return (~white_bg).astype(np.uint8) * 255


def classify_foreground_color(cell_bgr: np.ndarray, bg: str) -> str:
    """
    Determines stroke color (black, red, blue, green) of symbol patterns.
    """
    mask = foreground_mask(cell_bgr, bg)
    if np.sum(mask) == 0:
        return "no_foreground"

    h, s, v = dominant_hsv(cell_bgr, mask=mask)

    if v <= V_DARK_MAX or (v <= 2 * V_DARK_MAX and s <= 50):
        return "black"
    if (H_RED_LOW[0] <= h <= H_RED_LOW[1]) or (H_RED_HIGH[0] <= h <= H_RED_HIGH[1]):
        return "red"
    if H_BLUE_RANGE[0] <= h <= H_BLUE_RANGE[1]:
        return "blue"
    if H_GREEN_RANGE[0] <= h <= H_GREEN_RANGE[1]:
        return "green"

    return "unknown"


def binarize_foreground(cell_bgr: np.ndarray, bg: str) -> np.ndarray:
    """
    Converts cell to clean binary image with foreground as 1.
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    if bg == "white":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif bg == "yellow":
        th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, bw = cv2.threshold(gray, int(0.5 * th), 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    return (bw > 0).astype(np.uint8)


def max_line_transitions(binary_img: np.ndarray, direction: str = "horizontal") -> int:
    """
    Calculates the maximum count of 0-to-1 pixel transitions across scanlines.
    """
    img = (binary_img > 0).astype(np.int16)
    if direction == "horizontal":
        transitions = np.abs(np.diff(img, axis=1))
        per_line = np.sum(transitions, axis=1)
    elif direction == "vertical":
        transitions = np.abs(np.diff(img, axis=0))
        per_line = np.sum(transitions, axis=0)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")

    return int(np.max(per_line)) if len(per_line) > 0 else 0


def classify_symbol(cell_bgr: np.ndarray) -> str:
    """
    Classifies a single well-log lithology cell into one of 16 geological rock classes.
    """
    bg = classify_background(cell_bgr)

    # 1. Black background
    if bg == "black":
        return "OBSIDIAN"

    b01 = binarize_foreground(cell_bgr, bg)

    # 2. Yellow background
    if bg == "yellow":
        t = max_line_transitions(b01, direction="vertical")
        if t == 2:
            area = int(b01.sum())
            if area < 80:
                return "SAND"
            elif area < 150:
                return "MEDIUM SAND"
            else:
                return "COARSE SAND"
        elif t == 4:
            return "CONGLOMERATE"

    # 3. White background
    fg_color = classify_foreground_color(cell_bgr, bg="white")
    if fg_color == "no_foreground":
        return "blank"

    if fg_color == "black":
        top_crop = b01[:max(1, b01.shape[0] // 5), :]
        t = max_line_transitions(top_crop, direction="horizontal")
        return "DIORITE" if t == 0 else "CEMENT"

    if fg_color == "blue":
        t = max_line_transitions(b01, direction="vertical")
        return "MONZONITE" if t == 2 else "DACITE"

    if fg_color == "red":
        t = max_line_transitions(b01, direction="horizontal")
        return "GRANODIORITE" if t == 2 else "GRANITE"

    if fg_color == "green":
        t = max_line_transitions(b01, direction="horizontal")
        if t < 4:
            t2 = max_line_transitions(b01, direction="vertical")
            if t2 == 2:
                return "CLAY"
            if t2 == 4:
                return "CLAYSTONE"
        elif t == 4:
            return "RHYOLITE"
        else:
            return "GRANITE WASH"

    return "UNKNOWN"

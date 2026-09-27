"""
Module: document_layout
Performs projection profile line detection, morphological table grid extraction,
and column/row boundary delineation for well-logging documents.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


def find_next_line_boundaries(
    bw_normalized: np.ndarray,
    prev_line_end_boundary: int,
    is_horizontal: bool = True
) -> Tuple[Optional[int], Optional[int]]:
    """
    Scans a binary normalized image (0/1) for solid continuous grid line boundaries
    using directional 1D projection profiling.
    """
    processing_array = bw_normalized if is_horizontal else np.transpose(bw_normalized)
    total_dimension = processing_array.shape[0]

    search_start_index = prev_line_end_boundary + 1
    if search_start_index >= total_dimension:
        return None, None

    in_line = False
    start_boundary = None

    for current_boundary in range(search_start_index, total_dimension):
        is_solid_black = np.all(processing_array[current_boundary])

        if is_solid_black and not in_line:
            in_line = True
            start_boundary = current_boundary
        elif not is_solid_black and in_line:
            return start_boundary, current_boundary - 1

    if in_line:
        return start_boundary, total_dimension - 1

    return None, None


def extract_table_mask(img: np.ndarray) -> np.ndarray:
    """
    Applies directional morphological structuring elements to isolate horizontal
    and vertical grid lines from text and lithological symbology.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    bw_normalized = (bw > 0).astype(np.uint8)

    # Horizontal mask
    k_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_h = cv2.morphologyEx(bw_normalized, cv2.MORPH_DILATE, k_rect, iterations=1)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    mask_h = cv2.morphologyEx(mask_h, cv2.MORPH_ERODE, k_h, iterations=10)
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    mask_h = cv2.morphologyEx(mask_h, cv2.MORPH_ERODE, k_v, iterations=1)
    mask_h = cv2.morphologyEx(mask_h, cv2.MORPH_DILATE, k_rect, iterations=1)

    # Vertical mask
    mask_v = cv2.morphologyEx(bw_normalized, cv2.MORPH_DILATE, k_rect, iterations=1)
    mask_v = cv2.morphologyEx(mask_v, cv2.MORPH_ERODE, k_v, iterations=15)

    return cv2.bitwise_or(mask_v, mask_h)


def extract_column_boundaries(mask_top_rows: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extracts all vertical column coordinate pairs across the table header row.
    """
    boundaries = []
    _, start_col = find_next_line_boundaries(mask_top_rows, -1, is_horizontal=False)

    while True:
        end_col, next_start = find_next_line_boundaries(mask_top_rows, start_col, is_horizontal=False)
        if start_col is not None and end_col is not None:
            boundaries.append((start_col, end_col))
            start_col = next_start
        else:
            break

    return boundaries

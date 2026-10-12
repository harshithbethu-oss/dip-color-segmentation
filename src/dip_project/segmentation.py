"""Color-based segmentation utilities for each color space."""

from __future__ import annotations

import numpy as np


def _in_range(values: np.ndarray, lower: int, upper: int, wrap: bool = False) -> np.ndarray:
    """Create a boolean mask based on a simple numeric range."""
    if wrap:
        return (values >= lower) | (values <= upper)
    return (values >= lower) & (values <= upper)


def segment_rgb(image_rgb: np.ndarray, lower_r: tuple[int, int] = (0, 255), lower_g: tuple[int, int] = (0, 255), lower_b: tuple[int, int] = (0, 255)) -> np.ndarray:
    """Segment an RGB image using independent threshold ranges on R, G and B."""
    r_values = image_rgb[:, :, 0]
    g_values = image_rgb[:, :, 1]
    b_values = image_rgb[:, :, 2]

    mask = (
        _in_range(r_values, lower_r[0], lower_r[1])
        & _in_range(g_values, lower_g[0], lower_g[1])
        & _in_range(b_values, lower_b[0], lower_b[1])
    )
    return mask.astype(np.uint8) * 255


def segment_hsv(image_hsv: np.ndarray, h_range: tuple[int, int] = (0, 179), s_range: tuple[int, int] = (0, 255), v_range: tuple[int, int] = (0, 255), wrap_hue: bool = False) -> np.ndarray:
    """Segment an HSV image using hue, saturation and value thresholds."""
    h_values = image_hsv[:, :, 0]
    s_values = image_hsv[:, :, 1]
    v_values = image_hsv[:, :, 2]

    hue_mask = _in_range(h_values, h_range[0], h_range[1], wrap=wrap_hue)
    sat_mask = _in_range(s_values, s_range[0], s_range[1])
    val_mask = _in_range(v_values, v_range[0], v_range[1])
    mask = hue_mask & sat_mask & val_mask
    return mask.astype(np.uint8) * 255


def segment_hsi(image_hsi: np.ndarray, h_range: tuple[int, int] = (0, 360), s_range: tuple[int, int] = (0, 100), i_range: tuple[int, int] = (0, 100), wrap_hue: bool = False) -> np.ndarray:
    """Segment an HSI image using hue, saturation and intensity thresholds."""
    h_values = image_hsi[:, :, 0]
    s_values = image_hsi[:, :, 1]
    i_values = image_hsi[:, :, 2]

    hue_mask = _in_range(h_values, h_range[0], h_range[1], wrap=wrap_hue)
    sat_mask = _in_range(s_values, s_range[0], s_range[1])
    inten_mask = _in_range(i_values, i_range[0], i_range[1])
    mask = hue_mask & sat_mask & inten_mask
    return mask.astype(np.uint8) * 255


def segment_ycbcr(image_ycbcr: np.ndarray, y_range: tuple[int, int] = (0, 255), cb_range: tuple[int, int] = (0, 255), cr_range: tuple[int, int] = (0, 255)) -> np.ndarray:
    """Segment a YCbCr image using luminance and chrominance ranges."""
    y_values = image_ycbcr[:, :, 0]
    cb_values = image_ycbcr[:, :, 1]
    cr_values = image_ycbcr[:, :, 2]

    mask = (
        _in_range(y_values, y_range[0], y_range[1])
        & _in_range(cb_values, cb_range[0], cb_range[1])
        & _in_range(cr_values, cr_range[0], cr_range[1])
    )
    return mask.astype(np.uint8) * 255


def segment_image(color_space: str, image_color: np.ndarray, ranges: dict) -> np.ndarray:
    """Dispatch to the correct segmentation implementation."""
    space = color_space.upper()
    if space == "RGB":
        return segment_rgb(image_color, ranges["R"], ranges["G"], ranges["B"])
    if space == "HSV":
        return segment_hsv(image_color, ranges["H"], ranges["S"], ranges["V"], wrap_hue=ranges.get("wrap_hue", False))
    if space == "HSI":
        return segment_hsi(image_color, ranges["H"], ranges["S"], ranges["I"], wrap_hue=ranges.get("wrap_hue", False))
    if space in {"YCBCR", "YCbCr"}:
        return segment_ycbcr(image_color, ranges["Y"], ranges["Cb"], ranges["Cr"])
    raise ValueError(f"Unsupported segmentation space: {color_space}")


SEGMENTATION_DEFAULTS = {
    "RGB": {"R": (0, 255), "G": (0, 255), "B": (0, 255)},
    "HSV": {"H": (0, 179), "S": (0, 255), "V": (0, 255), "wrap_hue": False},
    "HSI": {"H": (0, 360), "S": (0, 100), "I": (0, 100), "wrap_hue": False},
    "YCbCr": {"Y": (0, 255), "Cb": (0, 255), "Cr": (0, 255)},
}

"""Color-space conversion utilities.

This module keeps the image processing logic explicit and channel-safe.
It accepts either RGB or OpenCV-style BGR input, converts to RGB internally,
and then performs the requested conversion without channel-order mistakes.

Channel ranges used by this project:
- RGB: R, G, B in [0, 255]
- HSV (OpenCV): H in [0, 179], S in [0, 255], V in [0, 255]
- HSI (custom): H in [0, 360) degrees, S and I in [0, 100]
- YCbCr (OpenCV): Y, Cb, Cr in [0, 255] for the full 8-bit output range
"""

from __future__ import annotations

import cv2
import numpy as np


def _ensure_rgb_image(image: np.ndarray, source_order: str = "RGB") -> np.ndarray:
    """Return an RGB image while tolerating either RGB or OpenCV BGR input."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a 3-channel array with shape (H, W, 3).")

    order = (source_order or "RGB").upper()
    image_u8 = image.astype(np.uint8, copy=False)
    if order == "RGB":
        return image_u8
    if order == "BGR":
        return cv2.cvtColor(image_u8, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image channel order: {source_order!r}. Use 'RGB' or 'BGR'.")


def _canonical_color_space_name(color_space: str) -> str:
    """Normalize alternate color-space names to the canonical project labels."""
    key = str(color_space).upper().replace(" ", "")
    aliases = {
        "YCBCR": "YCbCr",
        "YCRCB": "YCbCr",
        "YCRCB": "YCbCr",
        "YCRCb": "YCbCr",
        "YCBcr": "YCbCr",
        "YCRCB": "YCbCr",
    }
    return aliases.get(key, key)


def rgb_to_hsi(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to HSI using the standard formulation.

    For each pixel,
    - I = (R + G + B) / 3
    - S = 1 - 3 * min(R, G, B) / (R + G + B) when the intensity is non-zero
    - H is computed from the angle between the red and green axes and mapped to
      degrees in [0, 360)

    This implementation explicitly handles:
    - grayscale pixels: H = 0, S = 0
    - zero intensity: saturation is forced to 0
    - hue wrapping at 360 degrees via modulo arithmetic
    """
    rgb = _ensure_rgb_image(image_rgb, source_order="RGB")
    rgb_norm = rgb.astype(np.float32) / 255.0
    r = rgb_norm[:, :, 0]
    g = rgb_norm[:, :, 1]
    b = rgb_norm[:, :, 2]

    intensity = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    sum_rgb = r + g + b

    saturation = np.zeros_like(r, dtype=np.float32)
    valid = sum_rgb > 1e-8
    saturation[valid] = 1.0 - (3.0 * min_rgb[valid] / sum_rgb[valid])

    hue_deg = np.zeros_like(r, dtype=np.float32)
    non_gray = saturation > 1e-8
    if np.any(non_gray):
        r_n = r[non_gray]
        g_n = g[non_gray]
        b_n = b[non_gray]

        num = 0.5 * ((r_n - g_n) + (r_n - b_n))
        den = np.sqrt((r_n - g_n) ** 2 + (r_n - b_n) * (g_n - b_n) + 1e-8)
        cosine = np.clip(num / (den + 1e-8), -1.0, 1.0)
        theta = np.arccos(cosine)
        hue_rad = np.where(b_n <= g_n, theta, 2.0 * np.pi - theta)
        hue_deg[non_gray] = np.rad2deg(hue_rad) % 360.0

    hsi = np.zeros_like(rgb_norm, dtype=np.float32)
    hsi[:, :, 0] = hue_deg
    hsi[:, :, 1] = np.clip(saturation * 100.0, 0.0, 100.0)
    hsi[:, :, 2] = np.clip(intensity * 100.0, 0.0, 100.0)
    return hsi


def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to HSV using OpenCV.

    OpenCV uses H in [0, 179], S and V in [0, 255].
    """
    rgb = _ensure_rgb_image(image_rgb, source_order="RGB")
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)


def rgb_to_ycbcr(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to YCbCr using OpenCV and normalize to Y, Cb, Cr ordering.

    OpenCV’s native YCrCb order is Y, Cr, Cb. The project uses the conventional
    YCbCr naming order, so the chroma channels are swapped before returning the image.
    """
    rgb = _ensure_rgb_image(image_rgb, source_order="RGB")
    ycrcb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    return ycrcb[:, :, [0, 2, 1]]


def convert_to_color_space(image: np.ndarray, color_space: str, source_order: str = "RGB") -> np.ndarray:
    """Return the image in the requested color space.

    `source_order` lets the project accept either standard RGB arrays or OpenCV
    BGR arrays captured via cv2.imread without channel-order mistakes.
    """
    space = _canonical_color_space_name(color_space)
    rgb_image = _ensure_rgb_image(image, source_order=source_order)

    if space == "RGB":
        return rgb_image.astype(np.uint8)
    if space == "HSV":
        return rgb_to_hsv(rgb_image)
    if space == "HSI":
        return rgb_to_hsi(rgb_image)
    if space == "YCbCr":
        return rgb_to_ycbcr(rgb_image)
    raise ValueError(f"Unsupported color space: {color_space}")


def get_channel_images(image_color: np.ndarray, color_space: str):
    """Return per-channel images for visualization."""
    space = _canonical_color_space_name(color_space)
    if space == "RGB":
        channels = {"R": image_color[:, :, 0], "G": image_color[:, :, 1], "B": image_color[:, :, 2]}
    elif space == "HSV":
        channels = {"H": image_color[:, :, 0], "S": image_color[:, :, 1], "V": image_color[:, :, 2]}
    elif space == "HSI":
        channels = {"H": image_color[:, :, 0], "S": image_color[:, :, 1], "I": image_color[:, :, 2]}
    elif space == "YCbCr":
        channels = {"Y": image_color[:, :, 0], "Cb": image_color[:, :, 1], "Cr": image_color[:, :, 2]}
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    return channels


COLOR_SPACES = ["RGB", "HSV", "HSI", "YCbCr"]

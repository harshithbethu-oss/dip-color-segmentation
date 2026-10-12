"""Utility helpers for image loading, sample generation and export."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def generate_sample_image(sample_name: str) -> np.ndarray:
    """Generate an in-memory sample image for demonstration and testing."""
    height, width = 300, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    if sample_name == "Red circle":
        cv2.circle(image, (150, 150), 90, (0, 0, 255), -1)
        cv2.rectangle(image, (50, 30), (100, 90), (0, 255, 0), -1)
    elif sample_name == "Green square":
        cv2.rectangle(image, (60, 60), (240, 240), (0, 255, 0), -1)
    elif sample_name == "Color blocks":
        image[:, :100] = (255, 0, 0)
        image[:, 100:200] = (0, 255, 0)
        image[:, 200:] = (0, 0, 255)
    elif sample_name == "Multi-object scene":
        cv2.circle(image, (90, 100), 55, (0, 0, 255), -1)
        cv2.rectangle(image, (140, 120), (240, 220), (0, 255, 0), -1)
        cv2.circle(image, (220, 70), 35, (255, 0, 0), -1)
    else:
        cv2.circle(image, (150, 150), 80, (255, 0, 0), -1)
    return image


def load_uploaded_image(uploaded_file) -> np.ndarray:
    """Load a JPG/JPEG/PNG file and return a NumPy RGB image."""
    if uploaded_file is None:
        raise ValueError("No image file was uploaded.")

    try:
        with Image.open(uploaded_file) as img:
            rgb = img.convert("RGB")
            return np.asarray(rgb)
    except Exception as exc:
        raise ValueError("Could not read image. Please upload a valid JPG, JPEG or PNG file.") from exc


def load_uploaded_mask(uploaded_file) -> np.ndarray:
    """Load a ground-truth mask as a grayscale image."""
    if uploaded_file is None:
        return None

    try:
        with Image.open(uploaded_file) as img:
            gray = img.convert("L")
            return np.asarray(gray)
    except Exception as exc:
        raise ValueError("Could not read the ground-truth mask. Please upload a valid image file.") from exc


def align_mask_to_image(mask: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int]) -> np.ndarray:
    """Align a reference mask to the target image geometry without using bilinear resizing.

    This helper is intentionally strict: if the aspect ratio differs, the mask cannot be
    aligned confidently and a ValueError is raised so the UI can warn and skip metric
    calculation instead of silently computing invalid values.
    """
    if mask is None:
        raise ValueError("No ground-truth mask was supplied.")

    arr = np.asarray(mask)
    if arr.size == 0:
        raise ValueError("Ground-truth mask is empty.")

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        elif arr.shape[-1] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError("Ground-truth mask must be grayscale or RGB/RGBA.")

    if arr.ndim != 2:
        raise ValueError("Ground-truth mask must be a 2D array after conversion.")

    target_h, target_w = image_shape[:2]
    if arr.shape == (target_h, target_w):
        return arr.astype(np.uint8)

    mask_h, mask_w = arr.shape
    if mask_h <= 0 or mask_w <= 0:
        raise ValueError("Ground-truth mask has an invalid shape.")

    ratio_mask = mask_h / mask_w
    ratio_target = target_h / target_w
    tol = 1e-3
    if abs(ratio_mask - ratio_target) <= tol:
        resized = cv2.resize(arr.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return resized

    if abs((mask_w / mask_h) - (target_w / target_h)) <= tol:
        arr = arr.T
        resized = cv2.resize(arr.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return resized

    raise ValueError(
        f"Mask geometry cannot be aligned safely: predicted image size {target_h}x{target_w} does not match reference mask {mask_h}x{mask_w}."
    )


def prepare_mask_for_download(mask: np.ndarray) -> bytes:
    """Convert a binary mask to PNG bytes for download."""
    array = np.asarray(mask)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    encoded = cv2.imencode(".png", array.astype(np.uint8))[1]
    return encoded.tobytes()


def prepare_segmented_image_for_download(image_rgb: np.ndarray) -> bytes:
    """Convert an RGB image to PNG bytes for download."""
    encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))[1]
    return encoded.tobytes()


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    """Normalize a mask to 0/255 grayscale."""
    mask_arr = np.asarray(mask)
    if mask_arr.dtype != np.uint8:
        mask_arr = mask_arr.astype(np.uint8)
    if mask_arr.max() <= 1:
        mask_arr = mask_arr * 255
    return mask_arr.astype(np.uint8)

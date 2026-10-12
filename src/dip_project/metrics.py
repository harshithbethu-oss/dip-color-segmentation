"""Evaluation metrics for segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a mask image to a 2D binary array where foreground = 1."""
    if mask is None:
        raise ValueError("Mask input cannot be None.")

    arr = np.asarray(mask)
    if arr.size == 0:
        raise ValueError("Mask input cannot be empty.")
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[-1] in (3, 4):
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY if arr.shape[-1] == 3 else cv2.COLOR_RGBA2GRAY)
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D after pre-processing, received shape {arr.shape}.")
    arr = arr.astype(np.float32)
    return (arr > 0).astype(np.uint8)


def _align_mask_pair(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and align masks when their dimensions differ but share the same geometry."""
    pred = _to_binary_mask(pred_mask)
    gt = _to_binary_mask(gt_mask)

    if pred.shape == gt.shape:
        return pred, gt

    pred_h, pred_w = pred.shape
    gt_h, gt_w = gt.shape
    if pred_h <= 0 or pred_w <= 0 or gt_h <= 0 or gt_w <= 0:
        raise ValueError("Mask shapes are invalid.")

    pred_ratio = pred_h / pred_w
    gt_ratio = gt_h / gt_w
    tol = 1e-3
    if abs(pred_ratio - gt_ratio) <= tol:
        gt = cv2.resize(gt.astype(np.uint8), (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)
        return pred, gt.astype(np.uint8)

    if abs(pred_ratio - (gt_w / gt_h)) <= tol:
        gt = gt.T
        gt = cv2.resize(gt.astype(np.uint8), (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)
        return pred, gt.astype(np.uint8)

    raise ValueError(
        f"Mask shapes do not match and cannot be aligned safely: {pred.shape} != {gt.shape}."
    )


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute the Intersection-over-Union metric."""
    pred, gt = _align_mask_pair(pred_mask, gt_mask)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute the Dice coefficient."""
    pred, gt = _align_mask_pair(pred_mask, gt_mask)
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 0.0
    return float((2.0 * intersection) / total)


def compute_pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute overall pixel accuracy."""
    pred, gt = _align_mask_pair(pred_mask, gt_mask)
    correct = np.equal(pred, gt).sum()
    total = pred.size
    if total == 0:
        return 0.0
    return float(correct / total)


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Return IoU, Dice and pixel accuracy for a predicted mask."""
    pred, gt = _align_mask_pair(pred_mask, gt_mask)
    return {
        "IoU": compute_iou(pred, gt),
        "Dice": compute_dice(pred, gt),
        "Pixel Accuracy": compute_pixel_accuracy(pred, gt),
    }

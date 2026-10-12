import numpy as np
import pytest

from src.dip_project.metrics import compute_dice, compute_iou, compute_pixel_accuracy


def test_metrics_return_one_for_identical_masks():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255

    iou = compute_iou(mask, mask)
    dice = compute_dice(mask, mask)
    accuracy = compute_pixel_accuracy(mask, mask)

    assert iou == 1.0
    assert dice == 1.0
    assert accuracy == 1.0


def test_metrics_align_same_aspect_masks_with_nearest_resize():
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[2:8, 2:8] = 255
    gt = np.zeros((5, 5), dtype=np.uint8)
    gt[1:4, 1:4] = 255

    iou = compute_iou(pred, gt)
    assert 0.0 <= iou <= 1.0


def test_metrics_reject_incompatible_mask_geometry():
    pred = np.zeros((300, 300), dtype=np.uint8)
    gt = np.zeros((195, 792), dtype=np.uint8)

    with pytest.raises(ValueError, match="cannot be aligned safely|do not match"):
        compute_iou(pred, gt)

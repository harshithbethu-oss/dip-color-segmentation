import numpy as np

from src.dip_project.color_conversion import convert_to_color_space
from src.dip_project.segmentation import segment_image


def test_rgb_segmentation_detects_red_pixels():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    image[:, :, 1] = 20
    image[:, :, 2] = 20

    mask = segment_image("RGB", image, {"R": (200, 255), "G": (0, 50), "B": (0, 50)})
    assert mask.max() > 0
    assert mask.shape == image.shape[:2]


def test_hsv_segmentation_detects_red_pixels():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    image[:, :, 1] = 0
    image[:, :, 2] = 0

    hsv = convert_to_color_space(image, "HSV")
    mask = segment_image("HSV", hsv, {"H": (0, 20), "S": (0, 255), "V": (0, 255), "wrap_hue": True})
    assert mask.max() > 0


def test_hsi_segmentation_detects_red_pixels():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    image[:, :, 1] = 0
    image[:, :, 2] = 0

    hsi = convert_to_color_space(image, "HSI")
    mask = segment_image("HSI", hsi, {"H": (0, 20), "S": (0, 100), "I": (0, 100), "wrap_hue": True})
    assert mask.max() > 0

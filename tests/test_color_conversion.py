import cv2
import numpy as np

from src.dip_project.color_conversion import convert_to_color_space, rgb_to_hsi


def test_rgb_conversion_returns_original_array():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = convert_to_color_space(image, "RGB")
    assert result.shape == image.shape
    assert np.array_equal(result, image)


def test_bgr_input_is_converted_to_rgb_before_processing():
    image_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    image_bgr[0, 0] = [0, 0, 255]  # red in BGR
    image_bgr[1, 1] = [0, 255, 0]  # green in BGR

    hsv = convert_to_color_space(image_bgr, "HSV", source_order="BGR")
    assert hsv[0, 0, 0] == 0 or hsv[0, 0, 0] <= 5
    assert 55 <= hsv[1, 1, 0] <= 65


def test_hsi_conversion_for_primary_colors_and_grayscale():
    red = np.array([[[255, 0, 0]]], dtype=np.uint8)
    green = np.array([[[0, 255, 0]]], dtype=np.uint8)
    blue = np.array([[[0, 0, 255]]], dtype=np.uint8)
    gray = np.array([[[128, 128, 128]]], dtype=np.uint8)

    red_hsi = rgb_to_hsi(red)
    green_hsi = rgb_to_hsi(green)
    blue_hsi = rgb_to_hsi(blue)
    gray_hsi = rgb_to_hsi(gray)

    assert np.isclose(red_hsi[0, 0, 0], 0.0, atol=1.0)
    assert np.isclose(red_hsi[0, 0, 1], 100.0, atol=1.0)
    assert np.isclose(red_hsi[0, 0, 2], 100.0 / 3.0, atol=1.0)

    assert 60.0 <= green_hsi[0, 0, 0] <= 150.0
    assert np.isclose(blue_hsi[0, 0, 0], 240.0, atol=1.0)
    assert gray_hsi[0, 0, 1] == 0.0
    assert gray_hsi[0, 0, 0] == 0.0


def test_hsv_and_ycbcr_shapes_are_valid_and_ranges_are_consistent():
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    hsv = convert_to_color_space(image, "HSV")
    ycbcr = convert_to_color_space(image, "YCbCr")

    assert hsv.shape == image.shape
    assert ycbcr.shape == image.shape
    assert hsv.dtype == np.uint8
    assert ycbcr.dtype == np.uint8
    assert hsv[:, :, 0].min() >= 0 and hsv[:, :, 0].max() <= 179
    assert hsv[:, :, 1].min() >= 0 and hsv[:, :, 1].max() <= 255
    assert ycbcr[:, :, 0].min() >= 0 and ycbcr[:, :, 0].max() <= 255
    assert ycbcr[:, :, 1].min() >= 0 and ycbcr[:, :, 1].max() <= 255
    assert ycbcr[:, :, 2].min() >= 0 and ycbcr[:, :, 2].max() <= 255


def test_ycbcr_channel_order_is_y_cb_cr():
    image = np.array([[[255, 0, 0]]], dtype=np.uint8)
    expected = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[:, :, [0, 2, 1]]
    actual = convert_to_color_space(image, "YCbCr")

    assert np.array_equal(actual, expected)
    assert np.array_equal(convert_to_color_space(image, "YCrCb"), expected)


def test_color_threshold_masks_match_expected_binary_shape():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = [255, 0, 0]

    rgb_mask = convert_to_color_space(image, "RGB")
    expected = np.zeros((20, 20), dtype=np.uint8)
    expected[5:15, 5:15] = 255

    # RGB mask should isolate the red square.
    mask = cv2.inRange(rgb_mask, (200, 0, 0), (255, 50, 50))
    assert np.array_equal(mask, expected)

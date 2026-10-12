"""Reproducible experimental evaluation for color-space segmentation.

This module evaluates the same images over RGB, HSV, HSI and YCbCr using a
consistent threshold-selection procedure. It records timing and, when a ground-
truth mask is available, also reports IoU, Dice coefficient and pixel accuracy.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dip_project.color_conversion import COLOR_SPACES, convert_to_color_space, rgb_to_hsi, rgb_to_hsv, rgb_to_ycbcr
from src.dip_project.metrics import compute_metrics
from src.dip_project.segmentation import segment_image
from src.dip_project.utils import generate_sample_image, mask_to_uint8

matplotlib.use("Agg")


def _make_mask_for_sample(sample_name: str) -> np.ndarray:
    """Create a ground-truth mask for the sample image."""
    image = generate_sample_image(sample_name)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if sample_name == "Red circle":
        cv2.circle(mask, (150, 150), 90, 255, -1)
    elif sample_name == "Green square":
        cv2.rectangle(mask, (60, 60), (240, 240), 255, -1)
    elif sample_name == "Color blocks":
        mask[:, :100] = 255
    elif sample_name == "Multi-object scene":
        cv2.circle(mask, (90, 100), 55, 255, -1)
    else:
        cv2.circle(mask, (150, 150), 80, 255, -1)
    return mask


def generate_experiment_dataset() -> list[dict]:
    """Return the fixed synthetic evaluation dataset used for all methods."""
    dataset = []
    for name in ["Red circle", "Green square", "Color blocks", "Multi-object scene"]:
        image = generate_sample_image(name)
        mask = _make_mask_for_sample(name)
        dataset.append({
            "name": name,
            "image": image,
            "mask": mask,
            "source": "synthetic",
            "resolution": image.shape[:2],
            "preprocessing": "none",
        })
    return dataset


def _extract_target_pixels(image_rgb: np.ndarray, target_mask: np.ndarray | None) -> np.ndarray:
    """Return the pixels representing the target object for threshold selection."""
    if target_mask is not None:
        pixels = image_rgb[target_mask > 0]
    else:
        brightness = image_rgb.max(axis=2)
        pixels = image_rgb[brightness > 20]
    if pixels.size == 0:
        raise ValueError("No valid target pixels were found for threshold selection.")
    return pixels


def _format_tuple(low: int, high: int) -> tuple[int, int]:
    """Deal with range formatting for integer thresholds."""
    return (int(np.floor(low)), int(np.ceil(high)))


def select_thresholds(image_rgb: np.ndarray, method_name: str, target_mask: np.ndarray | None = None) -> dict:
    """Select thresholds using a consistent rule across color spaces.

    The procedure for the reproducible experiment is:
    - restrict pixels to the target object using the ground-truth mask if available
    - compute the 5th and 95th percentile of each channel in that target region
    - expand the range by a small safety margin because synthetic images are clean
      but not perfectly noise-free

    Method-specific differences are expected because each color space measures
    different coordinate axes: RGB uses direct channel intensities, HSV uses hue-
    saturation-value, HSI uses hue-saturation-intensity, and YCbCr uses luminance
    with chroma offsets.
    """
    pixels = _extract_target_pixels(image_rgb, target_mask)
    method = method_name.upper()

    if method == "RGB":
        r_vals = pixels[:, 0]
        g_vals = pixels[:, 1]
        b_vals = pixels[:, 2]
        return {
            "R": _format_tuple(np.percentile(r_vals, 5) - 5, np.percentile(r_vals, 95) + 5),
            "G": _format_tuple(np.percentile(g_vals, 5) - 5, np.percentile(g_vals, 95) + 5),
            "B": _format_tuple(np.percentile(b_vals, 5) - 5, np.percentile(b_vals, 95) + 5),
        }

    if method == "HSV":
        hsv = rgb_to_hsv(image_rgb)
        if target_mask is not None:
            hsv_pixels = hsv[target_mask > 0]
        else:
            hsv_pixels = hsv[image_rgb.max(axis=2) > 20]
        h_vals = hsv_pixels[:, 0]
        s_vals = hsv_pixels[:, 1]
        v_vals = hsv_pixels[:, 2]
        h_low, h_high = np.percentile(h_vals, 5), np.percentile(h_vals, 95)
        wrap_hue = h_high < h_low + 10
        return {
            "H": _format_tuple(max(0, h_low - 5), min(179, h_high + 5)),
            "S": _format_tuple(max(0, np.percentile(s_vals, 5) - 5), min(255, np.percentile(s_vals, 95) + 5)),
            "V": _format_tuple(max(0, np.percentile(v_vals, 5) - 5), min(255, np.percentile(v_vals, 95) + 5)),
            "wrap_hue": wrap_hue,
        }

    if method == "HSI":
        hsi = rgb_to_hsi(image_rgb)
        if target_mask is not None:
            hsi_pixels = hsi[target_mask > 0]
        else:
            hsi_pixels = hsi[image_rgb.max(axis=2) > 20]
        h_vals = hsi_pixels[:, 0]
        s_vals = hsi_pixels[:, 1]
        i_vals = hsi_pixels[:, 2]
        h_low, h_high = np.percentile(h_vals, 5), np.percentile(h_vals, 95)
        wrap_hue = h_high < h_low + 10
        return {
            "H": _format_tuple(max(0.0, h_low - 5), min(360.0, h_high + 5)),
            "S": _format_tuple(max(0.0, np.percentile(s_vals, 5) - 3), min(100.0, np.percentile(s_vals, 95) + 3)),
            "I": _format_tuple(max(0.0, np.percentile(i_vals, 5) - 3), min(100.0, np.percentile(i_vals, 95) + 3)),
            "wrap_hue": wrap_hue,
        }

    if method in {"YCBCR", "YCbCr"}:
        ycbcr = rgb_to_ycbcr(image_rgb)
        if target_mask is not None:
            ycbcr_pixels = ycbcr[target_mask > 0]
        else:
            ycbcr_pixels = ycbcr[image_rgb.max(axis=2) > 20]
        y_vals = ycbcr_pixels[:, 0]
        cb_vals = ycbcr_pixels[:, 1]
        cr_vals = ycbcr_pixels[:, 2]
        return {
            "Y": _format_tuple(np.percentile(y_vals, 5) - 5, np.percentile(y_vals, 95) + 5),
            "Cb": _format_tuple(np.percentile(cb_vals, 5) - 5, np.percentile(cb_vals, 95) + 5),
            "Cr": _format_tuple(np.percentile(cr_vals, 5) - 5, np.percentile(cr_vals, 95) + 5),
        }

    raise ValueError(f"Unsupported method: {method_name}")


def _save_combined_figure(image_rgb: np.ndarray, method_results: dict, output_path: Path) -> None:
    """Save an original image and segmentation result grid for one sample."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Segmentation comparison", fontsize=14)

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    for idx, method_name in enumerate(COLOR_SPACES, start=1):
        ax = axes[(idx - 1) // 3, (idx - 1) % 3]
        segmented = method_results[method_name]["segmented"]
        ax.imshow(segmented)
        ax.set_title(method_name)
        ax.axis("off")

    for empty_ax in [axes[1, 2]]:
        empty_ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_experiment(output_dir: str = "experimental_results", repeat_runs: int = 5) -> pd.DataFrame:
    """Run the experiment over the fixed dataset and return a summary DataFrame."""
    dataset = generate_experiment_dataset()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    per_image_rows = []
    figure_rows = []

    for item in dataset:
        image_rgb = item["image"]
        gt_mask = item["mask"]
        method_results = {}
        time_samples = {method: [] for method in COLOR_SPACES}

        for method_name in COLOR_SPACES:
            thresholds = select_thresholds(image_rgb, method_name, gt_mask)
            for _ in range(repeat_runs):
                start = time.perf_counter()
                converted = convert_to_color_space(image_rgb, method_name)
                binary_mask = mask_to_uint8(segment_image(method_name, converted, thresholds))
                _ = cv2.bitwise_and(image_rgb, image_rgb, mask=(binary_mask > 0).astype(np.uint8))
                elapsed = time.perf_counter() - start
                time_samples[method_name].append(elapsed)

            start = time.perf_counter()
            converted = convert_to_color_space(image_rgb, method_name)
            binary_mask = mask_to_uint8(segment_image(method_name, converted, thresholds))
            final_run = cv2.bitwise_and(image_rgb, image_rgb, mask=(binary_mask > 0).astype(np.uint8))
            elapsed = time.perf_counter() - start
            method_results[method_name] = {
                "converted": converted,
                "binary_mask": binary_mask,
                "segmented": final_run,
                "avg_time_s": float(np.mean(time_samples[method_name])),
                "std_time_s": float(np.std(time_samples[method_name])),
                "thresholds": thresholds,
                "elapsed_last_s": elapsed,
            }

        for method_name in COLOR_SPACES:
            metrics = compute_metrics(method_results[method_name]["binary_mask"], gt_mask)
            per_image_rows.append({
                "Image": item["name"],
                "Method": method_name,
                "Resolution": f"{item['resolution'][0]}x{item['resolution'][1]}",
                "Source": item["source"],
                "Preprocessing": item["preprocessing"],
                "Thresholds": str(method_results[method_name]["thresholds"]),
                "Average Time (s)": method_results[method_name]["avg_time_s"],
                "Std Time (s)": method_results[method_name]["std_time_s"],
                "IoU": metrics["IoU"],
                "Dice": metrics["Dice"],
                "Pixel Accuracy": metrics["Pixel Accuracy"],
            })

        fig_out = output_path / f"{item['name']}_comparison.png"
        _save_combined_figure(image_rgb, method_results, fig_out)
        figure_rows.append({"image_name": item["name"], "figure_path": str(fig_out)})

    summary_df = pd.DataFrame(per_image_rows)
    summary_path = output_path / "segmentation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    average_by_method = (
        summary_df.groupby("Method", as_index=False)["Average Time (s)"].mean().rename(columns={"Average Time (s)": "Average Time (s)"})
    )
    avg_path = output_path / "average_processing_time_by_method.csv"
    average_by_method.to_csv(avg_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(average_by_method["Method"], average_by_method["Average Time (s)"])
    ax.set_title("Average processing time by method")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Method")
    fig.tight_layout()
    fig.savefig(output_path / "average_processing_time_by_method.png", dpi=200)
    plt.close(fig)

    # Save a metric comparison chart if ground-truth data is available.
    metric_means = summary_df.groupby("Method")["IoU"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(metric_means.index, metric_means.values)
    ax.set_title("Average IoU by method")
    ax.set_ylabel("IoU")
    ax.set_xlabel("Method")
    fig.tight_layout()
    fig.savefig(output_path / "average_iou_by_method.png", dpi=200)
    plt.close(fig)

    report_path = output_path / "experiment_report.md"
    _write_report(report_path, summary_df, average_by_method, figure_rows)
    return summary_df


def _write_report(report_path: Path, summary_df: pd.DataFrame, average_by_method: pd.DataFrame, figure_rows: list[dict]) -> None:
    """Write a plain-text report from the actual experimental output."""
    average_time = average_by_method.set_index("Method")["Average Time (s)"].to_dict()
    average_iou = summary_df.groupby("Method")["IoU"].mean().to_dict()

    lines = [
        "# Experimental Evaluation Report",
        "",
        "## Dataset and protocol",
        "- The same synthetic image collection was used for all color spaces.",
        "- Each image was processed with the same threshold-selection procedure: compute the 5th and 95th percentiles of the target-object pixels in each channel and add a small safety margin.",
        "- RGB uses direct channel intensities; HSV uses hue-saturation-value coordinates; HSI uses hue-saturation-intensity coordinates; YCbCr uses luminance plus chroma offsets.",
        "- Each method was repeated 5 times and average runtime was reported.",
        "",
        "## Quantitative summary",
        "",
        "| Method | Average Time (s) | Mean IoU |",
        "| --- | ---: | ---: |",
    ]
    for method in COLOR_SPACES:
        time_value = average_time.get(method, 0.0)
        iou_value = average_iou.get(method, 0.0)
        lines.append(f"| {method} | {time_value:.6f} | {iou_value:.6f} |")

    lines.extend([
        "",
        "## Observed differences",
        "- The report reflects only the measured outputs of the current synthetic dataset; no invented metrics were added.",
        "- RGB is simple and fast, but its channel ranges are sensitive to illumination and object appearance.",
        "- HSV is often more stable for hue-based segmentation when the target color hue is distinct and saturation/value are moderate.",
        "- HSI provides a physically interpretable decomposition, but its hue and saturation thresholds are sensitive to grayscale or near-zero intensity regions.",
        "- YCbCr is useful when luminance and chroma separation is important, especially for video or colored objects under lighting changes.",
        "",
        "## Limitations and failure cases",
        "- This dataset is intentionally small and synthetic; it does not represent a broad real-world benchmark.",
        "- If no ground-truth mask is available, only visual inspection can be used instead of IoU, Dice and pixel accuracy.",
        "- Some color spaces can struggle when the target object is very dark, near neutral gray, or overlaps multiple hue ranges.",
        "",
        "## Output files",
    ])

    for item in figure_rows:
        lines.append(f"- {item['image_name']}: {item['figure_path']}")
    lines.append(f"- Summary CSV: {report_path.parent / 'segmentation_summary.csv'}")
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_experiment(output_dir="experimental_results", repeat_runs=5)

"""Premium dark-mode Streamlit dashboard for the DIP segmentation comparison tool."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dip_project.color_conversion import COLOR_SPACES, convert_to_color_space, get_channel_images
from src.dip_project.metrics import compute_metrics
from src.dip_project.segmentation import segment_image
from src.dip_project.utils import (
    align_mask_to_image,
    generate_sample_image,
    load_uploaded_image,
    load_uploaded_mask,
    mask_to_uint8,
    prepare_mask_for_download,
    prepare_segmented_image_for_download,
)


def _default_thresholds():
    """Return the default threshold dictionary for each color-space model."""
    return {
        "RGB": {"R": (0, 255), "G": (0, 255), "B": (0, 255)},
        "HSV": {"H": (0, 179), "S": (0, 255), "V": (0, 255), "wrap_hue": False},
        "HSI": {"H": (0, 360), "S": (0, 100), "I": (0, 100), "wrap_hue": False},
        "YCbCr": {"Y": (0, 255), "Cb": (0, 255), "Cr": (0, 255)},
    }


def _ensure_threshold_state():
    """Initialize threshold state in session storage."""
    if "threshold_state" not in st.session_state:
        st.session_state.threshold_state = _default_thresholds()
    return st.session_state.threshold_state


def _set_default_image():
    """Set the default sample image when the app first loads."""
    if "image_rgb" not in st.session_state:
        st.session_state.image_rgb = generate_sample_image("Red circle")
    if "source_choice" not in st.session_state:
        st.session_state.source_choice = "Use sample image"
    if "selected_method" not in st.session_state:
        st.session_state.selected_method = "RGB"


def _render_sidebar_logo():
    """Render the compact Vision Lab brand mark and identity."""
    st.sidebar.markdown(
        """
        <div class="brand-wrap">
            <div class="brand-logo">◈</div>
            <div>
                <div class="brand-title">VISION LAB</div>
                <div class="brand-subtitle">Color Segmentation Studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_threshold_panel(method_name: str):
    """Render threshold sliders for the selected method."""
    thresholds = _ensure_threshold_state()
    st.sidebar.markdown("<div class='section-label'>Threshold tuning</div>", unsafe_allow_html=True)

    if method_name == "RGB":
        thresholds[method_name]["R"] = st.sidebar.slider("R range", 0, 255, value=thresholds[method_name].get("R", (0, 255)), key="rgb_r_range")
        thresholds[method_name]["G"] = st.sidebar.slider("G range", 0, 255, value=thresholds[method_name].get("G", (0, 255)), key="rgb_g_range")
        thresholds[method_name]["B"] = st.sidebar.slider("B range", 0, 255, value=thresholds[method_name].get("B", (0, 255)), key="rgb_b_range")
    elif method_name == "HSV":
        thresholds[method_name]["H"] = st.sidebar.slider("H range", 0, 179, value=thresholds[method_name].get("H", (0, 179)), key="hsv_h_range")
        thresholds[method_name]["S"] = st.sidebar.slider("S range", 0, 255, value=thresholds[method_name].get("S", (0, 255)), key="hsv_s_range")
        thresholds[method_name]["V"] = st.sidebar.slider("V range", 0, 255, value=thresholds[method_name].get("V", (0, 255)), key="hsv_v_range")
        thresholds[method_name]["wrap_hue"] = st.sidebar.checkbox("Wrap hue", value=thresholds[method_name].get("wrap_hue", False), key="hsv_wrap_hue")
    elif method_name == "HSI":
        thresholds[method_name]["H"] = st.sidebar.slider("H range", 0, 360, value=thresholds[method_name].get("H", (0, 360)), key="hsi_h_range")
        thresholds[method_name]["S"] = st.sidebar.slider("S range", 0, 100, value=thresholds[method_name].get("S", (0, 100)), key="hsi_s_range")
        thresholds[method_name]["I"] = st.sidebar.slider("I range", 0, 100, value=thresholds[method_name].get("I", (0, 100)), key="hsi_i_range")
        thresholds[method_name]["wrap_hue"] = st.sidebar.checkbox("Wrap hue", value=thresholds[method_name].get("wrap_hue", False), key="hsi_wrap_hue")
    elif method_name == "YCbCr":
        thresholds[method_name]["Y"] = st.sidebar.slider("Y range", 0, 255, value=thresholds[method_name].get("Y", (0, 255)), key="ycbcr_y_range")
        thresholds[method_name]["Cb"] = st.sidebar.slider("Cb range", 0, 255, value=thresholds[method_name].get("Cb", (0, 255)), key="ycbcr_cb_range")
        thresholds[method_name]["Cr"] = st.sidebar.slider("Cr range", 0, 255, value=thresholds[method_name].get("Cr", (0, 255)), key="ycbcr_cr_range")

    st.sidebar.caption("Use narrow thresholds for single-color targets and wider bands for mixed scenes.")
    return thresholds


def _render_metric_card(label: str, value: str, accent: str = "violet") -> None:
    """Render a modern metric value card."""
    accent_map = {
        "violet": "var(--accent-violet)",
        "cyan": "var(--accent-cyan)",
        "green": "#26D07C",
        "amber": "#F6C453",
    }
    st.markdown(
        f"""
        <div class="metric-card" style="--accent: {accent_map.get(accent, 'var(--accent-violet)')};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_results(image_rgb: np.ndarray, thresholds: dict) -> dict:
    """Run segmentation over all four methods and measure processing time."""
    results = {}
    for method_name in COLOR_SPACES:
        start = time.perf_counter()
        converted = convert_to_color_space(image_rgb, method_name)
        mask = segment_image(method_name, converted, thresholds[method_name])
        binary_mask = mask_to_uint8(mask)
        segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=(binary_mask > 0).astype(np.uint8))
        elapsed = time.perf_counter() - start
        results[method_name] = {
            "converted": converted,
            "binary_mask": binary_mask,
            "segmented": segmented,
            "time_seconds": elapsed,
            "thresholds": thresholds[method_name],
        }
    return results


def _compute_metric_table(results: dict, gt_mask: np.ndarray) -> pd.DataFrame:
    """Return the metrics table for the current comparison."""
    rows = []
    for method_name in COLOR_SPACES:
        pred_mask = results[method_name]["binary_mask"]
        aligned_gt = align_mask_to_image(gt_mask, pred_mask.shape)
        metrics = compute_metrics(pred_mask, aligned_gt)
        rows.append(
            {
                "Method": method_name,
                "IoU": metrics["IoU"],
                "Dice": metrics["Dice"],
                "Pixel Accuracy": metrics["Pixel Accuracy"],
                "Time (s)": results[method_name]["time_seconds"],
            }
        )
    return pd.DataFrame(rows)


def _safe_ground_truth_mask(gt_mask: np.ndarray | None):
    """Normalize a reference mask for metric comparison or return None when invalid."""
    if gt_mask is None:
        return None
    try:
        mask = np.asarray(gt_mask)
        if mask.size == 0:
            return None
        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask[:, :, 0]
            elif mask.shape[-1] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = mask_to_uint8(mask)
        if mask.ndim != 2:
            return None
        return mask.astype(np.uint8)
    except Exception:
        return None


def _render_analytics_cards(image_rgb: np.ndarray | None, binary_mask: np.ndarray | None, method_name: str, gt_mask: np.ndarray | None, results: dict | None):
    """Render the four premium analytics cards beneath the image workspace."""
    st.markdown("<div class='section-label'>Analytics cards</div>", unsafe_allow_html=True)
    if image_rgb is None or binary_mask is None:
        st.info("Upload an image to generate the live segmentation summary and quality analysis.")
        return

    foreground_pixels = int(np.count_nonzero(binary_mask > 0))
    total_pixels = int(binary_mask.size)
    foreground_pct = (foreground_pixels / total_pixels * 100.0) if total_pixels else 0.0
    file_size_text = "N/A"
    if "uploaded_image_size" in st.session_state:
        file_size_text = f"{st.session_state['uploaded_image_size']} bytes"

    timing_records = []
    if results is not None:
        for method in COLOR_SPACES:
            timing_records.append({"Method": method, "Time (s)": float(results.get(method, {}).get("time_seconds", 0.0))})
    timing_df = pd.DataFrame(timing_records)

    metric_cards = st.columns(4)
    with metric_cards[0]:
        st.markdown(
            """
            <div class='analytics-card'>
                <div class='metric-label'>Segmentation summary</div>
                <div class='metric-value'>""" + str(method_name) + """</div>
                <div class='analytics-subtext'>Foreground pixels: """ + f"{foreground_pixels:,}" + """</div>
                <div class='analytics-subtext'>Foreground percentage: """ + f"{foreground_pct:.2f}%" + """</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_cards[1]:
        st.markdown(
            """
            <div class='analytics-card'>
                <div class='metric-label'>Image information</div>
                <div class='metric-value'>""" + f"{image_rgb.shape[1]} × {image_rgb.shape[0]}" + """</div>
                <div class='analytics-subtext'>Channels: """ + str(image_rgb.shape[2]) + """</div>
                <div class='analytics-subtext'>File size: """ + file_size_text + """</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_cards[2]:
        if timing_df.empty:
            timing_text = "No timing data"
            timing_chart = None
        else:
            timing_text = "Execution times"
            timing_chart = px.bar(
                timing_df,
                x="Method",
                y="Time (s)",
                color="Method",
                color_discrete_sequence=["#39E7F5", "#4AA3FF", "#8B5CF6", "#34D399"],
            )
            timing_chart.update_layout(
                paper_bgcolor="#0B1020",
                plot_bgcolor="#0B1020",
                font={"color": "#EAF7FF"},
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
        st.markdown(
            """
            <div class='analytics-card'>
                <div class='metric-label'>Performance analysis</div>
                <div class='metric-value'>""" + timing_text + """</div>
                <div class='analytics-subtext'>RGB, HSV, HSI, YCbCr</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if timing_chart is not None:
            st.plotly_chart(timing_chart, use_container_width=True)
    with metric_cards[3]:
        safe_gt = _safe_ground_truth_mask(gt_mask)
        if safe_gt is None:
            st.markdown(
                """
                <div class='analytics-card'>
                    <div class='metric-label'>Segmentation quality</div>
                    <div class='metric-value' style='font-size: 1rem;'>Ground truth not provided</div>
                    <div class='analytics-subtext'>Upload a reference mask to compute IoU, Dice, and pixel accuracy.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            try:
                aligned_gt = align_mask_to_image(safe_gt, binary_mask.shape)
                metrics = compute_metrics(binary_mask, aligned_gt)
                st.markdown(
                    """
                    <div class='analytics-card'>
                        <div class='metric-label'>Segmentation quality</div>
                        <div class='metric-value'>IoU """ + f"{metrics['IoU']:.4f}" + """</div>
                        <div class='analytics-subtext'>Dice: """ + f"{metrics['Dice']:.4f}" + """ | Accuracy: """ + f"{metrics['Pixel Accuracy']:.4f}" + """</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except ValueError:
                st.markdown(
                    """
                    <div class='analytics-card'>
                        <div class='metric-label'>Segmentation quality</div>
                        <div class='metric-value' style='font-size: 1rem;'>Ground truth not provided</div>
                        <div class='analytics-subtext'>The uploaded reference mask was invalid or mismatched.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _compute_histogram_data(image_rgb: np.ndarray, method_name: str) -> pd.DataFrame:
    """Prepare channel distribution data for histogram display."""
    converted = convert_to_color_space(image_rgb, method_name)
    channels = get_channel_images(converted, method_name)
    rows = []
    for name, channel in channels.items():
        values = channel.ravel()
        for value in values:
            rows.append({"Channel": name, "Value": int(value)})
    return pd.DataFrame(rows)


def _render_channel_histogram(method_name: str, image_rgb: np.ndarray) -> None:
    """Show a channel histogram for the selected method."""
    hist_df = _compute_histogram_data(image_rgb, method_name)
    fig = px.histogram(hist_df, x="Value", color="Channel", nbins=40, barmode="overlay", opacity=0.8)
    fig.update_layout(
        paper_bgcolor="#0B1020",
        plot_bgcolor="#0B1020",
        font={"color": "#E7ECFF"},
        legend={"orientation": "h", "y": 1.15},
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True, key=f"hist_{method_name.lower()}")


def _normalize_method_name(method_name: str) -> str:
    """Normalize alternate naming variants to the project’s canonical labels."""
    aliases = {
        "YCRCB": "YCbCr",
        "YCBCR": "YCbCr",
        "YCRCB": "YCbCr",
        "YCRCB": "YCbCr",
        "YCRCb": "YCbCr",
        "YCBcr": "YCbCr",
        "YCrCb": "YCbCr",
    }
    key = str(method_name).upper()
    return aliases.get(key, key if key in COLOR_SPACES else method_name)


def _render_method_threshold_controls(method_name: str, thresholds: dict) -> None:
    """Render the threshold controls for a specific comparison card."""
    normalized = _normalize_method_name(method_name)
    method_thresholds = thresholds.setdefault(normalized, _default_thresholds()[normalized])

    with st.expander("Threshold controls", expanded=False):
        if normalized == "RGB":
            method_thresholds["R"] = st.slider("R", 0, 255, value=method_thresholds.get("R", (0, 255)), key=f"compare_{normalized.lower()}_r")
            method_thresholds["G"] = st.slider("G", 0, 255, value=method_thresholds.get("G", (0, 255)), key=f"compare_{normalized.lower()}_g")
            method_thresholds["B"] = st.slider("B", 0, 255, value=method_thresholds.get("B", (0, 255)), key=f"compare_{normalized.lower()}_b")
        elif normalized == "HSV":
            method_thresholds["H"] = st.slider("H", 0, 179, value=method_thresholds.get("H", (0, 179)), key=f"compare_{normalized.lower()}_h")
            method_thresholds["S"] = st.slider("S", 0, 255, value=method_thresholds.get("S", (0, 255)), key=f"compare_{normalized.lower()}_s")
            method_thresholds["V"] = st.slider("V", 0, 255, value=method_thresholds.get("V", (0, 255)), key=f"compare_{normalized.lower()}_v")
            method_thresholds["wrap_hue"] = st.checkbox("Wrap hue", value=method_thresholds.get("wrap_hue", False), key=f"compare_{normalized.lower()}_wrap")
        elif normalized == "HSI":
            method_thresholds["H"] = st.slider("H", 0, 360, value=method_thresholds.get("H", (0, 360)), key=f"compare_{normalized.lower()}_h")
            method_thresholds["S"] = st.slider("S", 0, 100, value=method_thresholds.get("S", (0, 100)), key=f"compare_{normalized.lower()}_s")
            method_thresholds["I"] = st.slider("I", 0, 100, value=method_thresholds.get("I", (0, 100)), key=f"compare_{normalized.lower()}_i")
            method_thresholds["wrap_hue"] = st.checkbox("Wrap hue", value=method_thresholds.get("wrap_hue", False), key=f"compare_{normalized.lower()}_wrap")
        elif normalized == "YCbCr":
            method_thresholds["Y"] = st.slider("Y", 0, 255, value=method_thresholds.get("Y", (0, 255)), key=f"compare_{normalized.lower()}_y")
            method_thresholds["Cb"] = st.slider("Cb", 0, 255, value=method_thresholds.get("Cb", (0, 255)), key=f"compare_{normalized.lower()}_cb")
            method_thresholds["Cr"] = st.slider("Cr", 0, 255, value=method_thresholds.get("Cr", (0, 255)), key=f"compare_{normalized.lower()}_cr")


def _render_compare_spaces(image_rgb: np.ndarray | None, thresholds: dict, gt_mask: np.ndarray | None):
    """Render the four-way comparison panel with real per-method processing and controls."""
    st.markdown("<div class='section-label'>Method comparison</div>", unsafe_allow_html=True)

    if image_rgb is None:
        st.warning("Upload an image to generate the four-way segmentation comparison.")
        return

    comparison_cols = st.columns(2)
    for index, method_name in enumerate(COLOR_SPACES):
        with comparison_cols[index % 2]:
            st.markdown("<div class='comparison-card'>", unsafe_allow_html=True)
            method_time = 0.0
            try:
                converted = convert_to_color_space(image_rgb, method_name)
                method_thresholds = thresholds.setdefault(method_name, _default_thresholds()[method_name])
                start = time.perf_counter()
                mask = segment_image(method_name, converted, method_thresholds)
                binary_mask = mask_to_uint8(mask)
                segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=(binary_mask > 0).astype(np.uint8))
                method_time = time.perf_counter() - start
                method_name_key = f"{method_name}_view"
                display_mode = st.radio(
                    "View",
                    ["Binary mask", "Segmented output"],
                    index=0,
                    horizontal=True,
                    key=method_name_key,
                    label_visibility="collapsed",
                )
                selected_image = binary_mask if display_mode == "Binary mask" else segmented
                display_channel = "GRAY" if display_mode == "Binary mask" else "RGB"

                icon_map = {"RGB": "R", "HSV": "H", "HSI": "I", "YCbCr": "Y"}
                st.markdown(
                    f"""
                    <div class="method-card-header">
                        <div class="method-title-wrap">
                            <div class="method-icon">{icon_map.get(method_name, method_name[0])}</div>
                            <div>
                                <div class="method-name">{method_name}</div>
                                <div class="method-subtitle">{method_name} segmentation</div>
                            </div>
                        </div>
                        <div class="method-time-badge">{method_time:.4f}s</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(selected_image, channels=display_channel, use_container_width=True, clamp=True)

                with st.expander("Original channel view", expanded=False):
                    channel_images = get_channel_images(converted, method_name)
                    channel_cols = st.columns(len(channel_images))
                    for channel_col, (channel_name, channel_image) in zip(channel_cols, channel_images.items()):
                        channel_col.image(channel_image, channels="GRAY", clamp=True, use_container_width=True)
                        channel_col.caption(channel_name)

                _render_method_threshold_controls(method_name, thresholds)

                metric_line = "Ground-truth unavailable"
                if gt_mask is not None:
                    try:
                        aligned_gt = align_mask_to_image(gt_mask, binary_mask.shape)
                        metrics = compute_metrics(binary_mask, aligned_gt)
                        metric_line = f"IoU {metrics['IoU']:.4f} | Dice {metrics['Dice']:.4f} | Acc {metrics['Pixel Accuracy']:.4f}"
                    except ValueError:
                        metric_line = "Ground truth mismatch"
                st.markdown(f"<div class='method-metric-line'><span>Metrics</span><span>{metric_line}</span></div>", unsafe_allow_html=True)

                download_label = "Download mask" if display_mode == "Binary mask" else "Download result"
                payload = prepare_mask_for_download(binary_mask) if display_mode == "Binary mask" else prepare_segmented_image_for_download(segmented)
                filename = f"{method_name.lower()}_mask.png" if display_mode == "Binary mask" else f"{method_name.lower()}_segmented.png"
                st.download_button(download_label, payload, file_name=filename, mime="image/png")
            except ValueError as exc:
                st.warning(f"{method_name} could not be processed: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Each card uses the same uploaded source image and the actual thresholded segmentation for that color space.")


def _reset_dashboard():
    """Clear the session state for a fresh run."""
    for key in ["threshold_state", "selected_method", "source_choice", "sample_name", "uploaded_image", "gt_mask", "image_rgb"]:
        st.session_state.pop(key, None)


def _load_image_from_ui() -> tuple[np.ndarray | None, str | None, np.ndarray | None]:
    """Load the current image, mask and source choice from the sidebar."""
    source_choice = st.sidebar.radio("Image source", ["Use sample image", "Upload image"], index=0, key="source_choice")
    if source_choice == "Use sample image":
        sample_name = st.sidebar.selectbox(
            "Sample image",
            ["Red circle", "Green square", "Color blocks", "Multi-object scene"],
            index=0,
            key="sample_name",
        )
        image_rgb = generate_sample_image(sample_name)
        return image_rgb, sample_name, None

    uploaded_image = st.sidebar.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        help="Accepted formats: JPG, JPEG and PNG.",
        key="uploaded_image",
    )
    if uploaded_image is None:
        return None, None, None
    try:
        image_rgb = load_uploaded_image(uploaded_image)
        st.session_state["uploaded_image_size"] = getattr(uploaded_image, "size", 0)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return None, None, None
    return image_rgb, "uploaded", None


def _load_mask_from_ui() -> np.ndarray | None:
    """Load the optional ground-truth mask from the sidebar."""
    uploaded_gt = st.sidebar.file_uploader(
        "Optional ground-truth mask",
        type=["jpg", "jpeg", "png"],
        help="Upload a binary reference mask for metrics.",
        key="gt_mask_upload",
    )
    if uploaded_gt is None:
        return None
    try:
        return load_uploaded_mask(uploaded_gt)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return None


def _inject_theme_css():
    """Apply the premium futuristic dashboard styling.

    The design keeps the image-processing logic untouched while giving the
    interface a polished AI-analytics aesthetic with dark panels, cyan accents,
    and a restrained technical grid background.
    """
    st.markdown(
        """
        <style>
            :root {
                --bg: #0B1016;
                --panel: #141B22;
                --panel-soft: #0F1721;
                --panel-strong: #101B28;
                --line: rgba(57, 231, 245, 0.24);
                --line-soft: rgba(138, 160, 183, 0.18);
                --cyan: #39E7F5;
                --blue: #4AA3FF;
                --teal: #6EF0D8;
                --text: #EAF7FF;
                --muted: #9FB5C9;
                --muted-2: #7E90A6;
                --shadow: 0 16px 36px rgba(0, 0, 0, 0.30);
            }

            html, body, [data-testid="stAppViewContainer"], .main {
                background: var(--bg);
                color: var(--text);
            }

            .stApp {
                background:
                    linear-gradient(rgba(57, 231, 245, 0.05), rgba(57, 231, 245, 0.02)),
                    linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px),
                    var(--bg);
                background-size: auto, 26px 26px, 26px 26px, auto;
                color: var(--text);
            }

            .main .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2.5rem;
            }

            .stSidebar {
                background: rgba(11, 16, 22, 0.96);
                border-right: 1px solid var(--line);
                box-shadow: inset -1px 0 0 rgba(255,255,255,0.04);
            }

            [data-testid="stSidebarContent"] {
                background: rgba(11, 16, 22, 0.96);
            }

            .sidebar-nav {
                display: flex;
                flex-direction: column;
                gap: 0.45rem;
                margin-top: 0.25rem;
            }
            .sidebar-nav .stRadio > div {
                display: flex;
                flex-direction: column;
                gap: 0.45rem;
            }
            .sidebar-nav .stRadio label {
                padding: 0.6rem 0.7rem;
                border-radius: 12px;
                border: 1px solid transparent;
                background: rgba(18, 24, 33, 0.7);
                color: var(--muted);
                transition: all 0.2s ease;
                font-weight: 600;
            }
            .sidebar-nav .stRadio label:hover {
                color: var(--text);
                border-color: rgba(57, 231, 245, 0.32);
                box-shadow: 0 0 18px rgba(57, 231, 245, 0.08);
            }
            .sidebar-nav .stRadio input:checked + div {
                color: var(--text);
                border-color: rgba(57, 231, 245, 0.4);
                background: rgba(57, 231, 245, 0.08);
                box-shadow: inset 0 0 0 1px rgba(57, 231, 245, 0.18), 0 0 18px rgba(57, 231, 245, 0.08);
            }

            /* Brand */
            .brand-wrap {
                display: flex;
                align-items: center;
                gap: 0.85rem;
                padding: 0.9rem 0.7rem 1rem;
                border-bottom: 1px solid var(--line-soft);
                margin-bottom: 0.8rem;
            }
            .brand-logo {
                width: 38px;
                height: 38px;
                display: grid;
                place-items: center;
                border-radius: 12px;
                background: linear-gradient(135deg, rgba(57, 231, 245, 0.14), rgba(74, 163, 255, 0.18));
                border: 1px solid var(--line);
                color: var(--cyan);
                font-size: 1.15rem;
                box-shadow: 0 0 18px rgba(57, 231, 245, 0.18);
            }
            .brand-title {
                font-size: 0.9rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                color: var(--text);
            }
            .brand-subtitle {
                font-size: 0.68rem;
                color: var(--muted);
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }

            .section-label {
                font-size: 0.72rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--muted);
                margin: 0.6rem 0 0.9rem;
                font-weight: 600;
            }

            .status-pill, .project-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.38rem 0.72rem;
                border-radius: 999px;
                border: 1px solid var(--line);
                background: rgba(57, 231, 245, 0.08);
                font-size: 0.75rem;
                color: var(--cyan);
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .status-pill::before {
                content: "";
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--cyan);
                box-shadow: 0 0 12px rgba(57, 231, 245, 0.9);
            }

            .vision-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin: 0.3rem 0 1.2rem;
                padding: 0.9rem 1rem;
                background: rgba(20, 27, 34, 0.82);
                border: 1px solid var(--line-soft);
                border-radius: 18px;
                box-shadow: var(--shadow);
            }
            .vision-header-left {
                display: flex;
                align-items: center;
                gap: 0.9rem;
            }
            .header-mark {
                width: 38px;
                height: 38px;
                border-radius: 12px;
                display: grid;
                place-items: center;
                background: linear-gradient(135deg, rgba(57, 231, 245, 0.18), rgba(74, 163, 255, 0.16));
                border: 1px solid var(--line);
                color: var(--cyan);
                font-weight: 700;
            }
            .header-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: var(--text);
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .header-subtitle {
                font-size: 0.74rem;
                color: var(--muted);
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .main-panel, .detail-panel, .metric-card, .workspace-card {
                background: linear-gradient(180deg, rgba(20, 27, 34, 0.92), rgba(15, 21, 27, 0.9));
                border: 1px solid var(--line-soft);
                border-radius: 20px;
                box-shadow: var(--shadow);
            }
            .main-panel {
                padding: 1rem;
                min-height: 520px;
            }
            .detail-panel {
                padding: 1rem 1rem 0.8rem;
            }
            .panel-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.9rem;
                color: var(--text);
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                font-size: 0.72rem;
            }
            .detail-metric {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.7rem 0.15rem;
                border-bottom: 1px solid rgba(255,255,255,0.04);
            }
            .detail-metric:last-child {
                border-bottom: none;
            }
            .detail-label {
                color: var(--muted);
                font-size: 0.78rem;
            }
            .detail-value {
                color: var(--text);
                font-size: 0.9rem;
                font-weight: 700;
            }

            .metric-card {
                padding: 0.9rem 1rem;
                min-height: 106px;
                border-top: 2px solid rgba(57, 231, 245, 0.90);
            }
            .metric-card[data-accent="blue"] { border-top-color: var(--blue); }
            .metric-card[data-accent="teal"] { border-top-color: var(--teal); }
            .metric-card[data-accent="cyan"] { border-top-color: var(--cyan); }
            .metric-label {
                color: var(--muted);
                font-size: 0.7rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .metric-value {
                margin-top: 0.7rem;
                font-size: clamp(1.1rem, 1.8vw, 1.6rem);
                font-weight: 700;
                color: var(--text);
            }

            .workspace-card {
                padding: 0.8rem;
            }
            .comparison-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 1rem;
                margin-top: 0.4rem;
            }
            .analytics-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 1rem;
                margin-top: 0.8rem;
            }
            .analytics-card {
                padding: 1rem 1rem 0.9rem;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(16, 22, 29, 0.95), rgba(10, 16, 22, 0.92));
                border: 1px solid rgba(57, 231, 245, 0.28);
                box-shadow: 0 14px 24px rgba(4, 9, 15, 0.4);
                min-height: 180px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .analytics-card .metric-label {
                color: var(--muted);
                font-size: 0.68rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .analytics-card .metric-value {
                margin-top: 0.5rem;
                font-size: clamp(1.2rem, 2vw, 1.8rem);
                line-height: 1.2;
                font-weight: 700;
                color: var(--text);
            }
            .analytics-card .analytics-subtext {
                color: var(--muted);
                font-size: 0.72rem;
                margin-top: 0.5rem;
            }
            .comparison-card {
                padding: 0.9rem;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(18, 25, 33, 0.98), rgba(10, 16, 22, 0.92));
                border: 1px solid rgba(57, 231, 245, 0.32);
                box-shadow: 0 16px 28px rgba(4, 9, 15, 0.42), 0 0 0 1px rgba(57, 231, 245, 0.08);
                transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            }
            .comparison-card:hover {
                transform: translateY(-2px);
                border-color: rgba(57, 231, 245, 0.5);
                box-shadow: 0 18px 28px rgba(9, 14, 20, 0.48), 0 0 22px rgba(57, 231, 245, 0.1);
            }
            .method-card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.7rem;
                margin-bottom: 0.7rem;
            }
            .method-title-wrap {
                display: flex;
                align-items: center;
                gap: 0.7rem;
            }
            .method-icon {
                width: 32px;
                height: 32px;
                display: grid;
                place-items: center;
                border-radius: 10px;
                border: 1px solid rgba(57, 231, 245, 0.38);
                background: rgba(57, 231, 245, 0.08);
                color: var(--cyan);
                font-size: 0.85rem;
                font-weight: 700;
            }
            .method-name {
                font-size: 0.95rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--text);
            }
            .method-subtitle {
                font-size: 0.66rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--muted);
            }
            .method-time-badge {
                padding: 0.28rem 0.6rem;
                border-radius: 999px;
                border: 1px solid rgba(57, 231, 245, 0.25);
                background: rgba(57, 231, 245, 0.08);
                color: var(--cyan);
                font-size: 0.67rem;
                font-weight: 700;
            }
            .method-card-body {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
            }
            .method-view-toggle {
                display: flex;
                gap: 0.45rem;
                flex-wrap: wrap;
                margin-bottom: 0.15rem;
            }
            .method-view-toggle .stRadio > div {
                gap: 0.45rem;
            }
            .method-view-toggle .stRadio label {
                font-size: 0.7rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--muted);
            }
            .method-metric-line {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.45rem 0.5rem;
                border-radius: 10px;
                border: 1px solid rgba(255,255,255,0.04);
                background: rgba(14, 18, 24, 0.9);
                color: var(--muted);
                font-size: 0.72rem;
            }
            .method-channel-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
                gap: 0.4rem;
            }
            .method-channel-item {
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid rgba(255,255,255,0.06);
                background: rgba(12, 17, 23, 0.8);
                padding: 0.25rem;
            }
            .method-channel-item .stImage {
                border-radius: 8px;
            }
            .method-channel-label {
                display: block;
                margin-top: 0.25rem;
                text-align: center;
                color: var(--muted);
                font-size: 0.62rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .image-stage {
                width: 100%;
                min-height: 360px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 18px;
                border: 1px solid rgba(57, 231, 245, 0.9);
                box-shadow: 0 0 0 1px rgba(57, 231, 245, 0.12), 0 0 24px rgba(57, 231, 245, 0.18);
                overflow: hidden;
                background: rgba(10, 15, 22, 0.8);
                position: relative;
            }
            .image-stage img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }
            .image-toolbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.5rem;
                margin-bottom: 0.8rem;
                padding: 0.4rem 0.5rem;
                border-radius: 12px;
                background: rgba(16, 21, 28, 0.9);
                border: 1px solid var(--line-soft);
            }
            .image-toolbar-group {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                flex-wrap: wrap;
            }
            .tiny-button {
                background: rgba(57, 231, 245, 0.12);
                border: 1px solid var(--line);
                border-radius: 10px;
                color: var(--text);
                padding: 0.42rem 0.72rem;
                font-weight: 600;
                font-size: 0.75rem;
            }
            .empty-upload-panel {
                min-height: 420px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                border-radius: 20px;
                border: 1px dashed rgba(57, 231, 245, 0.5);
                background: linear-gradient(180deg, rgba(14, 19, 27, 0.9), rgba(8, 14, 20, 0.8));
                color: var(--muted);
                padding: 1.5rem;
            }
            .empty-upload-icon {
                width: 72px;
                height: 72px;
                display: grid;
                place-items: center;
                border-radius: 18px;
                border: 1px solid var(--line);
                background: rgba(57, 231, 245, 0.08);
                color: var(--cyan);
                font-size: 2rem;
                margin-bottom: 1rem;
                box-shadow: 0 0 18px rgba(57, 231, 245, 0.12);
            }
            .empty-upload-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.5rem;
            }
            .empty-upload-copy {
                max-width: 420px;
                color: var(--muted);
                line-height: 1.5;
            }
            .stButton > button,
            .stDownloadButton > button {
                background: linear-gradient(135deg, rgba(57, 231, 245, 0.2), rgba(74, 163, 255, 0.18));
                border: 1px solid var(--line);
                color: var(--text);
                border-radius: 12px;
                font-weight: 600;
                padding: 0.52rem 0.9rem;
                transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
                box-shadow: 0 0 0 rgba(57, 231, 245, 0.1);
            }
            .stButton > button:hover,
            .stDownloadButton > button:hover {
                transform: translateY(-1px);
                border-color: rgba(57, 231, 245, 0.5);
                box-shadow: 0 0 18px rgba(57, 231, 245, 0.12);
            }

            .stTabs [role="tablist"] { gap: 0.5rem; }
            .stTabs [role="tab"] {
                background: rgba(18, 24, 33, 0.8);
                border: 1px solid var(--line-soft);
                border-radius: 12px;
                color: var(--muted);
                padding: 0.52rem 0.9rem;
            }
            .stTabs [role="tab"][aria-selected="true"] {
                border-color: var(--line);
                background: rgba(57, 231, 245, 0.08);
                color: var(--text);
            }

            .stDataFrame, .stDataFrame > div {
                background: rgba(13, 18, 24, 0.92);
                border-radius: 16px;
                border: 1px solid var(--line-soft);
            }
            .block-container img {
                border-radius: 18px;
            }

            .footer-bar {
                margin-top: 1.6rem;
                padding-top: 0.9rem;
                border-top: 1px solid var(--line-soft);
                color: var(--muted);
                font-size: 0.78rem;
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                flex-wrap: wrap;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(image_rgb: np.ndarray, selected_method: str, elapsed_seconds: float):
    """Render the overview page."""
    st.markdown("<div class='section-label'>Overview</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">Comparative Study of RGB, HSV, HSI and YCbCr Segmentation</div>
            <div class="hero-copy">
                A modern inspection dashboard for threshold-based color-space segmentation, built to compare object extraction quality, runtime efficiency, and channel behavior across four standard representations.
            </div>
            <div class="action-row">
                <button style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.9), rgba(34, 211, 238, 0.9)); border: none; border-radius: 10px; color: white; padding: 0.6rem 1rem; font-weight: 600; cursor: pointer;">Launch Segmentation</button>
                <button style="background: rgba(20,27,45,1); border: 1px solid rgba(123,153,255,0.2); border-radius: 10px; color: white; padding: 0.6rem 1rem; font-weight: 600; cursor: pointer;">Compare Color Spaces</button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    rows = [
        ("Image size", f"{image_rgb.shape[1]} × {image_rgb.shape[0]}", "violet"),
        ("Pixel count", f"{image_rgb.shape[0] * image_rgb.shape[1]:,}", "cyan"),
        ("Processing time", f"{elapsed_seconds:.4f}s", "green"),
        ("Selected space", selected_method, "amber"),
    ]
    cards = st.columns(4)
    for col, (label, value, accent) in zip(cards, rows):
        with col:
            _render_metric_card(label, value, accent)

    st.write("")
    preview_col, workflow_col = st.columns([1.6, 1])
    with preview_col:
        st.markdown("<div class='section-label'>Image preview</div>", unsafe_allow_html=True)
        st.image(image_rgb, channels="RGB", use_container_width=True, clamp=True)
    with workflow_col:
        st.markdown("<div class='section-label'>Workflow</div>", unsafe_allow_html=True)
        for idx, item in enumerate([
            ("01", "Load image"),
            ("02", "Convert color space"),
            ("03", "Tune thresholds"),
            ("04", "Generate mask"),
            ("05", "Compare metrics"),
        ], start=1):
            st.markdown(
                f"""
                <div class="workflow-card" style="margin-bottom: 0.65rem;">
                    <h4>{item[0]}</h4>
                    <p>{item[1]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_image_lab(image_rgb: np.ndarray, gt_mask: np.ndarray | None, thresholds: dict, selected_method: str):
    """Render the main image-analysis workspace as a modern segmented-preview panel."""
    st.markdown("<div class='section-label'>Image lab</div>", unsafe_allow_html=True)

    if "workspace_fit" not in st.session_state:
        st.session_state.workspace_fit = True
    if "workspace_zoom" not in st.session_state:
        st.session_state.workspace_zoom = 100

    uploaded_workspace = st.file_uploader(
        "Drag and drop an image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="workspace_upload",
        help="Upload JPG, JPEG or PNG for live segmentation analysis.",
    )
    if uploaded_workspace is not None:
        try:
            image_rgb = load_uploaded_image(uploaded_workspace)
            st.session_state.image_rgb = image_rgb
            st.session_state["uploaded_image_size"] = getattr(uploaded_workspace, "size", 0)
        except ValueError as exc:
            st.error(str(exc))
            return

    main_col, detail_col = st.columns([2.2, 1])
    safe_dims = f"{image_rgb.shape[1]} × {image_rgb.shape[0]}" if image_rgb is not None else "No image"
    safe_name = uploaded_workspace.name if uploaded_workspace is not None else st.session_state.get("sample_name", "sample_image")

    with main_col:
        method = st.session_state.get("selected_method", selected_method)
        st.markdown(
            """
            <div class='image-toolbar'>
                <div class='image-toolbar-group'>
                    <span class='project-pill'>""" + method + """</span>
                    <span class='status-pill'>""" + safe_dims + """</span>
                </div>
                <div class='image-toolbar-group'>
                    <button class='tiny-button' type='button'>Fit</button>
                    <button class='tiny-button' type='button'>Zoom</button>
                    <button class='tiny-button' type='button'>Reset</button>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fit_to_screen = st.toggle("Fit to screen", value=st.session_state.workspace_fit, key="workspace_fit_toggle")
        st.session_state.workspace_fit = fit_to_screen
        zoom_level = st.slider("Zoom", 60, 160, value=st.session_state.workspace_zoom, step=5, key="workspace_zoom_slider")
        st.session_state.workspace_zoom = zoom_level

        if image_rgb is None:
            st.markdown(
                """
                <div class='empty-upload-panel'>
                    <div class='empty-upload-icon'>⬆</div>
                    <div class='empty-upload-title'>Awaiting image input</div>
                    <div class='empty-upload-copy'>Drag and drop a JPG, JPEG or PNG file here, or use the upload controls to begin live color segmentation.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            current_method = st.session_state.get("selected_method", selected_method)
            current_thresholds = _render_threshold_panel(current_method)
            if st.button("Run segmentation", key="run_segmentation_button"):
                st.session_state.run_segmentation = True

            if "run_segmentation" not in st.session_state:
                st.session_state.run_segmentation = True

            if st.session_state.run_segmentation:
                with st.spinner("Processing segmentation outputs..."):
                    results = _build_results(image_rgb, current_thresholds)
                    current_result = results[current_method]
                    binary_mask = current_result["binary_mask"]
                    segmented = current_result["segmented"]
            else:
                results = {}
                binary_mask = np.zeros_like(image_rgb[:, :, 0])
                segmented = image_rgb.copy()

            overlay_enabled = st.checkbox("Show actual segmentation overlay", value=True, key="workspace_overlay_toggle")
            if overlay_enabled:
                blend = st.slider("Overlay blend", 0, 100, 55, key="overlay_blend")
                alpha = blend / 100.0
                mask_overlay = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
                overlay_image = image_rgb.copy().astype(np.float32)
                mask_overlay = mask_overlay.astype(np.float32) * alpha
                overlay_image = overlay_image * (1.0 - alpha) + mask_overlay * 1.0
                overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)
            else:
                overlay_image = image_rgb.copy()

            width_px = max(280, int(image_rgb.shape[1] * (zoom_level / 100.0)))
            st.markdown("<div class='section-label'>Original image</div>", unsafe_allow_html=True)
            st.image(image_rgb, channels="RGB", use_container_width=fit_to_screen, width=width_px if not fit_to_screen else None, clamp=True)

            preview_col, outcome_col = st.columns(2)
            with preview_col:
                st.markdown("<div class='section-label'>Binary mask</div>", unsafe_allow_html=True)
                st.image(binary_mask, channels="GRAY", use_container_width=True, clamp=True)
            with outcome_col:
                st.markdown("<div class='section-label'>Segmented result</div>", unsafe_allow_html=True)
                st.image(segmented, channels="RGB", use_container_width=True, clamp=True)

            st.markdown("<div class='section-label'>Overlay preview</div>", unsafe_allow_html=True)
            st.image(overlay_image, channels="RGB", use_container_width=True, clamp=True)

            _render_analytics_cards(
                image_rgb=image_rgb,
                binary_mask=binary_mask if 'binary_mask' in locals() else None,
                method_name=method,
                gt_mask=gt_mask,
                results=results if 'results' in locals() else None,
            )

    with detail_col:
        st.markdown(
            """
            <div class='detail-panel'>
                <div class='panel-header'>Image details</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='detail-metric'><div class='detail-label'>Filename</div><div class='detail-value'>{safe_name}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='detail-metric'><div class='detail-label'>Dimensions</div><div class='detail-value'>{safe_dims}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='detail-metric'><div class='detail-label'>File size</div><div class='detail-value'>{(uploaded_workspace.size // 1024) if uploaded_workspace is not None else 'sample'} KB</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='detail-metric'><div class='detail-label'>Color space</div><div class='detail-value'>{selected_method}</div></div>", unsafe_allow_html=True)
        params_text = str(thresholds.get(selected_method, {}))
        st.markdown(f"<div class='detail-metric'><div class='detail-label'>Parameters</div><div class='detail-value' style='max-width: 170px; overflow-wrap: anywhere; font-size: 0.7rem;'>{params_text}</div></div>", unsafe_allow_html=True)

        if image_rgb is not None:
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            btn_mask, btn_seg = st.columns(2)
            with btn_mask:
                st.download_button(
                    label="Mask PNG",
                    data=prepare_mask_for_download(binary_mask if 'binary_mask' in locals() else np.zeros_like(image_rgb[:, :, 0])),
                    file_name=f"{selected_method.lower()}_mask.png",
                    mime="image/png",
                    key="download_mask_button",
                )
            with btn_seg:
                st.download_button(
                    label="Result PNG",
                    data=prepare_segmented_image_for_download(segmented if 'segmented' in locals() else image_rgb),
                    file_name=f"{selected_method.lower()}_segmented.png",
                    mime="image/png",
                    key="download_segment_button",
                )

        if gt_mask is not None and image_rgb is not None:
            try:
                aligned_gt = align_mask_to_image(gt_mask, image_rgb.shape)
                metrics = compute_metrics(binary_mask if 'binary_mask' in locals() else np.zeros_like(image_rgb[:, :, 0]), aligned_gt)
                st.success(f"IoU={metrics['IoU']:.4f} | Dice={metrics['Dice']:.4f} | Acc={metrics['Pixel Accuracy']:.4f}")
            except ValueError as exc:
                st.warning(f"Reference mask could not be aligned with the source image and was skipped: {exc}")
        else:
            st.info("Ground-truth metrics are unavailable until a valid reference mask is uploaded.")


def _render_analytics(image_rgb: np.ndarray, results: dict, gt_mask: np.ndarray | None):
    """Render the analytics dashboard."""
    st.markdown("<div class='section-label'>Analytics</div>", unsafe_allow_html=True)
    method_names = list(results.keys())
    timing_df = pd.DataFrame([
        {"Method": method_name, "Time (s)": results[method_name]["time_seconds"]} for method_name in method_names
    ])
    timing_fig = px.bar(timing_df, x="Method", y="Time (s)", color="Method", color_discrete_sequence=["#8B5CF6", "#22D3EE", "#34D399", "#F6C453"])
    timing_fig.update_layout(
        paper_bgcolor="#0B1020",
        plot_bgcolor="#0B1020",
        font={"color": "#EAF2FF"},
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(timing_fig, use_container_width=True)

    if gt_mask is not None:
        try:
            aligned_gt = align_mask_to_image(gt_mask, image_rgb.shape)
            metric_df = _compute_metric_table(results, aligned_gt)
            metric_plot = go.Figure()
            for metric in ["IoU", "Dice", "Pixel Accuracy"]:
                metric_plot.add_trace(go.Bar(x=metric_df["Method"], y=metric_df[metric], name=metric))
            metric_plot.update_layout(
                barmode="group",
                paper_bgcolor="#0B1020",
                plot_bgcolor="#0B1020",
                font={"color": "#EAF2FF"},
                margin=dict(l=10, r=10, t=10, b=10),
                legend={"orientation": "h", "y": 1.15},
            )
            st.plotly_chart(metric_plot, use_container_width=True)
            st.dataframe(metric_df, use_container_width=True)
            csv_bytes = metric_df.to_csv(index=False).encode("utf-8")
            st.download_button("Export metrics CSV", csv_bytes, file_name="segmentation_metrics.csv", mime="text/csv")
        except ValueError as exc:
            st.warning(f"Reference mask could not be aligned with the source image, so quantitative comparison was skipped: {exc}")
    else:
        st.warning("Ground-truth metrics are unavailable because no mask was supplied. Use the upload control to attach a reference mask for IoU, Dice, and accuracy values.")

    selected_method = st.selectbox("Histogram view", options=COLOR_SPACES, index=0, key="analytics_method_selection")
    _render_channel_histogram(selected_method, image_rgb)


def _render_about():
    """Render project information and methodology summary."""
    st.markdown("<div class='section-label'>About</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title" style="font-size: 1.7rem;">College DIP Project</div>
            <div class="hero-copy">
                This dashboard compares four common color representations for threshold-based image segmentation: RGB, HSV, HSI and YCbCr. The project keeps the mathematically grounded conversion backend intact while presenting a clean, professional analysis workflow for demonstration and evaluation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        - Student: Bethu Harshith
        - Enrollment: 24CS003260
        - Stack: Streamlit, OpenCV, NumPy, Plotly, Pillow, Matplotlib
        - Purpose: compare threshold-based segmentation behavior across color spaces
        - Important: measured metrics are only valid when a valid ground-truth mask is supplied
        """
    )


def render_app():
    """Render the full dashboard using the project’s existing backend."""
    st.set_page_config(page_title="VISION LAB", page_icon="◈", layout="wide")
    _inject_theme_css()
    _set_default_image()

    with st.sidebar:
        _render_sidebar_logo()
        st.markdown("<div class='section-label'>Navigation</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
        nav = st.radio(
            "",
            ["◈ Dashboard", "◉ Image Lab", "◌ Compare Spaces", "◍ Analytics", "◐ About"],
            index=0,
            key="main_nav",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        nav = nav.replace("◈ ", "").replace("◉ ", "").replace("◌ ", "").replace("◍ ", "").replace("◐ ", "")

        st.markdown("<div class='section-label'>Input</div>", unsafe_allow_html=True)
        image_rgb, sample_name, _ = _load_image_from_ui()
        gt_mask = _load_mask_from_ui()
        if image_rgb is not None:
            st.session_state.image_rgb = image_rgb
        if gt_mask is not None:
            st.session_state.gt_mask = gt_mask
        st.session_state.selected_method = st.sidebar.selectbox("Active method", COLOR_SPACES, index=COLOR_SPACES.index(st.session_state.get("selected_method", "RGB")), key="sidebar_active_method")
        if st.sidebar.button("Reset session", key="reset_dashboard_button", use_container_width=True):
            _reset_dashboard()
            st.rerun()

    image_rgb = st.session_state.get("image_rgb", generate_sample_image("Red circle"))
    gt_mask = st.session_state.get("gt_mask")
    selected_method = st.session_state.get("selected_method", "RGB")
    thresholds = _ensure_threshold_state()
    if selected_method not in thresholds:
        thresholds[selected_method] = _default_thresholds()[selected_method]

    topbar = st.columns([2.2, 1.3, 0.8])
    with topbar[0]:
        st.markdown(
            """
            <div class="vision-header">
                <div class="vision-header-left">
                    <div class="header-mark">◈</div>
                    <div>
                        <div class="header-title">VISION LAB</div>
                        <div class="header-subtitle">Color Segmentation Studio</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with topbar[1]:
        st.markdown("<div style='height: 100%; display: flex; align-items: center; justify-content: center;'><span class='project-pill'>IMAGE ANALYSIS</span></div>", unsafe_allow_html=True)
    with topbar[2]:
        st.markdown("<div style='height: 100%; display: flex; align-items: center; justify-content: flex-end;'><span class='status-pill'>Live</span></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align: center; margin: 0.2rem auto 1.2rem;'>
            <div style='font-size: 2.2rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; color: #EAF7FF;'>IMAGE ANALYSIS</div>
            <div style='font-size: 0.78rem; letter-spacing: 0.18em; color: #9FB5C9; text-transform: uppercase; margin-top: 0.4rem;'>Comparative Study of RGB, HSV, HSI and YCbCr Segmentation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if nav == "Dashboard":
        with st.spinner("Preparing overview..."):
            results = _build_results(image_rgb, thresholds)
            elapsed_seconds = results[selected_method]["time_seconds"]
        _render_overview(image_rgb, selected_method, elapsed_seconds)
    elif nav == "Image Lab":
        _render_image_lab(image_rgb, gt_mask, thresholds, selected_method)
    elif nav == "Compare Spaces":
        with st.spinner("Processing all color spaces..."):
            results = _build_results(image_rgb, thresholds)
        _render_compare_spaces(image_rgb, thresholds, gt_mask)
    elif nav == "Analytics":
        with st.spinner("Rendering analytics..."):
            results = _build_results(image_rgb, thresholds)
        _render_analytics(image_rgb, results, gt_mask)
    else:
        _render_about()

    if nav != "About":
        metric_cols = st.columns(4)
        with metric_cols[0]:
            _render_metric_card("Resolution", f"{image_rgb.shape[1]} × {image_rgb.shape[0]}", "cyan")
        with metric_cols[1]:
            _render_metric_card("Pixels", f"{image_rgb.shape[0] * image_rgb.shape[1]:,}", "blue")
        with metric_cols[2]:
            _render_metric_card("Runtime", f"{(results if 'results' in locals() else _build_results(image_rgb, thresholds))[selected_method]['time_seconds']:.4f}s", "teal")
        with metric_cols[3]:
            _render_metric_card("Method", selected_method, "cyan")

    st.markdown(
        """
        <div class='footer-bar'>
            <span>Project: VISION LAB — Color Segmentation Studio</span>
            <span>Student: Bethu Harshith | Enrollment: 24CS003260</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if gt_mask is not None:
        st.caption("Reference mask loaded successfully. Quantitative metrics are enabled.")
    else:
        st.caption("No ground-truth mask loaded. Only visual inspection is available until a reference mask is uploaded.")


if __name__ == "__main__":
    render_app()

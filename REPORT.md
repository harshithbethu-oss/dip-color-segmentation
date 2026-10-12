# Report Preparation Template for the DIP Project

## 1. Abstract

This project presents a comparative study of image segmentation in the RGB, HSV, HSI and YCbCr color spaces. The study investigates how different color representations affect segmentation performance when simple threshold-based rules are applied to synthetic objects and color scenes. The project integrates a Python-based image-processing pipeline with a Streamlit interface, enabling threshold adjustment, visual inspection, and quantitative evaluation when a ground-truth mask is supplied. The work emphasizes reproducibility, mathematical correctness of the conversions, and honest reporting of measured results.

## 2. Problem Statement

Color-based image segmentation is strongly dependent on the color space used to represent the image. Different color spaces emphasize different visual cues such as intensity, hue, saturation, or chroma. In practical image-processing tasks, selecting an inappropriate color space can degrade segmentation quality, increase processing time, or produce false positives when lighting and object color vary. This project studies the behavior of RGB, HSV, HSI and YCbCr segmentation for simple synthetic objects and compares their performance under consistent thresholding conditions.

## 3. Objectives

1. To study and compare threshold-based segmentation in RGB, HSV, HSI and YCbCr color spaces.
2. To implement and verify mathematically consistent conversion functions between RGB and the selected color spaces.
3. To design a simple but reproducible evaluation workflow using synthetic images and ground-truth masks.
4. To build an interactive Streamlit interface for visual comparison and threshold tuning.
5. To assess input validation, timing, metric computation, and reporting reliability.
6. To produce a transparent submission report that separates measured results from qualitative observations.

## 4. Methodology

The project follows a structured image-processing workflow:

1. Input image acquisition from sample generation or user upload.
2. Conversion to the target color space using explicit formulas and normalized channel ranges.
3. Threshold range selection based on the foreground object pixels or target region.
4. Binary mask generation using channel-wise range checks.
5. Segmented image reconstruction using the resulting mask.
6. Quantitative evaluation using IoU, Dice coefficient and pixel accuracy, when a reference mask is available.
7. Runtime measurement using repeated processing and average timing.
8. Visualization of original images, channel information and result comparisons.

## 5. Literature Review

Color-space-based image segmentation is a standard topic in digital image processing, and the choice of representation significantly affects segmentation quality and robustness. RGB is the most direct representation but is sensitive to illumination because it stores intensity in each channel separately. HSV and HSI are often preferred for color-based thresholding because they separate chromatic information from brightness, making hue-based segmentation easier under controlled lighting. YCbCr is widely used in video and image compression pipelines because it separates luminance and chroma components, which helps isolate color information from brightness variations.

Relevant references include standard digital image processing texts and color-science material, such as:

- Gonzalez, R. C., & Woods, R. E. Digital Image Processing. Pearson.
- Russ, J. C. The Image Processing Handbook. CRC Press.
- OpenCV documentation on color conversions and image processing routines.

## 6. Experimental Setup

### Dataset
- Synthetic dataset generated in the project for robust, repeatable benchmarking.
- Included images: red circle, green square, color blocks, and multi-object scene.
- Ground-truth masks were created for each synthetic image.

### Software and tools
- Python 3.11+
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Streamlit
- Pillow
- Pytest

### Evaluation protocol
- Each method was evaluated with the same synthetic dataset.
- Thresholds were selected from target-pixel statistics using a consistent percentile-based range rule.
- Each method was run repeatedly and average processing time was recorded.
- Metrics recorded: IoU, Dice coefficient, pixel accuracy.

## 7. Results Template

Use the following structure when filling the final report. Replace the placeholders with measured values from the actual run.

### 7.1 Quantitative summary

| Method | Average Time (s) | Mean IoU | Mean Dice | Mean Pixel Accuracy |
| --- | ---: | ---: | ---: | ---: |
| RGB | [measure] | [measure] | [measure] | [measure] |
| HSV | [measure] | [measure] | [measure] | [measure] |
| HSI | [measure] | [measure] | [measure] | [measure] |
| YCbCr | [measure] | [measure] | [measure] | [measure] |

### 7.2 Per-image results

| Image | Method | IoU | Dice | Pixel Accuracy | Time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| [image name] | RGB | [measure] | [measure] | [measure] | [measure] |
| [image name] | HSV | [measure] | [measure] | [measure] | [measure] |
| [image name] | HSI | [measure] | [measure] | [measure] | [measure] |
| [image name] | YCbCr | [measure] | [measure] | [measure] | [measure] |

### 7.3 Results still needing measurement

The following items require the actual target dataset or experiment run before they can be reported as final findings:

- Best-performing method for the chosen benchmark set
- Average runtime ranking across all methods
- Final conclusion for robustness under illumination or object variation
- Any claims regarding universal superiority of one color space

## 8. Limitations

1. The synthetic dataset is small and does not represent all real-world segmentation scenarios.
2. Threshold-based methods depend strongly on object color and image illumination.
3. Hue-based representations may fail for grayscale, near-neutral, or saturated objects.
4. Ground-truth masks are required for reliable quantitative evaluation; without them, only visual inspection is valid.
5. Results are specific to the dataset used in this project and should not be generalized without further testing.

## 9. Conclusion

The comparative analysis demonstrates that each color space has strengths and weaknesses depending on the image content and the segmentation objective. This project provides a reproducible and transparent pipeline for comparing color-space segmentation methods, and it emphasizes the importance of validation, proper conversion logic, and honest interpretation of measured performance. A final conclusion about the best-performing method should be based on the actual benchmark measurements obtained from the target dataset, not on assumptions.

## 10. References

- Gonzalez, R. C., & Woods, R. E. Digital Image Processing. Pearson.
- Russ, J. C. The Image Processing Handbook. CRC Press.
- OpenCV documentation: Color conversions and image processing functions.

## 11. Submission note

This template intentionally leaves quantitative results as placeholders until the actual benchmark is run on the final target dataset. No fabricated findings are included.

# Comparative Study of RGB, HSV, HSI and YCbCr Segmentation

This project is a college-level digital image processing dashboard that compares threshold-based segmentation in RGB, HSV, HSI and YCbCr. The app combines a modern Streamlit interface with the verified OpenCV and NumPy image-processing backend. It supports image upload, threshold tuning, visual comparison, and quantitative evaluation when a ground-truth mask is provided.

## Student Information

- Student Name: Bethu Harshith
- Enrollment Number: 24CS003260

## Project Structure

```text
.
├── app.py
├── README.md
├── REPORT.md
├── requirements.txt
├── run_experiment.py
├── experimental_results/
├── src/
│   └── dip_project/
│       ├── __init__.py
│       ├── color_conversion.py
│       ├── experiment.py
│       ├── interface.py
│       ├── metrics.py
│       ├── segmentation.py
│       └── utils.py
├── tests/
│   ├── test_color_conversion.py
│   ├── test_experiment.py
│   ├── test_metrics.py
│   └── test_segmentation.py
├── .gitignore
└── .pytest_cache/
```

## Architecture Overview

The original backend remains intact while the front-end is redesigned into a premium scientific dashboard:

1. `color_conversion.py` handles RGB, HSV, HSI and YCbCr conversion and normalizes BGR/RGB channel ordering safely.
2. `segmentation.py` performs threshold-based mask generation for each color space.
3. `metrics.py` computes IoU, Dice coefficient and pixel accuracy.
4. `interface.py` renders the dark-mode Streamlit product dashboard and analytics views.
5. `utils.py` handles uploads, sample generation, and export functions.
6. `experiment.py` creates a reproducible synthetic evaluation workflow and saves benchmark artifacts.

## Design Features

- Deep navy scientific theme with violet and cyan accents
- Multi-page Streamlit dashboard with Overview, Image Lab, Compare Spaces, Analytics and About views
- Upload and sample-image workflow with clean validation states
- Adjustable threshold controls with consistent channel ranges
- Live binary mask and segmented output previews
- Plotly analytics and CSV export for measured results
- Responsive desktop layout suitable for project demonstration

## Installation

1. Open the project folder in VS Code.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Verify the environment and test suite:

```bash
pytest -q
```

## Run the Dashboard

From the project root, start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Run the Experiment Pipeline

To generate the synthetic benchmark outputs and save the summary CSV and report:

```bash
python run_experiment.py
```

This writes the results into the `experimental_results/` folder.

## Notes

- Quantitative metrics are only valid when a ground-truth mask is available.
- No method is assumed to be universally best; performance depends on the image content and threshold choices.
- This project intentionally reports measured results honestly and keeps the synthetic benchmark separate from any real-world claims.

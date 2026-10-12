import numpy as np

from src.dip_project.experiment import generate_experiment_dataset, select_thresholds


def test_experiment_dataset_is_reproducible_and_complete():
    dataset = generate_experiment_dataset()
    assert len(dataset) == 4
    for item in dataset:
        assert item["image"].ndim == 3
        assert item["image"].shape[2] == 3
        assert item["mask"].shape[:2] == item["image"].shape[:2]
        assert item["source"] in {"synthetic", "generated"}


def test_threshold_selection_returns_method_specific_ranges():
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    image[15:45, 15:45] = [255, 0, 0]

    for method_name in ["RGB", "HSV", "HSI", "YCbCr"]:
        thresholds = select_thresholds(image, method_name)
        assert isinstance(thresholds, dict)
        assert len(thresholds) >= 3
        assert all(isinstance(v, tuple) and len(v) == 2 for k, v in thresholds.items() if k not in {"wrap_hue"})

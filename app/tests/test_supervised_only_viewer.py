import csv
import json
from pathlib import Path

from app.loader import load_index
from app.summary import dataset_counts, metric_summary


def test_old_reference_free_csv_not_shown_as_metrics(tmp_path):
    folder = tmp_path / "runs" / "otsu" / "predictions"
    folder.mkdir(parents=True)
    (folder / "slide.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []})
    )
    old = tmp_path / "old.csv"
    with old.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "filename",
                "run_id",
                "evaluation_mode",
                "num_objects",
                "coverage_ratio",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "slide.geojson",
                "run_id": "otsu",
                "evaluation_mode": "unsupervised",
                "num_objects": 3,
                "coverage_ratio": 0.5,
            }
        )
    index = load_index(tmp_path / "runs", metrics_file=old)
    assert index.available_metrics == []
    assert index.metrics_available is False
    assert "otsu" in index.methods  # Prediction-only browsing remains available.
    assert index.wsis[0].run_status == "prediction_only"
    assert dataset_counts(index).prediction_only_slides == 1
    assert metric_summary(index) == []
    assert "unsupervised" not in index.scores["slide"]["otsu"]["metrics"]


def test_gt_suffix_matches_viewer_and_ambiguous_annotations_rejected(tmp_path):
    import pytest

    root = tmp_path / "runs" / "otsu" / "predictions"
    root.mkdir(parents=True)
    payload = {"type": "FeatureCollection", "features": []}
    (root / "sample.geojson").write_text(json.dumps(payload))
    gt = tmp_path / "truth"
    gt.mkdir()
    (gt / "sample_gt.geojson").write_text(json.dumps(payload))
    index = load_index(tmp_path / "runs", ground_truth_dir=gt)
    assert index.wsis[0].has_ground_truth
    assert index.wsis[0].ground_truth_url == "/api/ground-truth/sample"
    (gt / "sample.geojson").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Ambiguous ground truth"):
        load_index(tmp_path / "runs", ground_truth_dir=gt)

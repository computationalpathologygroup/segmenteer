"""Unit tests for segmenteer.metrics.

Geometry is kept intentionally simple (axis-aligned rectangles) so that
all expected values can be derived by hand.

Coordinate convention
---------------------
All polygons use pixel-space (x=col, y=row) with rings closed by repeating
the first vertex.

Shapes used
-----------
FULL  : 100×100 square  → area=10 000, perimeter=400
INNER : 50×50 square centred at (50,50) → area=2 500, perimeter=200
LEFT  : left 50×100 strip → area=5 000
RIGHT : right 50×100 strip → area=5 000
"""

from __future__ import annotations

import math

import pytest

from segmenteer.metrics.supervised import (SupervisedMetrics, _union,
                                           compute_all_supervised_metrics,
                                           compute_dice, compute_hausdorff,
                                           compute_iou,
                                           compute_over_segmentation_rate,
                                           compute_precision, compute_recall,
                                           compute_under_segmentation_rate)
from segmenteer.metrics.unsupervised import (UnsupervisedMetrics,
                                             compute_unsupervised_metrics)

# ---------------------------------------------------------------------------
# GeoJSON fixture helpers
# ---------------------------------------------------------------------------


def _fc(*rings: list[list[float]]) -> dict:
    """Build a GeoJSON FeatureCollection from one polygon ring per argument."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {},
            }
            for ring in rings
        ],
    }


def _rect(x0, y0, x1, y1) -> list[list[float]]:
    """Closed ring for an axis-aligned rectangle [x0,y0]→[x1,y1]."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


FULL = _fc(_rect(0, 0, 100, 100))  # 100×100 = area 10 000
INNER = _fc(_rect(25, 25, 75, 75))  # 50×50  = area  2 500, inside FULL
LEFT = _fc(_rect(0, 0, 50, 100))  # 50×100 = area  5 000, left half
RIGHT = _fc(_rect(50, 0, 100, 100))  # 50×100 = area  5 000, right half
EMPTY = {"type": "FeatureCollection", "features": []}


# ---------------------------------------------------------------------------
# SupervisedMetrics dataclass
# ---------------------------------------------------------------------------


class TestSupervisedMetricsDataclass:
    def test_frozen(self):
        m = SupervisedMetrics.zero()
        with pytest.raises((TypeError, AttributeError)):
            m.dice = 0.5  # type: ignore[misc]

    def test_zero(self):
        m = SupervisedMetrics.zero()
        assert m.dice == 0.0
        assert m.iou == 0.0
        assert math.isinf(m.hausdorff)
        assert m.precision == 0.0
        assert m.recall == 0.0


# ---------------------------------------------------------------------------
# _union helper
# ---------------------------------------------------------------------------


class TestUnion:
    def test_empty_returns_none(self):
        assert _union(EMPTY) is None

    def test_single_polygon(self):
        u = _union(FULL)
        assert u is not None
        assert pytest.approx(u.area, rel=1e-6) == 10_000.0

    def test_two_disjoint_polygons(self):
        fc = _fc(_rect(0, 0, 10, 10), _rect(20, 20, 30, 30))
        u = _union(fc)
        assert pytest.approx(u.area, rel=1e-6) == 200.0  # 100 + 100

    def test_invalid_geometry_skipped(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[]]},
                    "properties": {},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_rect(0, 0, 10, 10)],
                    },
                    "properties": {},
                },
            ],
        }
        u = _union(fc)
        assert u is not None
        assert u.area > 0


# ---------------------------------------------------------------------------
# compute_dice
# ---------------------------------------------------------------------------


class TestDice:
    def test_perfect_overlap(self):
        assert pytest.approx(compute_dice(FULL, FULL), abs=1e-9) == 1.0

    def test_no_overlap(self):
        assert pytest.approx(compute_dice(LEFT, RIGHT), abs=1e-9) == 0.0

    def test_both_empty(self):
        assert compute_dice(EMPTY, EMPTY) == 1.0

    def test_pred_empty(self):
        assert compute_dice(EMPTY, FULL) == 0.0

    def test_gt_empty(self):
        assert compute_dice(FULL, EMPTY) == 0.0

    def test_partial_overlap(self):
        # pred=FULL (10 000), gt=INNER (2 500), intersection=2 500
        # dice = 2*2500 / (10000+2500) = 5000/12500 = 0.4
        assert pytest.approx(compute_dice(FULL, INNER), rel=1e-6) == 0.4


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


class TestIoU:
    def test_perfect_overlap(self):
        assert pytest.approx(compute_iou(FULL, FULL), abs=1e-9) == 1.0

    def test_no_overlap(self):
        assert pytest.approx(compute_iou(LEFT, RIGHT), abs=1e-9) == 0.0

    def test_both_empty(self):
        assert compute_iou(EMPTY, EMPTY) == 1.0

    def test_partial_overlap(self):
        # pred=FULL, gt=INNER → intersection=2500, union=FULL.area=10000
        assert pytest.approx(compute_iou(FULL, INNER), rel=1e-6) == 0.25

    def test_consistency_with_dice(self):
        # For A ⊆ B:  iou = |A| / |B|,  dice = 2|A| / (|A|+|B|)
        # dice = 2*iou / (1 + iou)
        d = compute_dice(FULL, INNER)
        iou = compute_iou(FULL, INNER)
        assert pytest.approx(d, rel=1e-6) == 2 * iou / (1 + iou)


# ---------------------------------------------------------------------------
# compute_precision / compute_recall
# ---------------------------------------------------------------------------


class TestPrecisionRecall:
    def test_perfect(self):
        assert pytest.approx(compute_precision(FULL, FULL), abs=1e-9) == 1.0
        assert pytest.approx(compute_recall(FULL, FULL), abs=1e-9) == 1.0

    def test_recall_is_one_when_pred_contains_gt(self):
        # FULL contains INNER entirely → recall = 1
        assert pytest.approx(compute_recall(FULL, INNER), rel=1e-6) == 1.0

    def test_precision_drops_when_pred_larger(self):
        # pred=FULL (10 000), gt=INNER (2 500) → precision = 2500/10000 = 0.25
        assert pytest.approx(compute_precision(FULL, INNER), rel=1e-6) == 0.25

    def test_empty_pred(self):
        assert compute_precision(EMPTY, FULL) == 0.0
        assert compute_recall(EMPTY, FULL) == 0.0

    def test_empty_gt(self):
        assert compute_precision(FULL, EMPTY) == 0.0
        assert compute_recall(FULL, EMPTY) == 0.0


# ---------------------------------------------------------------------------
# compute_over/under_segmentation_rate
# ---------------------------------------------------------------------------


class TestSegmentationRates:
    def test_perfect_no_over_or_under(self):
        assert (
            pytest.approx(compute_over_segmentation_rate(FULL, FULL), abs=1e-9) == 0.0
        )
        assert (
            pytest.approx(compute_under_segmentation_rate(FULL, FULL), abs=1e-9) == 0.0
        )

    def test_no_overlap_full_under(self):
        # pred=LEFT, gt=RIGHT: FN = RIGHT.area = 5000, target = 5000 → under = 1.0
        assert (
            pytest.approx(compute_under_segmentation_rate(LEFT, RIGHT), rel=1e-6) == 1.0
        )

    def test_no_overlap_full_over(self):
        # pred=RIGHT, gt=LEFT: FP = RIGHT.area = 5000, target = 5000 → over = 1.0
        assert (
            pytest.approx(compute_over_segmentation_rate(RIGHT, LEFT), rel=1e-6) == 1.0
        )

    def test_pred_contains_gt_no_under(self):
        # FULL contains INNER → no under-seg
        assert (
            pytest.approx(compute_under_segmentation_rate(FULL, INNER), abs=1e-9) == 0.0
        )

    def test_pred_contains_gt_over_rate(self):
        # FP = FULL - INNER = 7500, target = INNER = 2500 → over = 3.0
        assert (
            pytest.approx(compute_over_segmentation_rate(FULL, INNER), rel=1e-6) == 3.0
        )

    def test_over_plus_recall_equals_pred_area_ratio(self):
        # precision + over_seg_rate != 1 in general, but:
        # recall + under_seg_rate == 1  (TP/GT + FN/GT = 1)
        recall = compute_recall(LEFT, FULL)
        under = compute_under_segmentation_rate(LEFT, FULL)
        assert pytest.approx(recall + under, abs=1e-9) == 1.0


# ---------------------------------------------------------------------------
# compute_hausdorff
# ---------------------------------------------------------------------------


class TestHausdorff:
    def test_identical_shapes_zero(self):
        # Exact same polygon → Hausdorff distance should be 0
        assert pytest.approx(compute_hausdorff(FULL, FULL), abs=1e-9) == 0.0

    def test_empty_returns_inf(self):
        assert math.isinf(compute_hausdorff(EMPTY, FULL))
        assert math.isinf(compute_hausdorff(FULL, EMPTY))

    def test_disjoint_positive(self):
        # LEFT and RIGHT share the x=50 edge but centroids are 50 units apart
        h = compute_hausdorff(LEFT, RIGHT)
        assert h > 0

    def test_symmetry(self):
        h1 = compute_hausdorff(FULL, INNER)
        h2 = compute_hausdorff(INNER, FULL)
        assert pytest.approx(h1, rel=1e-6) == h2


# ---------------------------------------------------------------------------
# compute_all_supervised_metrics
# ---------------------------------------------------------------------------


class TestComputeAll:
    def test_perfect_match(self):
        m = compute_all_supervised_metrics(FULL, FULL)
        assert pytest.approx(m.dice, abs=1e-9) == 1.0
        assert pytest.approx(m.iou, abs=1e-9) == 1.0
        assert pytest.approx(m.precision, abs=1e-9) == 1.0
        assert pytest.approx(m.recall, abs=1e-9) == 1.0
        assert pytest.approx(m.over_segmentation_rate, abs=1e-9) == 0.0
        assert pytest.approx(m.under_segmentation_rate, abs=1e-9) == 0.0
        assert pytest.approx(m.hausdorff, abs=1e-9) == 0.0

    def test_pixel_metrics_zero_without_shape(self):
        m = compute_all_supervised_metrics(FULL, FULL)
        assert m.pixel_accuracy == 0.0
        assert m.mae == 0.0
        assert m.balanced_error_rate == 0.0

    def test_pixel_accuracy_perfect_match(self):
        m = compute_all_supervised_metrics(FULL, FULL, image_shape=(100, 100))
        assert pytest.approx(m.pixel_accuracy, abs=1e-9) == 1.0
        assert pytest.approx(m.mae, abs=1e-9) == 0.0

    def test_returns_frozen_dataclass(self):
        m = compute_all_supervised_metrics(FULL, FULL)
        assert isinstance(m, SupervisedMetrics)
        with pytest.raises((TypeError, AttributeError)):
            m.dice = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# UnsupervisedMetrics dataclass
# ---------------------------------------------------------------------------


class TestUnsupervisedMetricsDataclass:
    def test_frozen(self):
        m = UnsupervisedMetrics.zero()
        with pytest.raises((TypeError, AttributeError)):
            m.num_objects = 5  # type: ignore[misc]

    def test_zero(self):
        m = UnsupervisedMetrics.zero()
        assert m.num_objects == 0
        assert m.coverage_ratio == 0.0


# ---------------------------------------------------------------------------
# compute_unsupervised_metrics
# ---------------------------------------------------------------------------


class TestComputeUnsupervised:
    def test_empty_collection(self):
        m = compute_unsupervised_metrics(EMPTY, image_area=10_000)
        assert m.num_objects == 0
        assert m.coverage_ratio == 0.0
        assert m.total_area == 0.0

    def test_single_square_area(self):
        m = compute_unsupervised_metrics(FULL, image_area=10_000)
        assert m.num_objects == 1
        assert pytest.approx(m.total_area, rel=1e-6) == 10_000.0
        assert pytest.approx(m.mean_area, rel=1e-6) == 10_000.0
        assert pytest.approx(m.std_area, abs=1e-9) == 0.0
        assert pytest.approx(m.coverage_ratio, rel=1e-6) == 1.0

    def test_single_square_perimeter(self):
        m = compute_unsupervised_metrics(FULL, image_area=10_000)
        assert pytest.approx(m.total_perimeter, rel=1e-6) == 400.0
        assert pytest.approx(m.mean_perimeter, rel=1e-6) == 400.0

    def test_single_square_compactness(self):
        # compactness = 4π * A / P² = 4π * 10000 / 160000 = π/4
        m = compute_unsupervised_metrics(FULL, image_area=10_000)
        assert pytest.approx(m.mean_compactness, rel=1e-5) == math.pi / 4

    def test_single_square_solidity(self):
        # A square equals its convex hull → solidity = 1
        m = compute_unsupervised_metrics(FULL, image_area=10_000)
        assert pytest.approx(m.mean_solidity, rel=1e-6) == 1.0

    def test_two_squares(self):
        fc = _fc(_rect(0, 0, 10, 10), _rect(20, 20, 30, 30))  # two 10×10
        m = compute_unsupervised_metrics(fc, image_area=1_000)
        assert m.num_objects == 2
        assert pytest.approx(m.total_area, rel=1e-6) == 200.0
        assert pytest.approx(m.mean_area, rel=1e-6) == 100.0
        assert pytest.approx(m.std_area, abs=1e-9) == 0.0
        assert pytest.approx(m.coverage_ratio, rel=1e-6) == 0.2

    def test_coverage_above_one_allowed(self):
        # image_area smaller than actual tissue area — shouldn't crash
        m = compute_unsupervised_metrics(FULL, image_area=5_000)
        assert m.coverage_ratio == pytest.approx(2.0, rel=1e-6)

    def test_zero_image_area_no_crash(self):
        m = compute_unsupervised_metrics(FULL, image_area=0)
        assert m.coverage_ratio == 0.0

    def test_min_max_area(self):
        fc = _fc(_rect(0, 0, 10, 10), _rect(0, 0, 20, 20))  # 100 and 400
        m = compute_unsupervised_metrics(fc, image_area=10_000)
        assert pytest.approx(m.min_area, rel=1e-6) == 100.0
        assert pytest.approx(m.max_area, rel=1e-6) == 400.0

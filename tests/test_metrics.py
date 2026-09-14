"""
Test metrics for ISIC Skin Lesion Detection and Segmentation.

Kiểm tra:
1. Detection metrics:
   - IoU
   - Precision
   - Recall
   - F1-score
   - Mean IoU
   - Aggregate metrics

2. Segmentation metrics:
   - Dice
   - IoU
   - Precision
   - Recall
   - F1-score
   - Accuracy
   - Aggregate metrics
"""

import sys
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT METRICS
# ============================================================

from evaluation.detection_metrics import (
    calculate_iou,
    calculate_precision,
    calculate_recall,
    calculate_f1_score as calculate_f1,
    calculate_mean_iou,
    calculate_detection_metrics,
)

from evaluation.segmentation_metrics import (
    calculate_dice as dice_score,
    calculate_iou as iou_score,
    calculate_precision as precision_score,
    calculate_recall as recall_score,
    calculate_f1_score as f1_score,
    calculate_accuracy as accuracy_score,
    calculate_segmentation_metrics,
)


# ============================================================
# DETECTION METRICS
# ============================================================

def test_detection_iou():
    """
    Kiểm tra Intersection over Union.
    """

    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]

    iou = calculate_iou(box1, box2)

    expected = 2500 / 17500

    assert abs(iou - expected) < 1e-6

    print("[PASS] detection IoU")


def test_detection_precision():
    """
    Precision = TP / (TP + FP)
    """

    precision = calculate_precision(
        true_positive=8,
false_positive=2
    )

    assert abs(precision - 0.8) < 1e-6

    print("[PASS] detection Precision")


def test_detection_recall():
    """
    Recall = TP / (TP + FN)
    """

    recall = calculate_recall(
      true_positive=8,
false_negative=2
    )

    assert abs(recall - 0.8) < 1e-6

    print("[PASS] detection Recall")


def test_detection_f1():
    """
    F1-score là harmonic mean của Precision và Recall.
    """

    f1 = calculate_f1(
        precision=0.8,
        recall=0.8
    )

    assert abs(f1 - 0.8) < 1e-6

    print("[PASS] detection F1")


def test_detection_mean_iou():
    """
    Kiểm tra Mean IoU.
    """

    values = [
        0.5,
        0.7,
        0.9
    ]

    mean_iou = calculate_mean_iou(values)

    expected = (0.5 + 0.7 + 0.9) / 3

    assert abs(mean_iou - expected) < 1e-6

    print("[PASS] detection Mean IoU")


def test_detection_aggregate_metrics():
    """
    Kiểm tra tổng hợp Detection Metrics.
    """

    metrics = calculate_detection_metrics(
    true_positive=2,
    false_positive=1,
    false_negative=0,
    ious=[0.8, 0.6]
)

    assert "mean_iou" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics

    assert metrics["mean_iou"] == 0.7

    print("[PASS] detection aggregate metrics")


# ============================================================
# SEGMENTATION DATA
# ============================================================

def create_test_masks():
    """
    Tạo ground-truth mask và prediction mask giả.

    0 = background
    1 = lesion
    """

    ground_truth = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint8)

    prediction = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint8)

    return ground_truth, prediction


# ============================================================
# SEGMENTATION METRICS
# ============================================================

def test_segmentation_dice():
    """
    Kiểm tra Dice Score.
    """

    ground_truth, prediction = create_test_masks()

    dice = dice_score(
        ground_truth,
        prediction
    )

    # GT có 4 pixel lesion
    # Prediction có 3 pixel lesion
    # Intersection = 3
    # Dice = 2*3/(4+3) = 6/7
    expected = 6 / 7

    assert abs(dice - expected) < 1e-6

    print("[PASS] segmentation Dice")


def test_segmentation_iou():
    """
    Kiểm tra IoU của segmentation.
    """

    ground_truth, prediction = create_test_masks()

    iou = iou_score(
        ground_truth,
        prediction
    )

    # Intersection = 3
    # Union = 4
    # IoU = 3/4
    expected = 0.75

    assert abs(iou - expected) < 1e-6

    print("[PASS] segmentation IoU")


def test_segmentation_precision():
    """
    Kiểm tra Precision của segmentation.
    """

    ground_truth, prediction = create_test_masks()

    precision = precision_score(
        ground_truth,
        prediction
    )

    # TP = 3
    # FP = 0
    # Precision = 1
    expected = 1.0

    assert abs(precision - expected) < 1e-6

    print("[PASS] segmentation Precision")


def test_segmentation_recall():
    """
    Kiểm tra Recall của segmentation.
    """

    ground_truth, prediction = create_test_masks()

    recall = recall_score(
        ground_truth,
        prediction
    )

    # TP = 3
    # FN = 1
    # Recall = 3/4
    expected = 0.75

    assert abs(recall - expected) < 1e-6

    print("[PASS] segmentation Recall")


def test_segmentation_f1():
    """
    Kiểm tra F1-score của segmentation.
    """

    ground_truth, prediction = create_test_masks()

    f1 = f1_score(
        ground_truth,
        prediction
    )

    # Precision = 1
    # Recall = 0.75
    # F1 = 2*(1*0.75)/(1+0.75)
    expected = 2 * (1 * 0.75) / (1 + 0.75)

    assert abs(f1 - expected) < 1e-6

    print("[PASS] segmentation F1")


def test_segmentation_accuracy():
    """
    Kiểm tra Accuracy.
    """

    ground_truth, prediction = create_test_masks()

    accuracy = accuracy_score(
        ground_truth,
        prediction
    )

    # Tổng 16 pixel
    # Sai 1 pixel
    # Accuracy = 15/16
    expected = 15 / 16

    assert abs(accuracy - expected) < 1e-6

    print("[PASS] segmentation Accuracy")


def test_segmentation_aggregate_metrics():
    """
    Kiểm tra hàm tính toàn bộ segmentation metrics.
    """

    ground_truth, prediction = create_test_masks()

    metrics = calculate_segmentation_metrics(
        ground_truth,
        prediction
    )

    assert "dice" in metrics
    assert "iou" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "accuracy" in metrics

    assert abs(metrics["dice"] - (6 / 7)) < 1e-6
    assert abs(metrics["iou"] - 0.75) < 1e-6

    print("[PASS] segmentation aggregate metrics")


# ============================================================
# EDGE CASES
# ============================================================

def test_empty_segmentation_masks():
    """
    Kiểm tra trường hợp cả GT và prediction đều không có lesion.
    """

    ground_truth = np.zeros(
        (10, 10),
        dtype=np.uint8
    )

    prediction = np.zeros(
        (10, 10),
        dtype=np.uint8
    )

    dice = dice_score(
        ground_truth,
        prediction
    )

    iou = iou_score(
        ground_truth,
        prediction
    )

    assert dice == 1.0
    assert iou == 1.0

    print("[PASS] empty segmentation masks")


def test_perfect_segmentation():
    """
    Kiểm tra trường hợp prediction giống hoàn toàn ground truth.
    """

    ground_truth = np.array([
        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 0]
    ], dtype=np.uint8)

    prediction = ground_truth.copy()

    assert dice_score(
        ground_truth,
        prediction
    ) == 1.0

    assert iou_score(
        ground_truth,
        prediction
    ) == 1.0

    assert precision_score(
        ground_truth,
        prediction
    ) == 1.0

    assert recall_score(
        ground_truth,
        prediction
    ) == 1.0

    print("[PASS] perfect segmentation")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("METRICS TEST")
    print("=" * 60)

    print("\n--- Detection Metrics ---")

    test_detection_iou()
    test_detection_precision()
    test_detection_recall()
    test_detection_f1()
    test_detection_mean_iou()
    test_detection_aggregate_metrics()

    print("\n--- Segmentation Metrics ---")

    test_segmentation_dice()
    test_segmentation_iou()
    test_segmentation_precision()
    test_segmentation_recall()
    test_segmentation_f1()
    test_segmentation_accuracy()
    test_segmentation_aggregate_metrics()

    print("\n--- Edge Cases ---")

    test_empty_segmentation_masks()
    test_perfect_segmentation()

    print()
    print("=" * 60)
    print("ALL METRICS TESTS PASSED")
    print("=" * 60)

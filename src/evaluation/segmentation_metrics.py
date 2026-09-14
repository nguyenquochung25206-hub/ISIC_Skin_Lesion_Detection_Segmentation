"""
Segmentation metrics for U-Net.

Các chỉ số:
- Dice Coefficient
- IoU (Intersection over Union)
- Precision
- Recall
- F1-score
- Accuracy
"""

import numpy as np
from typing import Union


ArrayLike = Union[np.ndarray, list]


def _to_binary_mask(
    mask: ArrayLike,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Chuyển mask về dạng nhị phân 0/1.

    Parameters
    ----------
    mask : array-like
        Ground-truth mask hoặc predicted mask.

    threshold : float
        Ngưỡng để chuyển thành mask nhị phân.

    Returns
    -------
    np.ndarray
        Binary mask với giá trị 0 hoặc 1.
    """

    mask = np.asarray(mask)

    # Nếu mask là xác suất [0, 1]
    if mask.dtype.kind == "f":
        return (mask >= threshold).astype(np.uint8)

    # Nếu mask là ảnh 0-255
    if mask.max() > 1:
        return (mask > 127).astype(np.uint8)

    return (mask > 0).astype(np.uint8)


def calculate_dice(
    ground_truth: ArrayLike,
    prediction: ArrayLike,
    threshold: float = 0.5,
    smooth: float = 1e-7
) -> float:
    """
    Tính Dice Coefficient.

    Dice = 2 * Intersection / (Ground Truth + Prediction)

    Giá trị nằm trong khoảng 0 -> 1.
    """

    gt = _to_binary_mask(ground_truth, threshold)
    pred = _to_binary_mask(prediction, threshold)

    intersection = np.sum(gt * pred)

    dice = (
        2.0 * intersection + smooth
    ) / (
        np.sum(gt) + np.sum(pred) + smooth
    )

    return float(dice)


def calculate_iou(
    ground_truth: ArrayLike,
    prediction: ArrayLike,
    threshold: float = 0.5,
    smooth: float = 1e-7
) -> float:
    """
    Tính Intersection over Union (IoU).

    IoU = Intersection / Union

    Giá trị nằm trong khoảng 0 -> 1.
    """

    gt = _to_binary_mask(ground_truth, threshold)
    pred = _to_binary_mask(prediction, threshold)

    intersection = np.sum(gt * pred)

    union = np.sum(gt) + np.sum(pred) - intersection

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return float(iou)


def calculate_precision(
    ground_truth: ArrayLike,
    prediction: ArrayLike,
    threshold: float = 0.5,
    smooth: float = 1e-7
) -> float:
    """
    Tính Precision.

    Precision = TP / (TP + FP)
    """

    gt = _to_binary_mask(ground_truth, threshold)
    pred = _to_binary_mask(prediction, threshold)

    true_positive = np.sum(gt * pred)

    false_positive = np.sum(
        (1 - gt) * pred
    )

    precision = (
        true_positive + smooth
    ) / (
        true_positive + false_positive + smooth
    )

    return float(precision)


def calculate_recall(
    ground_truth: ArrayLike,
    prediction: ArrayLike,
    threshold: float = 0.5,
    smooth: float = 1e-7
) -> float:
    """
    Tính Recall.

    Recall = TP / (TP + FN)
    """

    gt = _to_binary_mask(ground_truth, threshold)
    pred = _to_binary_mask(prediction, threshold)

    true_positive = np.sum(gt * pred)

    false_negative = np.sum(
        gt * (1 - pred)
    )

    recall = (
        true_positive + smooth
    ) / (
        true_positive + false_negative + smooth
    )

    return float(recall)


def calculate_f1_score(
    ground_truth: ArrayLike,
    prediction: ArrayLike,
    threshold: float = 0.5,
    smooth: float = 1e-7
) -> float:
    """
    Tính F1-score.

    F1 = 2 * Precision * Recall / (Precision + Recall)
    """

    precision = calculate_precision(
        ground_truth,
        prediction,
        threshold,
        smooth
    )

    recall = calculate_recall(
        ground_truth,
        prediction,
        threshold,
        smooth
    )

    f1 = (
        2.0 * precision * recall
    ) / (
        precision + recall + smooth
    )

    return float(f1)


def calculate_accuracy(
    ground_truth: ArrayLike,
    prediction: ArrayLike,
    threshold: float = 0.5
) -> float:
    """
    Tính Pixel Accuracy.

    Accuracy = số pixel dự đoán đúng / tổng số pixel
    """

    gt = _to_binary_mask(ground_truth, threshold)
    pred = _to_binary_mask(prediction, threshold)

    correct_pixels = np.sum(gt == pred)
    total_pixels = gt.size

    if total_pixels == 0:
        return 0.0

    return float(correct_pixels / total_pixels)


def calculate_segmentation_metrics(
    ground_truth: ArrayLike,
    prediction: ArrayLike,
    threshold: float = 0.5
) -> dict:
    """
    Tính toàn bộ metrics cho segmentation.

    Parameters
    ----------
    ground_truth : array-like
        Mask thật.

    prediction : array-like
        Mask dự đoán.

    threshold : float
        Ngưỡng binary mask.

    Returns
    -------
    dict
        Dictionary chứa toàn bộ metrics.
    """

    dice = calculate_dice(
        ground_truth,
        prediction,
        threshold
    )

    iou = calculate_iou(
        ground_truth,
        prediction,
        threshold
    )

    precision = calculate_precision(
        ground_truth,
        prediction,
        threshold
    )

    recall = calculate_recall(
        ground_truth,
        prediction,
        threshold
    )

    f1 = calculate_f1_score(
        ground_truth,
        prediction,
        threshold
    )

    accuracy = calculate_accuracy(
        ground_truth,
        prediction,
        threshold
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy
    }


def print_segmentation_metrics(metrics: dict) -> None:
    """
    In kết quả segmentation metrics.
    """

    print("\n" + "=" * 50)
    print("SEGMENTATION EVALUATION RESULTS")
    print("=" * 50)

    print(f"Dice      : {metrics['dice']:.4f}")
    print(f"IoU       : {metrics['iou']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1_score']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")

    print("=" * 50)


if __name__ == "__main__":

    # Tạo dữ liệu mẫu để kiểm tra file
    ground_truth = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ])

    prediction = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ])

    metrics = calculate_segmentation_metrics(
        ground_truth,
        prediction
    )

    print_segmentation_metrics(metrics)

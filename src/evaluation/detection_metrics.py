"""
Detection metrics for Faster R-CNN.

Các chỉ số:
- IoU (Intersection over Union)
- Precision
- Recall
- F1-score
- Mean IoU
"""

from typing import List, Optional, Tuple


def calculate_iou(
    box1: List[float],
    box2: List[float]
) -> float:
    """
    Tính Intersection over Union (IoU) giữa hai bounding box.

    Bounding box có dạng:
    [x_min, y_min, x_max, y_max]

    Parameters
    ----------
    box1 : list
        Bounding box thứ nhất.
    box2 : list
        Bounding box thứ hai.

    Returns
    -------
    float
        Giá trị IoU từ 0 đến 1.
    """

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Tọa độ phần giao nhau
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Chiều rộng và chiều cao phần giao
    inter_width = max(0.0, inter_x_max - inter_x_min)
    inter_height = max(0.0, inter_y_max - inter_y_min)

    intersection_area = inter_width * inter_height

    # Diện tích box 1
    area1 = max(0.0, x1_max - x1_min) * max(
        0.0, y1_max - y1_min
    )

    # Diện tích box 2
    area2 = max(0.0, x2_max - x2_min) * max(
        0.0, y2_max - y2_min
    )

    # Union
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def calculate_precision(
    true_positive: int,
    false_positive: int
) -> float:
    """
    Tính Precision.

    Precision = TP / (TP + FP)
    """

    denominator = true_positive + false_positive

    if denominator == 0:
        return 0.0

    return true_positive / denominator


def calculate_recall(
    true_positive: int,
    false_negative: int
) -> float:
    """
    Tính Recall.

    Recall = TP / (TP + FN)
    """

    denominator = true_positive + false_negative

    if denominator == 0:
        return 0.0

    return true_positive / denominator


def calculate_f1_score(
    precision: float,
    recall: float
) -> float:
    """
    Tính F1-score.

    F1 = 2 * Precision * Recall / (Precision + Recall)
    """

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def calculate_mean_iou(ious: List[float]) -> float:
    """
    Tính Mean IoU.

    Parameters
    ----------
    ious : list
        Danh sách IoU của các ảnh.

    Returns
    -------
    float
        Giá trị Mean IoU.
    """

    if not ious:
        return 0.0

    return sum(ious) / len(ious)


def calculate_detection_metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    ious: Optional[List[float]] = None
) -> dict:
    """
    Tính toàn bộ metrics cho bài toán detection.

    Parameters
    ----------
    true_positive : int
        Số prediction đúng.

    false_positive : int
        Số prediction sai.

    false_negative : int
        Số object thực tế nhưng model không phát hiện.

    ious : list, optional
        Danh sách IoU.

    Returns
    -------
    dict
        Dictionary chứa các metrics.
    """

    if ious is None:
        ious = []

    precision = calculate_precision(
        true_positive,
        false_positive
    )

    recall = calculate_recall(
        true_positive,
        false_negative
    )

    f1 = calculate_f1_score(
        precision,
        recall
    )

    mean_iou = calculate_mean_iou(ious)

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_iou": mean_iou,
        "num_samples": len(ious)
    }


def print_detection_metrics(metrics: dict) -> None:
    """
    In metrics ra màn hình theo dạng dễ đọc.
    """

    print("\n" + "=" * 50)
    print("DETECTION EVALUATION RESULTS")
    print("=" * 50)

    print(f"True Positive  : {metrics['true_positive']}")
    print(f"False Positive : {metrics['false_positive']}")
    print(f"False Negative : {metrics['false_negative']}")

    print("-" * 50)

    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1-score       : {metrics['f1_score']:.4f}")
    print(f"Mean IoU       : {metrics['mean_iou']:.4f}")

    print("=" * 50)


if __name__ == "__main__":

    # Dữ liệu mẫu để kiểm tra file
    test_ious = [
        0.80,
        0.75,
        0.90,
        0.65
    ]

    metrics = calculate_detection_metrics(
        true_positive=8,
        false_positive=2,
        false_negative=1,
        ious=test_ious
    )

    print_detection_metrics(metrics)

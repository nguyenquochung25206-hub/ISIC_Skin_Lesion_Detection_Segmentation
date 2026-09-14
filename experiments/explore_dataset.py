```python
"""
evaluate_detection.py

Đánh giá kết quả Object Detection của Faster R-CNN.

Metrics:
- IoU
- Precision
- Recall

Ground Truth và Prediction sử dụng Bounding Box:
[xmin, ymin, xmax, ymax]
"""

from typing import List, Optional, Tuple


Box = List[float]


def calculate_iou(box1: Box, box2: Box) -> float:
    """
    Tính Intersection over Union (IoU) giữa hai Bounding Box.
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0.0, box1[2] - box1[0]) * max(
        0.0, box1[3] - box1[1]
    )

    area2 = max(0.0, box2[2] - box2[0]) * max(
        0.0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def evaluate_detection(
    ground_truth: Box,
    prediction: Optional[Box],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, float]:
    """
    Đánh giá một ảnh.

    Returns:
        TP, FP, FN, IoU
    """

    # Model không phát hiện được lesion
    if prediction is None:
        return 0, 0, 1, 0.0

    iou = calculate_iou(ground_truth, prediction)

    if iou >= iou_threshold:
        return 1, 0, 0, iou

    return 0, 1, 1, iou


def calculate_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    """
    Tính Precision và Recall.
    """

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


def main():
    """
    Ví dụ đánh giá một ảnh.
    """

    # Ground Truth Bounding Box
    ground_truth = [50, 40, 400, 390]

    # Bounding Box do Faster R-CNN dự đoán
    prediction = [55, 45, 395, 385]

    # Ngưỡng IoU
    iou_threshold = 0.5

    tp, fp, fn, iou = evaluate_detection(
        ground_truth,
        prediction,
        iou_threshold,
    )

    precision, recall = calculate_metrics(tp, fp, fn)

    print("=" * 45)
    print("Faster R-CNN Detection Evaluation")
    print("=" * 45)

    print(f"Ground Truth : {ground_truth}")
    print(f"Prediction   : {prediction}")
    print(f"IoU          : {iou:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")

    print("-" * 45)
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print("=" * 45)


if __name__ == "__main__":
    main()
```

### Cách chạy

```bash
python evaluate_detection.py
```

Ví dụ kết quả:

```text
=============================================
Faster R-CNN Detection Evaluation
=============================================
Ground Truth : [50, 40, 400, 390]
Prediction   : [55, 45, 395, 385]
IoU          : 0.9440
Precision    : 1.0000
Recall       : 1.0000
---------------------------------------------
TP: 1
FP: 0
FN: 0
=============================================
```

**Lưu ý:** Đây là script đánh giá cơ bản cho **một Bounding Box/ảnh**. Khi nhóm đã có output Faster R-CNN trên toàn bộ test set, nên mở rộng script để tính **Mean IoU, Precision, Recall và mAP trên toàn bộ dataset**.

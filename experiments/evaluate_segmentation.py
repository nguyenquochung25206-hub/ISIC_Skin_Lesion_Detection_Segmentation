
"""
evaluate_segmentation.py

Đánh giá kết quả Image Segmentation của U-Net.

Metrics:
- Dice
- IoU
- Precision
- Recall

Mask:
- Ground Truth: mask chuẩn
- Prediction: mask dự đoán
"""

from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: str) -> np.ndarray:
    """
    Đọc mask và chuyển thành binary mask 0/1.
    """

    mask = np.array(Image.open(path).convert("L"))

    # Pixel > 0 được xem là lesion
    return mask > 0


def calculate_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
):
    """
    Tính Dice, IoU, Precision và Recall.
    """

    # Chuyển về Boolean
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)

    # TP, FP, FN
    tp = np.logical_and(ground_truth, prediction).sum()
    fp = np.logical_and(~ground_truth, prediction).sum()
    fn = np.logical_and(ground_truth, ~prediction).sum()

    # Dice
    dice_denominator = 2 * tp + fp + fn

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / dice_denominator

    # IoU
    iou_denominator = tp + fp + fn

    if iou_denominator == 0:
        iou = 1.0
    else:
        iou = tp / iou_denominator

    # Precision
    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = tp / precision_denominator

    # Recall
    recall_denominator = tp + fn

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = tp / recall_denominator

    return dice, iou, precision, recall


def evaluate_image(
    ground_truth_path: str,
    prediction_path: str,
):
    """
    Đánh giá một cặp Ground Truth / Prediction.
    """

    ground_truth = load_mask(ground_truth_path)
    prediction = load_mask(prediction_path)

    if ground_truth.shape != prediction.shape:
        raise ValueError(
            "Ground Truth và Prediction phải có cùng kích thước."
        )

    return calculate_metrics(ground_truth, prediction)


def main():
    """
    Ví dụ đánh giá một ảnh.
    """

    # Thay bằng đường dẫn mask thực tế
    ground_truth_path = "data/masks/test/ISIC_001.png"
    prediction_path = "results/segmentation/predictions/ISIC_001.png"

    if not Path(ground_truth_path).exists():
        print(f"Không tìm thấy Ground Truth: {ground_truth_path}")
        return

    if not Path(prediction_path).exists():
        print(f"Không tìm thấy Prediction: {prediction_path}")
        return

    dice, iou, precision, recall = evaluate_image(
        ground_truth_path,
        prediction_path,
    )

    print("=" * 45)
    print("U-Net Segmentation Evaluation")
    print("=" * 45)

    print(f"Ground Truth : {ground_truth_path}")
    print(f"Prediction   : {prediction_path}")

    print("-" * 45)
    print(f"Dice         : {dice:.4f}")
    print(f"IoU          : {iou:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")

    print("=" * 45)


if __name__ == "__main__":
    main()


#**Lưu ý:** File này hiện đánh giá **một ảnh**. Khi chạy project thật, nên mở rộng thành đánh giá toàn bộ `test/` và xuất bảng `CSV` gồm từng ảnh + **Mean Dice / Mean IoU / Mean Precision / Mean Recall** để đưa trực tiếp vào `segmentation_results.md`.
